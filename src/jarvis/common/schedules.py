# =============================================================================
# src/jarvis/common/schedules.py - the Schedule model
# =============================================================================
#
# A schedule is a standing instruction to do something when a time
# arrives: nightly memory maintenance, a periodic watcher check, a
# one-off reminder.
#
# It is a durable ROW, never an in-memory timer. Timers die with the
# process; rows survive restarts, and "what is due?" stays one indexed
# query rather than a scan through recomputed cron math.
#
# next_fire_ts is STORED rather than derived. Cron arithmetic runs once,
# just after a firing, and the hot path only compares timestamps.
#
# Schedules carry no trace_id: each FIRING mints its own, because each
# firing is a separate root cause. Turns, job completions, and schedule
# firings are the three things that start a causal story in this system.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now


class ScheduleKind(StrEnum):
    CRON = "cron"          # a cron expression, evaluated in the owner's timezone
    INTERVAL = "interval"  # every N seconds from the last firing
    ONCE = "once"          # fires exactly once, then disables itself


class Schedule(BaseModel):
    """A standing instruction bound to a time."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    name: str                              # human label, e.g. "nightly sleep cycle"
    kind: ScheduleKind

    # Exactly one of these matters, decided by kind.
    cron_expr: str | None = None           # e.g. "30 3 * * *" (03:30 daily)
    interval_s: int | None = Field(default=None, gt=0)

    # What firing does. Only job enqueueing exists today; scheduled
    # system turns ("remind me tomorrow" in JARVIS's own voice) arrive
    # with the initiative phase and become a second action kind here.
    job_type: str
    job_payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=0, le=9)

    enabled: bool = True
    next_fire_ts: datetime
    last_fired_ts: datetime | None = None
    fire_count: int = Field(default=0, ge=0)
    created_ts: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def _id_is_ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("next_fire_ts", "last_fired_ts", "created_ts")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("schedule timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _spec_matches_kind(self) -> "Schedule":
        # A cron schedule with no expression, or an interval schedule with
        # no interval, would sit in the table looking healthy and never
        # fire correctly. Catch it at construction.
        if self.kind is ScheduleKind.CRON and not self.cron_expr:
            raise ValueError("a cron schedule needs cron_expr")
        if self.kind is ScheduleKind.INTERVAL and not self.interval_s:
            raise ValueError("an interval schedule needs interval_s")
        return self