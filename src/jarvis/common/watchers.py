# =============================================================================
# src/jarvis/common/watchers.py - the Watcher model
# =============================================================================
#
# A watcher is a RECURRING CHECK WITH A MEMORY. It runs on a schedule,
# looks at something, compares against what it saw last time, and speaks
# only when something changed.
#
# The memory is the whole idea. A check without state either reports
# every time (noise the owner mutes) or never (silence he cannot trust).
# `state` holds whatever the check needs to answer "is this different
# from before" - a page hash, a count, a timestamp - and is written back
# after every run.
#
# What a watcher CANNOT do: run continuously, react instantly, or reach
# anything needing a login. A watcher checking every fifteen minutes
# finds out within fifteen minutes, and that is the honest trade.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now


class WatcherKind(StrEnum):
    """What sort of thing is being watched. Each maps to a check
    implementation; adding a kind means adding one function."""

    WEB_PAGE = "web_page"        # a URL's content
    JOB_HEALTH = "job_health"    # failures of a job type
    SPEND = "spend"              # accumulated cost against a threshold
    IDLE = "idle"                # something NOT happening for too long


class Watcher(BaseModel):
    """One standing thing to keep an eye on."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    name: str                        # human label, unique
    kind: WatcherKind

    # What to check - shape depends on kind. A web_page watcher carries
    # {"url": ...}; a spend watcher carries {"threshold_inr": ...}.
    config: dict[str, Any] = Field(default_factory=dict)

    # What the last check saw. The comparison basis for "has this
    # changed", and the reason a watcher is not merely a schedule.
    state: dict[str, Any] = Field(default_factory=dict)

    # How urgently a hit reaches the owner. Most watchers should be
    # quiet - a page changing is rarely worth waking someone.
    priority: int = Field(default=5, ge=0, le=9)

    # What the owner is actually asking about, in his own words. Sent
    # with the notification so a hit six weeks later still makes sense.
    note: str = ""

    enabled: bool = True
    hit_count: int = Field(default=0, ge=0)
    last_checked_ts: datetime | None = None
    last_hit_ts: datetime | None = None
    created_ts: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def _id_is_ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("last_checked_ts", "last_hit_ts", "created_ts")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("watcher timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)
