# =============================================================================
# src/jarvis/common/notifications.py - the Notification model
# =============================================================================
#
# A notification is JARVIS speaking to the owner WITHOUT being asked:
# finished work, a raised approval gate, a watcher hit. Every one is a
# durable row before it is a delivered message, so a failed send is a
# retryable pending row rather than a lost message.
#
# priority is carried but not yet acted on: the phase-2 policy delivers
# everything immediately. Quiet hours, digests, and priority routing
# arrive with the initiative phase and will read this same field.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now


class NotificationStatus(StrEnum):
    PENDING = "pending"        # created, not yet delivered
    DELIVERED = "delivered"    # reached the owner
    SUPPRESSED = "suppressed"  # policy or missing client decided not to send


class Notification(BaseModel):
    """One unprompted message to the owner."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    ts: datetime = Field(default_factory=utc_now)
    priority: int = Field(default=5, ge=0, le=9)   # 0 urgent .. 9 idle
    status: NotificationStatus = NotificationStatus.PENDING
    client_kind: str                    # which surface this is meant for
    text: str
    session_id: str | None = None       # conversation it belongs to
    job_id: str | None = None           # work it reports on
    artifact_id: str | None = None      # file to deliver alongside
    delivered_ts: datetime | None = None
    suppress_reason: str | None = None
    trace_id: str

    @field_validator("id", "trace_id")
    @classmethod
    def _ids_are_ulids(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("ts", "delivered_ts")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("notification timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)