# =============================================================================
# src/jarvis/common/approvals.py - the ApprovalRequest model
# =============================================================================
#
# One request for permission: what was going to happen, who wanted to do
# it, why it needed asking, and what the owner decided.
#
# Two fields carry the weight:
#
#   detail - the EXACT action, in full. Not "send an email" but the
#     actual recipient, subject, and body. An approval prompt that
#     summarises away the specifics trains the owner to tap yes without
#     reading, which is worse than having no gate at all.
#
#   expires_ts - an unanswered request cannot pause a job forever. Past
#     this moment the job fails honestly rather than lurking as a
#     zombie in the queue.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.common.capabilities import Gate
from jarvis.common.ids import is_ulid, new_ulid, utc_now


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"      # nobody answered in time


class ApprovalRequest(BaseModel):
    """One question put to the owner before work continues."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    job_id: str
    gate: Gate
    actor: str                      # which agent asked
    tool: str                       # what it wanted to run
    summary: str                    # one line, for a notification
    detail: str                     # the exact action, in full
    risk_note: str = ""             # what could go wrong
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision_note: str | None = None
    requested_ts: datetime = Field(default_factory=utc_now)
    expires_ts: datetime
    resolved_ts: datetime | None = None
    trace_id: str

    @field_validator("id", "job_id", "trace_id")
    @classmethod
    def _ids_are_ulids(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("requested_ts", "expires_ts", "resolved_ts")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("approval timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)

    @property
    def is_open(self) -> bool:
        return self.status is ApprovalStatus.PENDING