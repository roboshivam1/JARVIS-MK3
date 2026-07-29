# =============================================================================
# src/jarvis/common/jobs.py — Job model, statuses, state machine (doc 02 §4)
# =============================================================================
#
# A Job is the durable unit of work: persisted before acknowledged
# (durability rule 1), leased to workers, retried with backoff, resumable
# via checkpoints, gate-able for approval. This module defines the SHAPE
# and the LEGAL STATE MACHINE; the queue's behaviour (leasing, reclaiming,
# dispatching) is Phase 2 code in core/queue/.
#
# ALLOWED_TRANSITIONS is doc 02 §4.2's diagram, transcribed as data:
#   - tested against the doc in tests (a mismatch is a failing test),
#   - consulted by the Phase-2 queue before every UPDATE,
#   - paired with SQL optimistic checks (WHERE status = expected) so the
#     check and the write are atomic even if two coroutines race.
# The map answers "is this edge legal at all"; edge CONDITIONS (attempts
# exhausted? approval granted?) are queue logic, deliberately not encoded
# here.
#
# Typed sub-models: doc 02 types `lease` and `approval` as dict|null with
# their fields listed in prose. We formalise those exact fields as Lease
# and Approval models (owner-flagged clarification) — same wire shape,
# but validated.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now


class JobStatus(StrEnum):
    """Job lifecycle states (doc 02 §4.1). StrEnum: members ARE their
    string values, so they bind directly into SQL and serialise cleanly."""

    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)

# doc 02 §4.2, edge for edge. Empty set = terminal (no way out).
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({
        JobStatus.LEASED,             # dispatcher assigns to a worker
        JobStatus.CANCELLED,          # owner cancels while waiting
    }),
    JobStatus.LEASED: frozenset({
        JobStatus.RUNNING,            # worker confirms start
        JobStatus.QUEUED,             # lease/heartbeat expiry → reclaim
        JobStatus.CANCELLED,          # owner cancels
    }),
    JobStatus.RUNNING: frozenset({
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,             # attempts exhausted → terminal…
        JobStatus.QUEUED,             # …else requeue with backoff, or
                                      #    heartbeat expiry mid-run
        JobStatus.AWAITING_APPROVAL,  # job raised a gate
        JobStatus.CANCELLED,          # owner cancels
    }),
    JobStatus.AWAITING_APPROVAL: frozenset({
        JobStatus.RUNNING,            # approved → resumes
        JobStatus.CANCELLED,          # rejected, or gate TTL expired
    }),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


def is_legal_transition(current: JobStatus, new: JobStatus) -> bool:
    """The single yes/no authority on job status edges. Phase-2 queue code
    calls this before every UPDATE; anything else mutating status is a
    layering violation."""
    return new in ALLOWED_TRANSITIONS[current]


class Lease(BaseModel):
    """Who is holding this job right now, and until when (doc 02 §4.2).

    ttl_s counts from heartbeat_ts, not leased_ts: each heartbeat extends
    the lease. heartbeat_ts + ttl_s < now  ⇒  the Core may reclaim.
    """

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    leased_ts: datetime
    heartbeat_ts: datetime
    ttl_s: int = Field(gt=0)

    @field_validator("leased_ts", "heartbeat_ts")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("lease timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)


class Approval(BaseModel):
    """An approval gate raised on this job (doc 02 §4.1 prose fields).
    resolved_ts/decision stay None while the owner hasn't answered."""

    model_config = ConfigDict(extra="forbid")

    gate: str                              # gate name from doc 06 §3 taxonomy
    requested_ts: datetime
    resolved_ts: datetime | None = None
    decision: str | None = None            # "approve" | "reject" | None (pending)
    note: str | None = None                # owner's optional reason

    @field_validator("requested_ts", "resolved_ts")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("approval timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)

    @field_validator("decision")
    @classmethod
    def _decision_vocab(cls, v: str | None) -> str | None:
        if v is not None and v not in ("approve", "reject"):
            raise ValueError(f"decision must be approve|reject, got {v!r}")
        return v


class Job(BaseModel):
    """One durable unit of work (doc 02 §4.1, field-exact)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    type: str                              # registered job type, e.g. "research.brief"
    status: JobStatus = JobStatus.QUEUED
    priority: int = Field(default=5, ge=0, le=9)   # 0 urgent … 9 idle backlog
    requires: list[str] = Field(default_factory=list)  # [] ⇒ Core's built-in worker
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    session_id: str | None = None          # conversation to attach the result to
    parent_job_id: str | None = None       # job chaining
    approval: Approval | None = None
    checkpoint: dict[str, Any] | None = None   # job-type-defined resume state
    attempts: int = Field(default=0, ge=0)     # incremented on each LEASE
    max_attempts: int = Field(default=3, ge=1)
    lease: Lease | None = None
    not_before: datetime | None = None     # scheduling / retry backoff
    created_ts: datetime = Field(default_factory=utc_now)
    updated_ts: datetime = Field(default_factory=utc_now)
    finished_ts: datetime | None = None
    trace_id: str                          # required — every job has a cause

    # ── field validators ─────────────────────────────────────────────────────

    @field_validator("id", "trace_id")
    @classmethod
    def _ids_are_ulids(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("session_id", "parent_job_id")
    @classmethod
    def _optional_ids_are_ulids(cls, v: str | None) -> str | None:
        if v is not None and not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("artifacts")
    @classmethod
    def _artifact_ids_are_ulids(cls, v: list[str]) -> list[str]:
        for a in v:
            if not is_ulid(a):
                raise ValueError(f"artifact id is not a ULID: {a!r}")
        return v

    @field_validator("type")
    @classmethod
    def _type_shape(cls, v: str) -> str:
        # Same dotted-lowercase convention as envelope kinds ("research.brief").
        # Whether the type is REGISTERED is the Phase-2 registry's concern;
        # shape is ours.
        parts = v.split(".")
        if len(parts) < 2 or not all(
            p and all(c.islower() or c.isdigit() or c == "_" for c in p)
            for p in parts
        ):
            raise ValueError(f"job type must be dotted lowercase, got {v!r}")
        return v

    @field_validator("created_ts", "updated_ts", "finished_ts", "not_before")
    @classmethod
    def _timestamps_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("job timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)

    # ── cross-field invariants ───────────────────────────────────────────────

    @model_validator(mode="after")
    def _coherent_state(self) -> "Job":
        # A leased/running job must know who holds it; a waiting one must not.
        if self.status in (JobStatus.LEASED, JobStatus.RUNNING) and self.lease is None:
            raise ValueError(f"a {self.status} job must carry a lease")
        if self.status == JobStatus.QUEUED and self.lease is not None:
            raise ValueError("a queued job must not carry a lease")

        # Gates: the status and the record travel together.
        if self.status == JobStatus.AWAITING_APPROVAL and self.approval is None:
            raise ValueError("awaiting_approval requires an approval record")

        # Terminal jobs are finished; live jobs are not.
        if self.status in TERMINAL_STATUSES and self.finished_ts is None:
            raise ValueError(f"a {self.status} job must have finished_ts")
        if self.status not in TERMINAL_STATUSES and self.finished_ts is not None:
            raise ValueError("finished_ts on a non-terminal job")

        # Failure semantics: error text accompanies failure, success has none.
        if self.status == JobStatus.FAILED and not self.error:
            raise ValueError("a failed job must say why (error is empty)")
        if self.status == JobStatus.SUCCEEDED and self.error:
            raise ValueError("a succeeded job must not carry an error")

        return self