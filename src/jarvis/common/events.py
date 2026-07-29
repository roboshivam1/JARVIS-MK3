# =============================================================================
# src/jarvis/common/events.py — Event model + kind taxonomy (doc 02 §3)
# =============================================================================
#
# One Event = one immutable row in the append-only event log. The log is:
#   - the system's spine: the in-process bus delivers committed events to
#     subscribers (Batch 3 builds the storage; Phase 1 the bus),
#   - the episodic memory, verbatim and forever (doc 04 §1.1),
#   - never UPDATEd, never DELETEd (durability rule 5).
#
# Taxonomy discipline: event kinds are a CLOSED vocabulary (constants below,
# extended only via doc amendment). We are the only writer of our own log,
# so an unrecognised kind at write time is a bug, and the model rejects it.
# Contrast envelope kinds, which stay OPEN at parse time because the other
# end of a socket may be newer than us.
#
# trace_id is REQUIRED (not optional as on Envelope): every event belongs to
# some causal story (doc 06 §4.1). An untraceable event is unconstructible.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now


class EventKind:
    """The event vocabulary, verbatim from doc 02 §3. Grouped by prefix.
    Extend only alongside a doc amendment — the doc and this class must
    never disagree."""

    # core.* — daemon lifecycle
    CORE_STARTED = "core.started"
    CORE_RECOVERED = "core.recovered"

    # session.* — conversation lifecycle
    SESSION_OPENED = "session.opened"
    SESSION_TURN_USER = "session.turn_user"
    SESSION_TURN_ASSISTANT = "session.turn_assistant"
    SESSION_INTERRUPTED = "session.interrupted"

    # job.* — queue state machine transitions
    JOB_ENQUEUED = "job.enqueued"
    JOB_LEASED = "job.leased"
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_CHECKPOINT = "job.checkpoint"
    JOB_AWAITING_APPROVAL = "job.awaiting_approval"
    JOB_APPROVED = "job.approved"
    JOB_REJECTED = "job.rejected"
    JOB_SUCCEEDED = "job.succeeded"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"

    # worker.* — fleet membership
    WORKER_CONNECTED = "worker.connected"
    WORKER_DISCONNECTED = "worker.disconnected"
    WORKER_HEARTBEAT_MISSED = "worker.heartbeat_missed"

    # memory.* — memory system activity
    MEMORY_FACT_ADDED = "memory.fact_added"
    MEMORY_FACT_UPDATED = "memory.fact_updated"
    MEMORY_PROFILE_UPDATED = "memory.profile_updated"
    MEMORY_SLEEP_CYCLE_COMPLETED = "memory.sleep_cycle_completed"

    # initiative.* — proactivity
    INITIATIVE_SCHEDULE_FIRED = "initiative.schedule_fired"
    INITIATIVE_WATCHER_HIT = "initiative.watcher_hit"
    INITIATIVE_NOTIFICATION_SENT = "initiative.notification_sent"
    INITIATIVE_NOTIFICATION_SUPPRESSED = "initiative.notification_suppressed"

    # llm.* — the trace backbone
    LLM_CALL_COMPLETED = "llm.call_completed"


# Built once: every string constant defined on EventKind. This frozenset is
# what the Event model validates against.
ALL_EVENT_KINDS: frozenset[str] = frozenset(
    v for k, v in vars(EventKind).items()
    if not k.startswith("_") and isinstance(v, str)
)


class Event(BaseModel):
    """One immutable record: something happened (doc 02 §3, field-exact)."""

    # frozen: instances are immutable after construction, mirroring the
    # append-only log — code that tries to mutate an Event in flight gets
    # an immediate error instead of a silent divergence from what was stored.
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=new_ulid)
    ts: datetime = Field(default_factory=utc_now)
    kind: str
    source: str                       # "core.orchestrator", "worker.macbook", ...
    session_id: str | None = None
    job_id: str | None = None
    trace_id: str                     # REQUIRED — see module header
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "trace_id")
    @classmethod
    def _ids_are_ulids(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("session_id", "job_id")
    @classmethod
    def _optional_ids_are_ulids(cls, v: str | None) -> str | None:
        if v is not None and not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("ts")
    @classmethod
    def _ts_is_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("event ts must be timezone-aware (UTC)")
        return v.astimezone(timezone.utc)

    @field_validator("kind")
    @classmethod
    def _kind_in_taxonomy(cls, v: str) -> str:
        if v not in ALL_EVENT_KINDS:
            raise ValueError(
                f"unknown event kind {v!r} — the taxonomy is closed; "
                f"extend EventKind alongside a doc-02 amendment"
            )
        return v

    @field_validator("source")
    @classmethod
    def _source_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("event source must be non-empty")
        return v