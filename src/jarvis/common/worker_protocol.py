# =============================================================================
# src/jarvis/common/worker_protocol.py - the Core <-> worker wire vocabulary
# =============================================================================
#
# Every message on the worker link is an Envelope (doc 02 shape) whose
# payload is one of the models below, registered by kind. Shared by both
# sides: the Core imports this, the worker imports this, and neither can
# invent a message the other cannot parse.
#
# The connection is always dialled BY the worker. That is what makes a
# laptop behind a home router - or in a cafe - work with no port
# forwarding, no static address, and no firewall holes. The socket stays
# open afterwards, so the Core can push work down a connection it never
# had to establish.
#
# Offers are OFFERS: a worker may decline one, and a decline costs the
# job nothing (no attempt consumed), because declining is not failing.
# =============================================================================

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.envelope import register_kind

# -- worker -> core -----------------------------------------------------------

class WorkerHello(BaseModel):
    """First frame after connecting. Anything else first is a protocol
    error and the socket closes."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str                       # stable name, e.g. "macbook"
    token: str                           # shared secret, checked on arrival
    capabilities: list[str] = Field(default_factory=list)
    max_concurrency: int = Field(default=2, ge=1)
    version: str = "0.1.0"


class WorkerHeartbeat(BaseModel):
    """Sent on a fixed interval even when idle, so the Core can tell
    'quiet' from 'gone'. Lists the jobs still running, whose leases it
    extends."""

    model_config = ConfigDict(extra="forbid")

    running_job_ids: list[str] = Field(default_factory=list)


class WorkerJobAccept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str


class WorkerJobDecline(BaseModel):
    """Cannot run this after all. Costs the job no attempt."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    reason: str = ""


class WorkerJobStarted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str


class WorkerJobProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    note: str
    pct: float | None = None


class WorkerJobCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    checkpoint: dict[str, Any] = Field(default_factory=dict)


class WorkerNeedsApproval(BaseModel):
    """A gated action stopped this job; the owner must decide.

    Workers have no database, so they cannot create an approval request
    themselves - they describe what they want to do and the Core raises
    the gate. The job then pauses until the owner answers, possibly
    hours later and possibly resuming on a different worker.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    gate: str                    # outbound | publish | credential | ...
    actor: str                   # which subagent asked
    tool: str                    # what it wanted to run
    summary: str                 # one line for the notification
    detail: str                  # the EXACT action, in full
    risk_note: str = ""


class WorkerJobResult(BaseModel):
    """Terminal report. Must arrive AFTER every artifact it references is
    fully uploaded, so a crash cannot leave a succeeded job whose files
    are missing."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str                          # "succeeded" | "failed"
    result: dict[str, Any] | None = None
    error: str | None = None
    permanent: bool = False              # true = do not retry this


class WorkerArtifactBegin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    name: str
    mime: str
    size: int
    sha256: str


class WorkerArtifactChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    seq: int
    data_b64: str


class WorkerArtifactEnd(BaseModel):
    """The Core verifies size and checksum here and rejects mismatches -
    which is the whole reason the checksum travels with the artifact."""

    model_config = ConfigDict(extra="forbid")

    job_id: str


# -- core -> worker -----------------------------------------------------------

class CoreWelcome(BaseModel):
    """Registration confirmed. The worker configures its timers from
    these numbers rather than hard-coding them, so tuning is a Core-side
    change."""

    model_config = ConfigDict(extra="forbid")

    heartbeat_interval_s: int = 15
    lease_ttl_s: int = 90
    protocol_v: int = 1


class CoreJobOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    requires: list[str] = Field(default_factory=list)
    checkpoint: dict[str, Any] | None = None    # resume state, if any
    timeout_s: int = 300
    trace_id: str
    approval_granted: bool = False
    # Input files, base64, keyed by filename.
    #
    # The CONTENTS travel, not the ids: a worker has no database access
    # by design, so it cannot resolve an artifact id to bytes. Sending
    # ids and hoping meant the sandbox silently received nothing and the
    # agent, finding no file, invented an answer instead.
    input_files: dict[str, str] = Field(default_factory=dict)


class CoreJobCancel(BaseModel):
    """Stop this job now. Without it, cancellation would only take effect
    when the lease expired - a gap in the original design docs."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    reason: str = ""


class ProtocolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str


# -- kind registration --------------------------------------------------------

WORKER_HELLO = "worker.hello"
WORKER_HEARTBEAT = "worker.heartbeat"
WORKER_JOB_ACCEPT = "worker.job_accept"
WORKER_JOB_DECLINE = "worker.job_decline"
WORKER_JOB_STARTED = "worker.job_started"
WORKER_JOB_PROGRESS = "worker.job_progress"
WORKER_JOB_CHECKPOINT = "worker.job_checkpoint"
WORKER_JOB_RESULT = "worker.job_result"
WORKER_NEEDS_APPROVAL = "worker.needs_approval"
WORKER_ARTIFACT_BEGIN = "worker.artifact_begin"
WORKER_ARTIFACT_CHUNK = "worker.artifact_chunk"
WORKER_ARTIFACT_END = "worker.artifact_end"

CORE_WELCOME = "core.welcome"
CORE_JOB_OFFER = "core.job_offer"
CORE_JOB_CANCEL = "core.job_cancel"
ERROR_PROTOCOL = "error.protocol"

register_kind(WORKER_HELLO, WorkerHello)
register_kind(WORKER_HEARTBEAT, WorkerHeartbeat)
register_kind(WORKER_JOB_ACCEPT, WorkerJobAccept)
register_kind(WORKER_JOB_DECLINE, WorkerJobDecline)
register_kind(WORKER_JOB_STARTED, WorkerJobStarted)
register_kind(WORKER_JOB_PROGRESS, WorkerJobProgress)
register_kind(WORKER_JOB_CHECKPOINT, WorkerJobCheckpoint)
register_kind(WORKER_JOB_RESULT, WorkerJobResult)
register_kind(WORKER_NEEDS_APPROVAL, WorkerNeedsApproval)
register_kind(WORKER_ARTIFACT_BEGIN, WorkerArtifactBegin)
register_kind(WORKER_ARTIFACT_CHUNK, WorkerArtifactChunk)
register_kind(WORKER_ARTIFACT_END, WorkerArtifactEnd)

register_kind(CORE_WELCOME, CoreWelcome)
register_kind(CORE_JOB_OFFER, CoreJobOffer)
register_kind(CORE_JOB_CANCEL, CoreJobCancel)
register_kind(ERROR_PROTOCOL, ProtocolError)