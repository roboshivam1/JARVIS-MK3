# =============================================================================
# src/jarvis/core/approvals/service.py - pausing work to ask the owner
# =============================================================================
#
# The bridge between "the guard said this needs permission" and "the
# owner tapped approve".
#
# WHY THE JOB PAUSES INSTEAD OF THE TOOL WAITING: a tool call could
# simply block until an answer arrives. But the owner may be asleep. A
# blocked job holds its slot, burns its lease, and dies with the process.
# So the job CHECKPOINTS and STOPS - status awaiting_approval - and
# resumes as a fresh attempt when the answer comes, possibly hours later
# and possibly on a different machine. Human latency gets the same
# durability treatment as machine failure.
#
# Three ways a request ends:
#   approved - job returns to queued and resumes from its checkpoint
#   rejected - job is cancelled; the owner said no and meant it
#   expired  - nobody answered in time; the job fails honestly rather
#              than lurking in the queue as a zombie
# =============================================================================

from __future__ import annotations

import asyncio
from datetime import timedelta

from jarvis.common.approvals import ApprovalRequest, ApprovalStatus
from jarvis.common.capabilities import Gate
from jarvis.common.events import Event, EventKind
from jarvis.common.ids import utc_now
from jarvis.common.jobs import Approval, JobStatus
from jarvis.common.log import get_logger
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo

log = get_logger("core.approvals")

SOURCE = "core.approvals"
_EXPIRY_SWEEP_INTERVAL_S = 60.0
DEFAULT_TTL_HOURS = 24


class ApprovalService:
    """Raises gates, records decisions, and resumes or kills the work."""

    def __init__(
        self,
        approvals: ApprovalsRepo,
        jobs: JobsRepo,
        events: EventsRepo,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> None:
        self._approvals = approvals
        self._jobs = jobs
        self._events = events
        self._ttl = timedelta(hours=ttl_hours)

    # -- raising --------------------------------------------------------------

    async def request(
        self,
        job_id: str,
        gate: Gate,
        actor: str,
        tool: str,
        summary: str,
        detail: str,
        risk_note: str = "",
    ) -> ApprovalRequest | None:
        """Pause a running job and ask the owner. Returns None if the job
        could not be paused (already finished, cancelled, or moved by
        someone else) - losing that race means the question is moot."""
        job = await self._jobs.get(job_id)
        if job is None or job.status is not JobStatus.RUNNING:
            log.warning("cannot gate a job that is not running", extra={
                "job_id": job_id,
                "status": job.status.value if job else "missing",
            })
            return None

        now = utc_now()
        request = ApprovalRequest(
            job_id=job_id, gate=gate, actor=actor, tool=tool,
            summary=summary, detail=detail, risk_note=risk_note,
            requested_ts=now, expires_ts=now + self._ttl,
            trace_id=job.trace_id,
        )
        await self._approvals.create(request)

        # The job's own approval field carries the CURRENT gate; the
        # model validator requires it in this status.
        moved = await self._jobs.transition(
            job_id, JobStatus.RUNNING, JobStatus.AWAITING_APPROVAL,
            set_fields={
                "approval": Approval(gate=gate.value, requested_ts=now),
                "lease": None,   # release the executor; this may take hours
            },
        )
        if not moved:
            # Someone finished or cancelled the job in the meantime.
            await self._approvals.resolve(
                request.id, ApprovalStatus.EXPIRED, "job moved on"
            )
            return None

        await self._events.append(Event(
            kind=EventKind.JOB_AWAITING_APPROVAL,
            source=SOURCE, job_id=job_id, session_id=job.session_id,
            trace_id=job.trace_id,
            payload={
                "approval_id": request.id, "gate": gate.value,
                "actor": actor, "tool": tool, "summary": summary,
            },
        ))
        return request

    # -- answering ------------------------------------------------------------

    async def decide(
        self,
        approval_id: str,
        approve: bool,
        note: str | None = None,
    ) -> bool:
        """Record the owner's answer and move the job accordingly."""
        request = await self._approvals.get(approval_id)
        if request is None or not request.is_open:
            return False

        status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
        if not await self._approvals.resolve(approval_id, status, note):
            return False   # expiry sweep got there first

        job = await self._jobs.get(request.job_id)
        if job is None:
            return True    # decision recorded; the job is gone

        if approve:
            # Back to the queue: the next attempt resumes from the
            # checkpoint the handler saved before pausing.
            await self._jobs.transition(
                job.id, JobStatus.AWAITING_APPROVAL, JobStatus.QUEUED,
                set_fields={"approval": None},
            )
            kind = EventKind.JOB_APPROVED
        else:
            await self._jobs.transition(
                job.id, JobStatus.AWAITING_APPROVAL, JobStatus.CANCELLED,
                set_fields={"error": "rejected by owner"},
            )
            kind = EventKind.JOB_REJECTED

        await self._events.append(Event(
            kind=kind, source=SOURCE, job_id=job.id,
            session_id=job.session_id, trace_id=job.trace_id,
            payload={"approval_id": approval_id, "note": note},
        ))
        log.info("approval decided", extra={
            "approval_id": approval_id, "approved": approve,
            "job_id": job.id,
        })
        return True

    # -- expiry ---------------------------------------------------------------

    async def sweep_expired(self) -> int:
        """Fail jobs whose gates went unanswered. Returns how many."""
        count = 0
        for request in await self._approvals.expired():
            if not await self._approvals.resolve(
                request.id, ApprovalStatus.EXPIRED, "no answer in time"
            ):
                continue   # the owner answered in the same moment; they win

            job = await self._jobs.get(request.job_id)
            if job is None or job.status is not JobStatus.AWAITING_APPROVAL:
                continue
            await self._jobs.transition(
                job.id, JobStatus.AWAITING_APPROVAL, JobStatus.CANCELLED,
                set_fields={
                    "error": "awaiting your say-so, sir - it timed out",
                    "approval": None,
                },
            )
            await self._events.append(Event(
                kind=EventKind.JOB_CANCELLED, source=SOURCE, job_id=job.id,
                session_id=job.session_id, trace_id=job.trace_id,
                payload={"reason": "approval expired",
                         "approval_id": request.id},
            ))
            count += 1
        return count

    async def run(self) -> None:
        """Resident loop: sweep expired approvals, forever."""
        while True:
            try:
                await self.sweep_expired()
            except Exception:
                log.error("approval expiry sweep failed", exc_info=True)
            await asyncio.sleep(_EXPIRY_SWEEP_INTERVAL_S)