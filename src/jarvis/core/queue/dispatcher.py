# =============================================================================
# src/jarvis/core/queue/dispatcher.py - lease reclaim and the failure path
# =============================================================================
#
# Two things live here:
#
#   requeue_or_fail() - THE failure path, used by every executor and by
#     reclaim alike: attempts left -> back to queued with exponential
#     backoff (30s, 60s, 120s...); attempts spent -> failed, terminal.
#     One implementation so a timeout, a crash, and a vanished worker
#     all age a job identically.
#
#   ReclaimLoop - the janitor. Every few seconds: any leased/running job
#     whose lease expired belongs to a presumed-dead executor; reclaim it
#     through the same failure path. reclaim_all() is the boot-recovery
#     variant with a stricter rule: after a restart, EVERY in-flight job
#     is orphaned (our in-process tasks died with the process). When
#     remote workers arrive, that boot rule softens to expiry-based for
#     their jobs - flagged for the worker batch.
#
# Remote-worker matching (offers over WebSocket) lands in this file in a
# later batch; the name "dispatcher" is aspirational until then.
# =============================================================================

from __future__ import annotations

import asyncio

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import utc_now
from jarvis.common.jobs import Job, JobStatus
from jarvis.common.log import get_logger
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo

log = get_logger("core.queue.dispatcher")

SOURCE = "core.queue"
_BACKOFF_BASE_S = 30
_RECLAIM_INTERVAL_S = 15


async def requeue_or_fail(
    jobs: JobsRepo,
    events: EventsRepo,
    job: Job,
    error: str,
    *,
    expected: JobStatus,
) -> None:
    """Age a failed attempt: requeue with backoff, or fail for good.
    `expected` is the status we believe the job is in; a lost race means
    someone else (owner cancel, another reclaim) already handled it, and
    losing that race is fine - the job is in exactly one pair of hands."""
    if job.attempts >= job.max_attempts:
        moved = await jobs.transition(
            job.id, expected, JobStatus.FAILED,
            set_fields={"error": error, "lease": None},
        )
        if moved:
            await events.append(Event(
                kind=EventKind.JOB_FAILED,
                source=SOURCE, job_id=job.id, session_id=job.session_id,
                trace_id=job.trace_id,
                payload={"error": error, "terminal": True,
                         "attempts": job.attempts},
            ))
            log.warning("job failed terminally", extra={
                "job_id": job.id, "type": job.type, "error": error,
            })
        return

    backoff_s = _BACKOFF_BASE_S * (2 ** (job.attempts - 1))
    not_before = utc_now().timestamp() + backoff_s
    from datetime import datetime, timezone
    moved = await jobs.transition(
        job.id, expected, JobStatus.QUEUED,
        set_fields={
            "error": error,
            "lease": None,
            "not_before": datetime.fromtimestamp(not_before, tz=timezone.utc),
        },
    )
    if moved:
        await events.append(Event(
            kind=EventKind.JOB_FAILED,
            source=SOURCE, job_id=job.id, session_id=job.session_id,
            trace_id=job.trace_id,
            payload={"error": error, "terminal": False,
                     "attempts": job.attempts, "retry_in_s": backoff_s},
        ))
        log.info("job requeued with backoff", extra={
            "job_id": job.id, "attempt": job.attempts, "retry_in_s": backoff_s,
        })


class ReclaimLoop:
    """Rescues work from dead executors, forever."""

    def __init__(self, jobs: JobsRepo, events: EventsRepo) -> None:
        self._jobs = jobs
        self._events = events

    async def run(self) -> None:
        """The resident loop: sweep, sleep, repeat."""
        while True:
            try:
                await self.sweep_expired()
            except Exception:
                # The janitor must outlive its own bad days.
                log.error("reclaim sweep failed", exc_info=True)
            await asyncio.sleep(_RECLAIM_INTERVAL_S)

    async def sweep_expired(self) -> int:
        """Reclaim jobs whose lease is past its expiry. Returns count."""
        now = utc_now()
        reclaimed = 0
        for job in await self._jobs.live_jobs():
            if job.status not in (JobStatus.LEASED, JobStatus.RUNNING):
                continue
            if job.lease is None:
                continue   # model coherence makes this near-impossible
            expiry = job.lease.heartbeat_ts.timestamp() + job.lease.ttl_s
            if now.timestamp() > expiry:
                await requeue_or_fail(
                    self._jobs, self._events, job,
                    error=f"lease expired (executor {job.lease.worker_id} presumed dead)",
                    expected=job.status,
                )
                reclaimed += 1
        return reclaimed

    async def reclaim_all_inflight(self) -> int:
        """Boot recovery: every leased/running job is orphaned after a
        restart - in-process executor tasks died with the process. This
        rule softens for remote workers when they exist."""
        reclaimed = 0
        for job in await self._jobs.live_jobs():
            if job.status in (JobStatus.LEASED, JobStatus.RUNNING):
                await requeue_or_fail(
                    self._jobs, self._events, job,
                    error="orphaned by core restart",
                    expected=job.status,
                )
                reclaimed += 1
        return reclaimed