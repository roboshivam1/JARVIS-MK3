# =============================================================================
# src/jarvis/core/queue/coreworker.py - the built-in executor for
# requires:[] jobs
# =============================================================================
#
# A polite loop inside the Core process: find a queued job needing no
# capabilities, lease it (attempts += 1 AT the lease - vanishing must
# consume an attempt), run its registered handler under a timeout,
# record the outcome through the shared failure path.
#
# Lease TTL = handler timeout + margin: while we are alive the timeout
# fires first, so the janitor never steals a merely-slow job; if the
# process dies, boot recovery reclaims everything anyway. That is why
# this executor needs no heartbeat of its own.
#
# Owner cancellation appears to this loop as a LOST RACE: the terminal
# transition returns False because the status already moved to cancelled.
# Losing that race means dropping the result on the floor - correct,
# because the owner said stop.
# =============================================================================

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import ValidationError

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import utc_now
from jarvis.common.jobs import Job, JobStatus, Lease
from jarvis.common.log import get_logger
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.queue.dispatcher import fail_permanently, requeue_or_fail
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, PermanentJobError

log = get_logger("core.queue.coreworker")

SOURCE = "core.coreworker"
WORKER_ID = "core"
_POLL_INTERVAL_S = 1.0
_LEASE_MARGIN_S = 60


class CoreWorker:
    """Executes requires:[] jobs inside the Core process."""

    def __init__(
        self,
        jobs: JobsRepo,
        events: EventsRepo,
        registry: JobTypeRegistry,
        artifacts: ArtifactsRepo,
        max_concurrency: int = 2,
    ) -> None:
        self._jobs = jobs
        self._events = events
        self._registry = registry
        self._artifacts = artifacts
        self._max = max_concurrency
        self._running: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        """The resident loop: claim work while there is capacity."""
        while True:
            try:
                if len(self._running) < self._max:
                    job = await self._jobs.next_queued(capabilities=set())
                    if job is not None and await self.lease(job):
                        task = asyncio.create_task(
                            self.execute(job), name=f"corejob-{job.id[-6:]}"
                        )
                        self._running.add(task)
                        task.add_done_callback(self._running.discard)
                        continue   # look for more work immediately
            except Exception:
                log.error("core worker loop error", exc_info=True)
            await asyncio.sleep(_POLL_INTERVAL_S)

    async def lease(self, job: Job) -> bool:
        """Claim the job. False = someone else got it first (fine)."""
        spec = self._registry.get(job.type)
        ttl = (spec.timeout_s if spec else 300) + _LEASE_MARGIN_S
        now = utc_now()
        won = await self._jobs.transition(
            job.id, JobStatus.QUEUED, JobStatus.LEASED,
            set_fields={
                "lease": Lease(
                    worker_id=WORKER_ID, leased_ts=now,
                    heartbeat_ts=now, ttl_s=ttl,
                ),
                "attempts": job.attempts + 1,
            },
        )
        if won:
            await self._events.append(Event(
                kind=EventKind.JOB_LEASED, source=SOURCE,
                job_id=job.id, session_id=job.session_id,
                trace_id=job.trace_id,
                payload={"worker": WORKER_ID, "attempt": job.attempts + 1},
            ))
        return won

    async def execute(self, leased: Job) -> None:
        """Run one job to a terminal (or requeued) state. Never raises."""
        job = await self._jobs.get(leased.id)
        if job is None:
            return

        spec = self._registry.get(job.type)
        if spec is None or spec.handler is None:
            # Queue drift: a job whose type this build cannot run. Waiting
            # will not conjure a handler, so this is permanent.
            await fail_permanently(
                self._jobs, self._events, job,
                error=f"no runnable handler for job type {job.type!r}",
                expected=JobStatus.LEASED,
            )
            return

        if not await self._jobs.transition(
            job.id, JobStatus.LEASED, JobStatus.RUNNING
        ):
            return   # cancelled or reclaimed between lease and start
        await self._events.append(Event(
            kind=EventKind.JOB_STARTED, source=SOURCE,
            job_id=job.id, session_id=job.session_id, trace_id=job.trace_id,
            payload={},
        ))

        async def save_checkpoint(state: dict[str, Any]) -> None:
            await self._jobs.set_checkpoint(job.id, state)
            await self._events.append(Event(
                kind=EventKind.JOB_CHECKPOINT, source=SOURCE,
                job_id=job.id, trace_id=job.trace_id, payload={},
            ))

        async def progress(note: str) -> None:
            await self._events.append(Event(
                kind=EventKind.JOB_PROGRESS, source=SOURCE,
                job_id=job.id, session_id=job.session_id,
                trace_id=job.trace_id, payload={"note": note},
            ))

        async def write_artifact(name: str, mime: str, content: bytes) -> str:
            artifact = await self._artifacts.write(
                name=name, mime=mime, content=content, created_by=job.id
            )
            await self._jobs.add_artifact(job.id, artifact.id)
            return artifact.id

        ctx = JobContext(
            job_id=job.id, trace_id=job.trace_id,
            checkpoint=job.checkpoint,
            save_checkpoint=save_checkpoint, progress=progress,
            write_artifact=write_artifact,
        )

        try:
            payload = spec.input_model.model_validate(job.payload)
            async with asyncio.timeout(spec.timeout_s):
                output = await spec.handler(payload, ctx)
            result: dict[str, Any] = spec.output_model.model_validate(
                output, from_attributes=True
            ).model_dump(mode="json")
        except (TimeoutError, asyncio.TimeoutError):
            fresh = await self._jobs.get(job.id)
            if fresh:
                await requeue_or_fail(
                    self._jobs, self._events, fresh,
                    error=f"timed out after {spec.timeout_s}s",
                    expected=JobStatus.RUNNING,
                )
            return
        except PermanentJobError as exc:
            # The handler knows retrying is pointless.
            fresh = await self._jobs.get(job.id)
            if fresh:
                await fail_permanently(
                    self._jobs, self._events, fresh,
                    error=str(exc), expected=JobStatus.RUNNING,
                )
            return
        except ValidationError as exc:
            # A payload that does not match its schema will not match it
            # on the next attempt either.
            fresh = await self._jobs.get(job.id)
            if fresh:
                await fail_permanently(
                    self._jobs, self._events, fresh,
                    error=f"payload/result validation failed: {exc}",
                    expected=JobStatus.RUNNING,
                )
            return
        
        except Exception as exc:
            log.error("job handler crashed", exc_info=True,
                      extra={"job_id": job.id, "type": job.type})
            fresh = await self._jobs.get(job.id)
            if fresh:
                await requeue_or_fail(
                    self._jobs, self._events, fresh,
                    error=f"{type(exc).__name__}: {exc}",
                    expected=JobStatus.RUNNING,
                )
            return

        moved = await self._jobs.transition(
            job.id, JobStatus.RUNNING, JobStatus.SUCCEEDED,
            set_fields={"result": result, "lease": None},
        )
        if moved:
            await self._events.append(Event(
                kind=EventKind.JOB_SUCCEEDED, source=SOURCE,
                job_id=job.id, session_id=job.session_id,
                trace_id=job.trace_id, payload={},
            ))
            log.info("job succeeded", extra={"job_id": job.id, "type": job.type})
        # Lost race here = owner cancelled mid-run; result dropped by design.