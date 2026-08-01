# =============================================================================
# tests/integration/test_queue.py - the durable work machinery
# =============================================================================
#
# The queue is what every later phase stands on, so its promises are
# pinned here: legal transitions only, races detected rather than
# overwritten, failures aged consistently, orphans recovered, and
# hopeless failures not retried three times at your expense.
# =============================================================================

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from pydantic import BaseModel, ConfigDict

from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.jobs import Job, JobStatus, Lease
from jarvis.core.db.database import Database
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import IllegalTransition, JobsRepo
from jarvis.core.queue.coreworker import CoreWorker
from jarvis.core.queue.dispatcher import ReclaimLoop, cancel_job, requeue_or_fail
from jarvis.core.queue.registry import (
    JobContext,
    JobTypeRegistry,
    JobTypeSpec,
    PermanentJobError,
)


def _job(**overrides: object) -> Job:
    payload: dict[str, object] = {
        "type": "test.work", "trace_id": new_ulid(), **overrides
    }
    return Job(**payload)  # type: ignore[arg-type]


def _lease(worker: str = "core", age_s: int = 0, ttl_s: int = 300) -> Lease:
    moment = utc_now() - timedelta(seconds=age_s)
    return Lease(worker_id=worker, leased_ts=moment,
                 heartbeat_ts=moment, ttl_s=ttl_s)


# -- creation and transitions -------------------------------------------------

class TestTransitions:
    async def test_create_writes_job_and_event_together(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        job = _job()
        await jobs.create(job)

        stored = await jobs.get(job.id)
        assert stored is not None and stored.status is JobStatus.QUEUED

        # The enqueued event shares the job's trace - one causal story.
        trail = await events.by_trace(job.trace_id)
        assert [e.kind for e in trail] == ["job.enqueued"]
        assert trail[0].job_id == job.id

    async def test_legal_transition_applies(self, jobs: JobsRepo) -> None:
        job = _job()
        await jobs.create(job)
        moved = await jobs.transition(
            job.id, JobStatus.QUEUED, JobStatus.LEASED,
            set_fields={"lease": _lease()},
        )
        assert moved
        stored = await jobs.get(job.id)
        assert stored is not None
        assert stored.status is JobStatus.LEASED
        assert stored.lease is not None and stored.lease.worker_id == "core"

    async def test_illegal_transition_raises_before_touching_db(
        self, jobs: JobsRepo
    ) -> None:
        job = _job()
        await jobs.create(job)
        with pytest.raises(IllegalTransition):
            await jobs.transition(job.id, JobStatus.QUEUED, JobStatus.SUCCEEDED)
        stored = await jobs.get(job.id)
        assert stored is not None and stored.status is JobStatus.QUEUED

    async def test_lost_race_returns_false_and_changes_nothing(
        self, jobs: JobsRepo
    ) -> None:
        # Two claimants, one job: exactly one may win.
        job = _job()
        await jobs.create(job)
        first, second = await asyncio.gather(
            jobs.transition(job.id, JobStatus.QUEUED, JobStatus.LEASED,
                            set_fields={"lease": _lease("worker-a")}),
            jobs.transition(job.id, JobStatus.QUEUED, JobStatus.LEASED,
                            set_fields={"lease": _lease("worker-b")}),
        )
        assert [first, second].count(True) == 1

    async def test_terminal_transition_stamps_finished(self, jobs: JobsRepo) -> None:
        job = _job()
        await jobs.create(job)
        await jobs.transition(job.id, JobStatus.QUEUED, JobStatus.LEASED,
                              set_fields={"lease": _lease()})
        await jobs.transition(job.id, JobStatus.LEASED, JobStatus.RUNNING)
        await jobs.transition(job.id, JobStatus.RUNNING, JobStatus.SUCCEEDED,
                              set_fields={"result": {"ok": True}, "lease": None})
        stored = await jobs.get(job.id)
        assert stored is not None
        assert stored.finished_ts is not None
        assert stored.result == {"ok": True}


# -- dispatch selection -------------------------------------------------------

class TestDispatchSelection:
    async def test_capability_filtering(self, jobs: JobsRepo) -> None:
        await jobs.create(_job(type="test.browser", requires=["browser"]))
        plain = _job(type="test.plain")
        await jobs.create(plain)

        # A bare Core worker may only take the unrestricted job.
        picked = await jobs.next_queued(capabilities=set())
        assert picked is not None and picked.id == plain.id

        # A browser-capable worker can see both; priority/age decides.
        picked = await jobs.next_queued(capabilities={"browser"})
        assert picked is not None

    async def test_priority_beats_age(self, jobs: JobsRepo) -> None:
        old_low = _job(priority=7)
        await jobs.create(old_low)
        new_high = _job(priority=1)
        await jobs.create(new_high)
        picked = await jobs.next_queued(capabilities=set())
        assert picked is not None and picked.id == new_high.id

    async def test_not_before_defers(self, jobs: JobsRepo) -> None:
        future = _job(not_before=utc_now() + timedelta(minutes=5))
        await jobs.create(future)
        assert await jobs.next_queued(capabilities=set()) is None


# -- failure ageing -----------------------------------------------------------

class TestFailurePath:
    async def test_requeues_with_backoff_while_attempts_remain(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        job = _job(max_attempts=3)
        await jobs.create(job)
        await jobs.transition(job.id, JobStatus.QUEUED, JobStatus.LEASED,
                              set_fields={"lease": _lease(), "attempts": 1})
        leased = await jobs.get(job.id)
        assert leased is not None

        await requeue_or_fail(jobs, events, leased, "provider hiccup",
                              expected=JobStatus.LEASED)

        stored = await jobs.get(job.id)
        assert stored is not None
        assert stored.status is JobStatus.QUEUED
        assert stored.lease is None
        assert stored.not_before is not None and stored.not_before > utc_now()

    async def test_fails_terminally_when_attempts_exhausted(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        job = _job(max_attempts=1)
        await jobs.create(job)
        await jobs.transition(job.id, JobStatus.QUEUED, JobStatus.LEASED,
                              set_fields={"lease": _lease(), "attempts": 1})
        leased = await jobs.get(job.id)
        assert leased is not None

        await requeue_or_fail(jobs, events, leased, "gave up",
                              expected=JobStatus.LEASED)

        stored = await jobs.get(job.id)
        assert stored is not None
        assert stored.status is JobStatus.FAILED
        assert stored.error == "gave up"
        assert stored.finished_ts is not None


# -- reclaim and recovery -----------------------------------------------------

class TestReclaim:
    async def test_expired_lease_is_reclaimed(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        job = _job()
        await jobs.create(job)
        # Heartbeat older than its own TTL: the executor is presumed dead.
        await jobs.transition(
            job.id, JobStatus.QUEUED, JobStatus.LEASED,
            set_fields={"lease": _lease(age_s=600, ttl_s=60), "attempts": 1},
        )

        reclaimed = await ReclaimLoop(jobs, events).sweep_expired()
        assert reclaimed == 1
        stored = await jobs.get(job.id)
        assert stored is not None and stored.status is JobStatus.QUEUED

    async def test_fresh_lease_is_left_alone(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        job = _job()
        await jobs.create(job)
        await jobs.transition(job.id, JobStatus.QUEUED, JobStatus.LEASED,
                              set_fields={"lease": _lease(age_s=5, ttl_s=300)})

        assert await ReclaimLoop(jobs, events).sweep_expired() == 0
        stored = await jobs.get(job.id)
        assert stored is not None and stored.status is JobStatus.LEASED

    async def test_boot_recovery_reclaims_all_inflight(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        # After a restart, in-process executors are gone regardless of
        # how fresh their leases look.
        job = _job()
        await jobs.create(job)
        await jobs.transition(job.id, JobStatus.QUEUED, JobStatus.LEASED,
                              set_fields={"lease": _lease(age_s=1), "attempts": 1})
        await jobs.transition(job.id, JobStatus.LEASED, JobStatus.RUNNING)

        assert await ReclaimLoop(jobs, events).reclaim_all_inflight() == 1
        stored = await jobs.get(job.id)
        assert stored is not None and stored.status is JobStatus.QUEUED


# -- cancellation -------------------------------------------------------------

class TestCancellation:
    async def test_cancels_queued_job(self, jobs: JobsRepo, events: EventsRepo) -> None:
        job = _job()
        await jobs.create(job)
        assert await cancel_job(jobs, events, job.id)
        stored = await jobs.get(job.id)
        assert stored is not None and stored.status is JobStatus.CANCELLED

    async def test_cannot_cancel_finished_job(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        job = _job()
        await jobs.create(job)
        await jobs.transition(job.id, JobStatus.QUEUED, JobStatus.CANCELLED,
                              set_fields={"error": "already done"})
        assert not await cancel_job(jobs, events, job.id)

    async def test_unknown_job_is_not_cancellable(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        assert not await cancel_job(jobs, events, new_ulid())


# -- execution ----------------------------------------------------------------

class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doubled: int


def _worker(db: Database, jobs: JobsRepo, events: EventsRepo,
            registry: JobTypeRegistry, tmp_path_root: str) -> CoreWorker:
    from pathlib import Path
    return CoreWorker(jobs, events, registry, ArtifactsRepo(db, Path(tmp_path_root)))


class TestExecution:
    async def test_successful_job_records_result(
        self, db: Database, jobs: JobsRepo, events: EventsRepo, tmp_path: object
    ) -> None:
        async def handler(payload: BaseModel, ctx: JobContext) -> BaseModel:
            assert isinstance(payload, _In)
            return _Out(doubled=payload.value * 2)

        registry = JobTypeRegistry()
        registry.register(JobTypeSpec(
            type="test.double", input_model=_In, output_model=_Out,
            execution="idempotent", handler=handler,
        ))
        worker = _worker(db, jobs, events, registry, str(tmp_path))

        job = _job(type="test.double", payload={"value": 21})
        await jobs.create(job)
        assert await worker.lease(job)
        await worker.execute(job)

        stored = await jobs.get(job.id)
        assert stored is not None
        assert stored.status is JobStatus.SUCCEEDED
        assert stored.result == {"doubled": 42}

    async def test_crashing_handler_is_retried(
        self, db: Database, jobs: JobsRepo, events: EventsRepo, tmp_path: object
    ) -> None:
        async def handler(payload: BaseModel, ctx: JobContext) -> BaseModel:
            raise RuntimeError("transient trouble")

        registry = JobTypeRegistry()
        registry.register(JobTypeSpec(
            type="test.crash", input_model=_In, output_model=_Out,
            execution="idempotent", handler=handler,
        ))
        worker = _worker(db, jobs, events, registry, str(tmp_path))

        job = _job(type="test.crash", payload={"value": 1}, max_attempts=3)
        await jobs.create(job)
        assert await worker.lease(job)
        await worker.execute(job)

        stored = await jobs.get(job.id)
        assert stored is not None
        assert stored.status is JobStatus.QUEUED   # requeued, not failed
        assert stored.attempts == 1

    async def test_permanent_error_is_not_retried(
        self, db: Database, jobs: JobsRepo, events: EventsRepo, tmp_path: object
    ) -> None:
        # The lesson of a KeyError that cost three research runs: a bug
        # fails identically every time, so it must fail once.
        async def handler(payload: BaseModel, ctx: JobContext) -> BaseModel:
            raise PermanentJobError("this can never work")

        registry = JobTypeRegistry()
        registry.register(JobTypeSpec(
            type="test.doomed", input_model=_In, output_model=_Out,
            execution="idempotent", handler=handler,
        ))
        worker = _worker(db, jobs, events, registry, str(tmp_path))

        job = _job(type="test.doomed", payload={"value": 1}, max_attempts=3)
        await jobs.create(job)
        assert await worker.lease(job)
        await worker.execute(job)

        stored = await jobs.get(job.id)
        assert stored is not None
        assert stored.status is JobStatus.FAILED
        assert stored.attempts == 1          # one attempt, not three
        assert "never work" in (stored.error or "")

    async def test_invalid_payload_fails_permanently(
        self, db: Database, jobs: JobsRepo, events: EventsRepo, tmp_path: object
    ) -> None:
        async def handler(payload: BaseModel, ctx: JobContext) -> BaseModel:
            return _Out(doubled=0)

        registry = JobTypeRegistry()
        registry.register(JobTypeSpec(
            type="test.strict", input_model=_In, output_model=_Out,
            execution="idempotent", handler=handler,
        ))
        worker = _worker(db, jobs, events, registry, str(tmp_path))

        job = _job(type="test.strict", payload={"wrong": "shape"})
        await jobs.create(job)
        assert await worker.lease(job)
        await worker.execute(job)

        stored = await jobs.get(job.id)
        assert stored is not None
        assert stored.status is JobStatus.FAILED

    async def test_unregistered_type_fails_permanently(
        self, db: Database, jobs: JobsRepo, events: EventsRepo, tmp_path: object
    ) -> None:
        worker = _worker(db, jobs, events, JobTypeRegistry(), str(tmp_path))
        job = _job(type="test.unknown")
        await jobs.create(job)
        assert await worker.lease(job)
        await worker.execute(job)

        stored = await jobs.get(job.id)
        assert stored is not None and stored.status is JobStatus.FAILED

    async def test_artifacts_are_attached_to_their_job(
        self, db: Database, jobs: JobsRepo, events: EventsRepo, tmp_path: object
    ) -> None:
        async def handler(payload: BaseModel, ctx: JobContext) -> BaseModel:
            await ctx.write_artifact("out.md", "text/markdown", b"# hello")
            return _Out(doubled=0)

        registry = JobTypeRegistry()
        registry.register(JobTypeSpec(
            type="test.artifact", input_model=_In, output_model=_Out,
            execution="idempotent", handler=handler,
        ))
        worker = _worker(db, jobs, events, registry, str(tmp_path))

        job = _job(type="test.artifact", payload={"value": 1})
        await jobs.create(job)
        assert await worker.lease(job)
        await worker.execute(job)

        stored = await jobs.get(job.id)
        assert stored is not None
        assert len(stored.artifacts) == 1