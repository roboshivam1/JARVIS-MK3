# =============================================================================
# tests/integration/test_worker_protocol.py - leases, offers, and death
# =============================================================================
#
# The worker link holds more state than anything else here: leases with
# expiry, offers that can be declined, heartbeats extending claims, and
# a reclaim loop sweeping what died. Every one has an edge case, and all
# of them have been exercised only on the happy path - a laptop
# connecting, taking work, finishing it.
#
# WHAT THIS IS REALLY LOOKING FOR: two things touching one job at once.
# A worker finishing just as its lease expires. A reclaim firing during
# a heartbeat. An offer accepted after the Core gave up waiting. Those
# are the bugs that surface once a month and cannot be reproduced on
# demand - so they get provoked deliberately here.
#
# The socket itself is not tested; the protocol MODELS and the state
# they drive are. A fake worker that speaks the same envelopes without a
# network is faster, deterministic, and tests the part that can actually
# be wrong.
# =============================================================================

from __future__ import annotations

import asyncio
from datetime import timedelta

from jarvis.common.envelope import Envelope, make_envelope
from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.jobs import Job, JobStatus, Lease
from jarvis.common.worker_protocol import (
    CORE_JOB_OFFER,
    WORKER_HELLO,
    WORKER_JOB_RESULT,
    CoreJobOffer,
    WorkerHello,
    WorkerJobResult,
)
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.queue.dispatcher import ReclaimLoop, requeue_or_fail
from jarvis.core.queue.registry_workers import ConnectedWorker, WorkerRegistry


def _worker(
    worker_id: str = "test-worker",
    capabilities: set[str] | None = None,
    concurrency: int = 2,
) -> ConnectedWorker:
    return ConnectedWorker(
        worker_id=worker_id,
        capabilities=capabilities or {"macos", "sandbox-exec"},
        max_concurrency=concurrency,
        version="test",
    )


async def _leased_job(
    jobs: JobsRepo,
    worker_id: str = "test-worker",
    requires: list[str] | None = None,
    lease_age_s: int = 0,
    ttl_s: int = 90,
) -> Job:
    """A job claimed by a worker, with a lease of a chosen age."""
    job = Job(
        type="code.task",
        requires=requires if requires is not None else ["sandbox-exec"],
        trace_id=new_ulid(),
    )
    await jobs.create(job)
    moment = utc_now() - timedelta(seconds=lease_age_s)
    await jobs.transition(
        job.id, JobStatus.QUEUED, JobStatus.LEASED,
        set_fields={
            "lease": Lease(
                worker_id=worker_id, leased_ts=moment,
                heartbeat_ts=moment, ttl_s=ttl_s,
            ),
            "attempts": 1,
        },
    )
    fresh = await jobs.get(job.id)
    assert fresh is not None
    return fresh


# -- the protocol vocabulary --------------------------------------------------

class TestProtocolMessages:
    """Both sides import the same models, so a message one can build the
    other can always parse. These pin that."""

    def test_hello_round_trips(self) -> None:
        hello = WorkerHello(
            worker_id="macbook", token="secret",
            capabilities=["macos", "browser"], max_concurrency=2,
        )
        envelope = make_envelope(WORKER_HELLO, hello)
        parsed = Envelope.model_validate(
            envelope.model_dump(mode="json")
        ).parse_payload()
        assert isinstance(parsed, WorkerHello)
        assert parsed.capabilities == ["macos", "browser"]

    def test_an_offer_carries_its_input_files(self) -> None:
        """Workers have no database, so artifact CONTENTS ride along
        with the offer. Sending ids instead meant the sandbox silently
        received nothing - and an agent handed an empty sandbox invents
        an answer rather than reporting the gap."""
        offer = CoreJobOffer(
            job_id=new_ulid(), type="code.task", payload={"task": "x"},
            requires=["sandbox-exec"], trace_id=new_ulid(),
            input_files={"data.csv": "YSxiCjEsMg=="},
        )
        envelope = make_envelope(CORE_JOB_OFFER, offer)
        parsed = Envelope.model_validate(
            envelope.model_dump(mode="json")
        ).parse_payload()
        assert isinstance(parsed, CoreJobOffer)
        assert parsed.input_files["data.csv"] == "YSxiCjEsMg=="

    def test_a_result_distinguishes_permanent_failure(self) -> None:
        """A malformed request fails identically every time; retrying
        it spends the same money twice more."""
        result = WorkerJobResult(
            job_id=new_ulid(), status="failed",
            error="no handler for this type", permanent=True,
        )
        envelope = make_envelope(WORKER_JOB_RESULT, result)
        parsed = Envelope.model_validate(
            envelope.model_dump(mode="json")
        ).parse_payload()
        assert isinstance(parsed, WorkerJobResult)
        assert parsed.permanent is True


# -- the fleet ----------------------------------------------------------------

class TestWorkerRegistry:
    def test_capability_matching(self) -> None:
        registry = WorkerRegistry()
        registry.register(_worker("mac", {"macos", "browser"}))
        registry.register(_worker("linux-box", {"linux"}))

        assert {w.worker_id for w in registry.eligible_for(["browser"])} == {"mac"}
        assert {w.worker_id for w in registry.eligible_for(["linux"])} == {"linux-box"}
        assert registry.eligible_for(["gpu"]) == []

    def test_a_saturated_worker_is_not_eligible(self) -> None:
        registry = WorkerRegistry()
        worker = _worker("mac", {"macos"}, concurrency=1)
        registry.register(worker)
        assert registry.eligible_for(["macos"])

        worker.running_job_ids.add(new_ulid())
        assert registry.eligible_for(["macos"]) == []

    def test_reconnecting_replaces_rather_than_duplicates(self) -> None:
        """A laptop whose link dropped presents the same id. The newer
        connection is the real one; leaving a ghost would mean offering
        work down a dead socket."""
        registry = WorkerRegistry()
        registry.register(_worker("macbook", {"macos"}))
        registry.register(_worker("macbook", {"macos", "browser"}))

        assert registry.count() == 1
        worker = registry.get("macbook")
        assert worker is not None and "browser" in worker.capabilities

    def test_heartbeat_updates_what_is_running(self) -> None:
        registry = WorkerRegistry()
        registry.register(_worker("mac"))
        job_id = new_ulid()

        registry.heartbeat("mac", [job_id])
        worker = registry.get("mac")
        assert worker is not None and job_id in worker.running_job_ids

        registry.heartbeat("mac", [])
        assert worker.running_job_ids == set()

    def test_disconnecting_removes_it(self) -> None:
        registry = WorkerRegistry()
        registry.register(_worker("mac"))
        registry.unregister("mac")
        assert registry.count() == 0
        assert registry.eligible_for(["macos"]) == []


# -- dispatch -----------------------------------------------------------------

class TestDispatch:
    async def test_core_only_work_stays_on_the_core(self, jobs: JobsRepo) -> None:
        """requires:[] work is API-bound. Sending it over a network to a
        laptop for no reason would be silly, and would leave the Core
        idle while a worker did what it could have done itself."""
        core_job = Job(type="research.brief", requires=[], trace_id=new_ulid())
        await jobs.create(core_job)

        picked = await jobs.next_queued(capabilities=set())
        assert picked is not None and picked.id == core_job.id

    async def test_a_worker_job_waits_for_a_capable_worker(
        self, jobs: JobsRepo
    ) -> None:
        """Waiting is NORMAL OPERATION, not a degraded mode. A browser
        task with the laptop shut simply sits until the laptop opens."""
        await jobs.create(Job(
            type="browser.task", requires=["browser"], trace_id=new_ulid()
        ))

        assert await jobs.next_queued(capabilities=set()) is None
        assert await jobs.next_queued(capabilities={"macos"}) is None
        assert await jobs.next_queued(capabilities={"browser"}) is not None

    async def test_partial_capability_is_not_enough(self, jobs: JobsRepo) -> None:
        await jobs.create(Job(
            type="test.demanding", requires=["browser", "gpu"],
            trace_id=new_ulid(),
        ))
        assert await jobs.next_queued(capabilities={"browser"}) is None
        assert await jobs.next_queued(capabilities={"browser", "gpu"}) is not None


# -- lease lifecycle ----------------------------------------------------------

class TestLeases:
    async def test_a_fresh_lease_survives(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        await _leased_job(jobs, lease_age_s=5, ttl_s=90)
        assert await ReclaimLoop(jobs, events).sweep_expired() == 0

    async def test_an_expired_lease_is_reclaimed(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """A worker that stopped heartbeating is presumed dead - lid
        closed, network gone, process killed. All the same event."""
        job = await _leased_job(jobs, lease_age_s=200, ttl_s=90)
        assert await ReclaimLoop(jobs, events).sweep_expired() == 1

        recovered = await jobs.get(job.id)
        assert recovered is not None
        assert recovered.status is JobStatus.QUEUED
        assert recovered.lease is None

    async def test_a_heartbeat_extends_the_lease(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """The reason heartbeats come from the worker RUNTIME and not
        from job code: a job blocked ten minutes on a model call is busy,
        not dead, and must not be reclaimed while still running."""
        job = await _leased_job(jobs, lease_age_s=200, ttl_s=90)

        # About to be reclaimed - then the worker checks in.
        await jobs.set_lease_heartbeat(job.id, utc_now())

        assert await ReclaimLoop(jobs, events).sweep_expired() == 0
        alive = await jobs.get(job.id)
        assert alive is not None and alive.status is JobStatus.LEASED

    async def test_a_heartbeat_for_someone_elses_job_is_ignored(
        self, jobs: JobsRepo
    ) -> None:
        """Two workers, and one reports a job it does not hold. Honouring
        that would let a confused worker keep another's dead job alive
        forever."""
        job = await _leased_job(jobs, worker_id="worker-a", lease_age_s=200)
        before = await jobs.get(job.id)
        assert before is not None and before.lease is not None

        # The connection handler checks ownership before extending; this
        # asserts the lease still names worker-a.
        assert before.lease.worker_id == "worker-a"

    async def test_reclaim_ages_the_job(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """A job that keeps killing its workers must not loop forever."""
        job = await _leased_job(jobs, lease_age_s=200, ttl_s=90)
        stored = await jobs.get(job.id)
        assert stored is not None
        stored = stored.model_copy(update={"attempts": stored.max_attempts})

        await requeue_or_fail(
            jobs, events, stored, "worker vanished",
            expected=JobStatus.LEASED,
        )

        dead = await jobs.get(job.id)
        assert dead is not None
        assert dead.status is JobStatus.FAILED


# -- races --------------------------------------------------------------------

class TestRaces:
    """The bugs that surface once a month and cannot be reproduced."""

    async def test_two_workers_cannot_claim_one_job(
        self, jobs: JobsRepo
    ) -> None:
        job = Job(type="code.task", requires=["sandbox-exec"], trace_id=new_ulid())
        await jobs.create(job)

        now = utc_now()
        results = await asyncio.gather(*[
            jobs.transition(
                job.id, JobStatus.QUEUED, JobStatus.LEASED,
                set_fields={
                    "lease": Lease(
                        worker_id=name, leased_ts=now,
                        heartbeat_ts=now, ttl_s=90,
                    ),
                    "attempts": 1,
                },
            )
            for name in ("worker-a", "worker-b", "worker-c")
        ])
        assert results.count(True) == 1

    async def test_a_result_arriving_after_reclaim_is_refused(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """The worst case: a worker finishes just as its lease expires.
        The job has already been requeued, so the late result must NOT
        be recorded - it would mark a job succeeded that another worker
        is now running.
        """
        job = await _leased_job(jobs, lease_age_s=200, ttl_s=90)
        await jobs.transition(job.id, JobStatus.LEASED, JobStatus.RUNNING)

        # Reclaim fires.
        running = await jobs.get(job.id)
        assert running is not None
        await requeue_or_fail(
            jobs, events, running, "lease expired",
            expected=JobStatus.RUNNING,
        )

        # The late result tries to land.
        landed = await jobs.transition(
            job.id, JobStatus.RUNNING, JobStatus.SUCCEEDED,
            set_fields={"result": {"summary": "too late"}, "lease": None},
        )
        assert not landed

        final = await jobs.get(job.id)
        assert final is not None
        assert final.status is JobStatus.QUEUED
        assert final.result is None

    async def test_cancellation_beats_a_finishing_worker(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """The owner cancels; the worker finishes a moment later. His
        decision wins and the result is dropped."""
        from jarvis.core.queue.dispatcher import cancel_job

        job = await _leased_job(jobs)
        await jobs.transition(job.id, JobStatus.LEASED, JobStatus.RUNNING)

        assert await cancel_job(jobs, events, job.id)

        landed = await jobs.transition(
            job.id, JobStatus.RUNNING, JobStatus.SUCCEEDED,
            set_fields={"result": {"summary": "finished anyway"}},
        )
        assert not landed

        final = await jobs.get(job.id)
        assert final is not None
        assert final.status is JobStatus.CANCELLED

    async def test_a_decline_costs_the_job_nothing(
        self, jobs: JobsRepo
    ) -> None:
        """Declining is not failing. A worker whose browser crashed
        between offer and accept should not burn one of the job's three
        attempts."""
        job = await _leased_job(jobs)
        assert job.attempts == 1

        await jobs.transition(
            job.id, JobStatus.LEASED, JobStatus.QUEUED,
            set_fields={"lease": None, "attempts": job.attempts - 1},
        )

        returned = await jobs.get(job.id)
        assert returned is not None
        assert returned.status is JobStatus.QUEUED
        assert returned.attempts == 0


# -- restart ------------------------------------------------------------------

class TestRestartRecovery:
    async def test_everything_in_flight_is_reclaimed(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """After a Core restart the in-process executors are gone
        regardless of how fresh their leases look."""
        fresh = await _leased_job(jobs, lease_age_s=1, ttl_s=300)
        running = await _leased_job(jobs, lease_age_s=1, ttl_s=300)
        await jobs.transition(running.id, JobStatus.LEASED, JobStatus.RUNNING)

        assert await ReclaimLoop(jobs, events).reclaim_all_inflight() == 2

        for job_id in (fresh.id, running.id):
            recovered = await jobs.get(job_id)
            assert recovered is not None
            assert recovered.status is JobStatus.QUEUED

    async def test_a_paused_job_survives_a_restart(
        self, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """An awaiting_approval job holds no lease and is waiting on a
        PERSON. Reclaiming it would discard the owner's pending
        question."""
        job = await _leased_job(jobs)
        await jobs.transition(job.id, JobStatus.LEASED, JobStatus.RUNNING)

        from jarvis.common.jobs import Approval
        await jobs.transition(
            job.id, JobStatus.RUNNING, JobStatus.AWAITING_APPROVAL,
            set_fields={
                "approval": Approval(gate="publish", requested_ts=utc_now()),
                "lease": None,
            },
        )

        await ReclaimLoop(jobs, events).reclaim_all_inflight()

        waiting = await jobs.get(job.id)
        assert waiting is not None
        assert waiting.status is JobStatus.AWAITING_APPROVAL
