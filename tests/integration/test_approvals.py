# =============================================================================
# tests/integration/test_approvals.py - the safety layer, verified
# =============================================================================
#
# The guard and the approval gates are what stand between a confused
# model and something the owner cannot undo. Everything else in this
# system fails by producing a wrong answer; this fails by submitting a
# form, pushing a repository, or deleting a file.
#
# It has also been edited more than anything else - the policy has been
# rewritten by anchor-matching scripts five or six times, occasionally
# half-applying. A hole could exist right now and nothing would say so.
#
# TWO KINDS OF TEST, and the second matters more:
#   positive - a gate fires when it should
#   negative - NOTHING gets through that should not
# A policy edit that accidentally widens an allowlist passes every
# positive test and fails the negative ones.
# =============================================================================

from __future__ import annotations

from datetime import timedelta

import pytest

from jarvis.agentloop.guard import Verdict
from jarvis.agentloop.policies import (
    ARCHIVIST,
    ENGINEER,
    OPERATOR,
    ORCHESTRATOR,
    RESEARCHER,
    WRITER,
    create_guard,
)
from jarvis.common.approvals import ApprovalStatus
from jarvis.common.capabilities import Gate
from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.jobs import Job, JobStatus, Lease
from jarvis.core.approvals.service import ApprovalService
from jarvis.core.db.database import Database
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo


# -- the guard ----------------------------------------------------------------

class TestNothingGetsThroughUnasked:
    """The negative tests. Each one is an action that must never happen
    without the owner's say-so."""

    def test_form_submission_is_gated(self) -> None:
        guard = create_guard()
        decision = guard.check(
            OPERATOR, "browser_submit_form", {"url": "https://x.com/apply"}
        )
        assert decision.verdict is Verdict.GATE
        assert decision.gate is Gate.OUTBOUND

    def test_login_is_gated(self) -> None:
        guard = create_guard()
        decision = guard.check(
            OPERATOR, "browser_navigate", {"url": "https://bank.com/login"}
        )
        assert decision.verdict is Verdict.GATE
        assert decision.gate is Gate.CREDENTIAL

    def test_push_is_gated(self) -> None:
        guard = create_guard()
        decision = guard.check(ENGINEER, "git_push", {"project": "anything"})
        assert decision.verdict is Verdict.GATE
        assert decision.gate is Gate.PUBLISH

    def test_repo_creation_is_gated(self) -> None:
        guard = create_guard()
        decision = guard.check(
            ENGINEER, "git_create_repo", {"project": "new-thing"}
        )
        assert decision.verdict is Verdict.GATE
        assert decision.gate is Gate.PUBLISH

    def test_writing_outside_the_sandbox_is_gated(self) -> None:
        guard = create_guard()
        decision = guard.check(ENGINEER, "file_write", {"path": "/etc/hosts"})
        assert decision.verdict is Verdict.GATE
        assert decision.gate is Gate.DESTRUCTIVE_FILE

    def test_deletion_is_gated(self) -> None:
        guard = create_guard()
        assert guard.check(
            ENGINEER, "file_delete", {"path": "anything"}
        ).verdict is Verdict.GATE

    @pytest.mark.parametrize("actor", [
        ORCHESTRATOR, RESEARCHER, WRITER, ARCHIVIST,
    ])
    def test_only_the_operator_can_reach_a_browser(self, actor: str) -> None:
        """PROTEUS drives browsers. Nobody else should be able to,
        whatever they ask for."""
        guard = create_guard()
        assert not guard.check(actor, "browser_navigate", {"url": "x"}).allowed
        assert not guard.check(actor, "mcp__playwright__browser_click", {}).allowed

    @pytest.mark.parametrize("actor", [
        ORCHESTRATOR, RESEARCHER, OPERATOR, WRITER, ARCHIVIST,
    ])
    def test_only_the_engineer_can_run_code(self, actor: str) -> None:
        guard = create_guard()
        assert not guard.check(actor, "sandbox_run_python", {"code": "x"}).allowed

    @pytest.mark.parametrize("actor", [
        ORCHESTRATOR, RESEARCHER, OPERATOR, WRITER,
    ])
    def test_only_the_archivist_holds_privileged_memory(self, actor: str) -> None:
        """MNEMOSYNE may merge, supersede, and expire facts. The
        orchestrator may store and search; nobody else touches memory
        at all."""
        guard = create_guard()
        assert not guard.check(actor, "memory_expire", {}).allowed
        assert not guard.check(actor, "memory_supersede", {}).allowed

    def test_the_front_mind_has_no_hands(self) -> None:
        """JARVIS talks and delegates. Hands live in subagents, inside
        jobs, behind gates - so the conversational loop, which is what
        a prompt injection would reach first, can do nothing dangerous
        even if entirely persuaded."""
        guard = create_guard()
        for forbidden in (
            "browser_navigate", "sandbox_run_python", "file_write",
            "file_delete", "git_push", "git_create_repo",
        ):
            assert not guard.check(ORCHESTRATOR, forbidden, {}).allowed

    def test_an_unknown_actor_gets_nothing(self) -> None:
        guard = create_guard()
        for tool in ("get_time_context", "memory_search", "file_read"):
            assert guard.check("subagent.impostor", tool, {}).verdict is Verdict.DENY


class TestOrdinaryWorkIsNotObstructed:
    """The positive tests. A guard that refuses everything is safe and
    useless; these are the things that must keep working."""

    def test_the_orchestrator_can_talk_and_delegate(self) -> None:
        guard = create_guard()
        for tool in (
            "get_time_context", "memory_search", "memory_store",
            "run_subagent", "list_jobs", "create_watcher",
        ):
            assert guard.check(ORCHESTRATOR, tool, {}).allowed, tool

    def test_the_engineer_can_work_in_its_sandbox(self) -> None:
        guard = create_guard()
        for tool, args in (
            ("sandbox_run_python", {"code": "print(1)"}),
            ("file_write", {"path": "projects/thing/main.py"}),
            ("file_read", {"path": "projects/thing/main.py"}),
            ("file_edit", {"path": "projects/thing/main.py"}),
            ("git_commit", {"project": "thing"}),
            ("git_status", {"project": "thing"}),
        ):
            assert guard.check(ENGINEER, tool, args).allowed, tool

    def test_ordinary_browsing_is_not_gated(self) -> None:
        guard = create_guard()
        assert guard.check(
            OPERATOR, "browser_navigate", {"url": "https://news.ycombinator.com"}
        ).allowed

    def test_the_researcher_can_read_the_web(self) -> None:
        guard = create_guard()
        assert guard.check(RESEARCHER, "web_search", {}).allowed
        assert guard.check(RESEARCHER, "web_fetch", {}).allowed

    def test_decisions_explain_themselves(self) -> None:
        """The agent reads a denial and routes around it; the owner
        reads a gate and decides. An empty reason is a usability bug as
        well as a debugging one."""
        guard = create_guard()
        denial = guard.check(ORCHESTRATOR, "sandbox_run_python", {})
        assert denial.reason and "sandbox_run_python" in denial.reason

        gate = guard.check(ENGINEER, "git_push", {"project": "x"})
        assert gate.reason and len(gate.reason) > 10


# -- the approval flow --------------------------------------------------------

async def _running_job(jobs: JobsRepo) -> Job:
    """A job in the state a gate would interrupt."""
    job = Job(type="code.task", trace_id=new_ulid())
    await jobs.create(job)
    now = utc_now()
    await jobs.transition(
        job.id, JobStatus.QUEUED, JobStatus.LEASED,
        set_fields={
            "lease": Lease(
                worker_id="test", leased_ts=now, heartbeat_ts=now, ttl_s=300
            ),
            "attempts": 1,
        },
    )
    await jobs.transition(job.id, JobStatus.LEASED, JobStatus.RUNNING)
    fresh = await jobs.get(job.id)
    assert fresh is not None
    return fresh


async def _service(db: Database, jobs: JobsRepo, events: EventsRepo) -> ApprovalService:
    return ApprovalService(ApprovalsRepo(db), jobs, events)


class TestPausingForPermission:
    async def test_raising_a_gate_pauses_the_job(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        service = await _service(db, jobs, events)
        job = await _running_job(jobs)

        request = await service.request(
            job_id=job.id, gate=Gate.PUBLISH, actor=ENGINEER,
            tool="git_push", summary="Push to GitHub",
            detail="repo: test/thing", risk_note="cannot be undone",
        )
        assert request is not None

        paused = await jobs.get(job.id)
        assert paused is not None
        assert paused.status is JobStatus.AWAITING_APPROVAL
        # The lease is released: this may take hours, and holding a
        # worker slot while the owner sleeps would be absurd.
        assert paused.lease is None

    async def test_the_exact_action_is_recorded(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """An approval prompt that summarises away the specifics trains
        the owner to tap yes without reading."""
        service = await _service(db, jobs, events)
        job = await _running_job(jobs)

        detail = "To: someone@example.com\nSubject: Test\n\nThe actual body."
        request = await service.request(
            job_id=job.id, gate=Gate.OUTBOUND, actor=OPERATOR,
            tool="send_email", summary="Send an email",
            detail=detail, risk_note="leaves the machine as you",
        )
        assert request is not None

        stored = await ApprovalsRepo(db).get(request.id)
        assert stored is not None
        assert stored.detail == detail
        assert stored.risk_note

    async def test_a_finished_job_cannot_be_gated(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        service = await _service(db, jobs, events)
        job = Job(type="code.task", trace_id=new_ulid())
        await jobs.create(job)

        # Still queued, never ran - there is nothing to pause.
        assert await service.request(
            job_id=job.id, gate=Gate.PUBLISH, actor=ENGINEER,
            tool="git_push", summary="x", detail="y",
        ) is None


class TestDeciding:
    async def test_approval_returns_the_job_to_the_queue(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        service = await _service(db, jobs, events)
        job = await _running_job(jobs)
        request = await service.request(
            job_id=job.id, gate=Gate.PUBLISH, actor=ENGINEER,
            tool="git_push", summary="x", detail="y",
        )
        assert request is not None

        assert await service.decide(request.id, approve=True)

        resumed = await jobs.get(job.id)
        assert resumed is not None
        assert resumed.status is JobStatus.QUEUED
        assert resumed.approval is None

    async def test_rejection_cancels_the_job(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        service = await _service(db, jobs, events)
        job = await _running_job(jobs)
        request = await service.request(
            job_id=job.id, gate=Gate.OUTBOUND, actor=OPERATOR,
            tool="send_email", summary="x", detail="y",
        )
        assert request is not None

        assert await service.decide(request.id, approve=False, note="no")

        cancelled = await jobs.get(job.id)
        assert cancelled is not None
        assert cancelled.status is JobStatus.CANCELLED

    async def test_a_decision_lands_exactly_once(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """The owner taps twice, or taps just as the expiry sweep runs.
        Exactly one of those wins."""
        service = await _service(db, jobs, events)
        job = await _running_job(jobs)
        request = await service.request(
            job_id=job.id, gate=Gate.PUBLISH, actor=ENGINEER,
            tool="git_push", summary="x", detail="y",
        )
        assert request is not None

        assert await service.decide(request.id, approve=True)
        assert not await service.decide(request.id, approve=False)

        # And the first decision stands.
        settled = await jobs.get(job.id)
        assert settled is not None
        assert settled.status is JobStatus.QUEUED

    async def test_the_decision_is_on_the_record(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """History is what makes graduated autonomy earnable later."""
        service = await _service(db, jobs, events)
        approvals = ApprovalsRepo(db)
        job = await _running_job(jobs)
        request = await service.request(
            job_id=job.id, gate=Gate.PUBLISH, actor=ENGINEER,
            tool="git_push", summary="x", detail="y",
        )
        assert request is not None
        await service.decide(request.id, approve=True, note="looks right")

        history = await approvals.history_for_gate(ENGINEER, Gate.PUBLISH)
        assert len(history) == 1
        assert history[0].status is ApprovalStatus.APPROVED
        assert history[0].decision_note == "looks right"


class TestExpiry:
    async def test_an_unanswered_gate_fails_the_job(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        """A request nobody answers must not pause work forever."""
        service = ApprovalService(
            ApprovalsRepo(db), jobs, events, ttl_hours=0,
        )
        job = await _running_job(jobs)
        request = await service.request(
            job_id=job.id, gate=Gate.PUBLISH, actor=ENGINEER,
            tool="git_push", summary="x", detail="y",
        )
        assert request is not None

        assert await service.sweep_expired() == 1

        expired = await jobs.get(job.id)
        assert expired is not None
        assert expired.status is JobStatus.CANCELLED
        assert "timed out" in (expired.error or "")

    async def test_a_live_gate_survives_the_sweep(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        service = await _service(db, jobs, events)
        job = await _running_job(jobs)
        await service.request(
            job_id=job.id, gate=Gate.PUBLISH, actor=ENGINEER,
            tool="git_push", summary="x", detail="y",
        )

        assert await service.sweep_expired() == 0
        waiting = await jobs.get(job.id)
        assert waiting is not None
        assert waiting.status is JobStatus.AWAITING_APPROVAL
