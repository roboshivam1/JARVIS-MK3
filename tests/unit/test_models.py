# =============================================================================
# tests/unit/test_models.py - contract tests for common/ (ids, envelope,
# events, jobs)
# =============================================================================
#
# Each test pins one promise a module makes. The most load-bearing test in
# the file is test_state_machine_matches_design_doc: it restates the job
# state diagram by hand and compares it to the code, so doc and code
# cannot drift apart silently.
# =============================================================================

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from jarvis.common.envelope import (
    Envelope,
    UnknownKind,
    make_envelope,
    register_kind,
)
from jarvis.common.events import ALL_EVENT_KINDS, Event, EventKind
from jarvis.common.ids import is_ulid, new_ulid, ulid_timestamp, utc_now
from jarvis.common.jobs import (
    ALLOWED_TRANSITIONS,
    Approval,
    Job,
    JobStatus,
    Lease,
    is_legal_transition,
)


# -- ids ----------------------------------------------------------------------

class TestUlids:
    def test_shape(self) -> None:
        u = new_ulid()
        assert len(u) == 26
        assert is_ulid(u)

    def test_monotonic_within_burst(self) -> None:
        # Minting many ids as fast as possible forces same-millisecond
        # collisions; sort order must still equal creation order.
        ids = [new_ulid() for _ in range(5000)]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)  # and all unique

    def test_embedded_timestamp_is_roughly_now(self) -> None:
        u = new_ulid()
        delta = abs((utc_now() - ulid_timestamp(u)).total_seconds())
        assert delta < 5  # generous: same few seconds is all we claim

    def test_rejects_non_ulids(self) -> None:
        assert not is_ulid("hello")
        assert not is_ulid("")
        with pytest.raises(ValueError):
            ulid_timestamp("not-a-ulid")


# -- envelope -----------------------------------------------------------------

class _GreetingPayload(BaseModel):
    text: str


# Registration is import-time module state; do it once here for the tests.
register_kind("test.greeting", _GreetingPayload)


class TestEnvelope:
    def test_stage1_accepts_unknown_kind(self) -> None:
        # The outer frame must parse even for kinds we do not speak -
        # that is what lets a receiver reply instead of crash.
        e = Envelope(kind="future.mystery_kind", payload={"anything": 1})
        raw = e.model_dump(mode="json")
        back = Envelope.model_validate(raw)
        assert back.kind == "future.mystery_kind"

    def test_stage2_rejects_unknown_kind(self) -> None:
        e = Envelope(kind="future.mystery_kind")
        with pytest.raises(UnknownKind):
            e.parse_payload()

    def test_stage2_validates_registered_payload(self) -> None:
        e = Envelope(kind="test.greeting", payload={"text": "hello"})
        parsed = e.parse_payload()
        assert isinstance(parsed, _GreetingPayload)
        assert parsed.text == "hello"

    def test_stage2_rejects_bad_payload_for_known_kind(self) -> None:
        e = Envelope(kind="test.greeting", payload={"wrong_field": True})
        with pytest.raises(ValidationError):
            e.parse_payload()

    def test_make_envelope_enforces_kind_payload_match(self) -> None:
        env = make_envelope("test.greeting", _GreetingPayload(text="hi"))
        assert env.payload == {"text": "hi"}
        with pytest.raises(UnknownKind):
            make_envelope("test.unregistered", _GreetingPayload(text="hi"))

    def test_kind_shape_enforced(self) -> None:
        for bad in ("NoDots", "Upper.Case", "trailing.", ".leading", "a..b"):
            with pytest.raises(ValidationError):
                Envelope(kind=bad)


# -- events -------------------------------------------------------------------

class TestEvent:
    def test_valid_event_roundtrips(self) -> None:
        ev = Event(
            kind=EventKind.CORE_STARTED,
            source="test",
            trace_id=new_ulid(),
            payload={"n": 1},
        )
        back = Event.model_validate(ev.model_dump(mode="json"))
        assert back == ev

    def test_taxonomy_is_closed(self) -> None:
        with pytest.raises(ValidationError):
            Event(kind="core.invented", source="test", trace_id=new_ulid())

    def test_trace_id_required(self) -> None:
        with pytest.raises(ValidationError):
            Event.model_validate({"kind": EventKind.CORE_STARTED, "source": "test"})

    def test_events_are_immutable(self) -> None:
        ev = Event(kind=EventKind.CORE_STARTED, source="test", trace_id=new_ulid())
        with pytest.raises(ValidationError):
            ev.source = "tampered"  # type: ignore[misc]

    def test_taxonomy_constants_are_wellformed(self) -> None:
        # Every constant follows the dotted-lowercase convention.
        for kind in ALL_EVENT_KINDS:
            prefix = kind.split(".")[0]
            assert prefix in {
                "core", "session", "job", "worker",
                "memory", "initiative", "llm",
            }, kind


# -- jobs ---------------------------------------------------------------------

def _lease() -> Lease:
    now = utc_now()
    return Lease(worker_id="worker.test", leased_ts=now, heartbeat_ts=now, ttl_s=60)


class TestJobStateMachine:
    def test_state_machine_matches_design_doc(self) -> None:
        # The diagram from the data-and-jobs design doc, restated BY HAND.
        # If code and doc disagree, this fails and forces a decision.
        expected: dict[JobStatus, set[JobStatus]] = {
            JobStatus.QUEUED: {JobStatus.LEASED, JobStatus.CANCELLED},
            JobStatus.LEASED: {
                JobStatus.RUNNING, JobStatus.QUEUED, JobStatus.CANCELLED,
                # A worker can vanish on the final attempt, and a leased
                # job whose type has no handler is doomed on arrival -
                # both must be able to die without passing through
                # running. (Doc-02 amendment.)
                JobStatus.FAILED,
            },
            JobStatus.RUNNING: {
                JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.QUEUED,
                JobStatus.AWAITING_APPROVAL, JobStatus.CANCELLED,
            },
            JobStatus.AWAITING_APPROVAL: {JobStatus.RUNNING, JobStatus.CANCELLED},
            JobStatus.SUCCEEDED: set(),
            JobStatus.FAILED: set(),
            JobStatus.CANCELLED: set(),
        }
        assert {s: set(t) for s, t in ALLOWED_TRANSITIONS.items()} == expected

    def test_every_status_has_transition_entry(self) -> None:
        assert set(ALLOWED_TRANSITIONS) == set(JobStatus)

    def test_terminal_states_are_inescapable(self) -> None:
        for terminal in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
            for target in JobStatus:
                assert not is_legal_transition(terminal, target)


class TestJobInvariants:
    def test_minimal_valid_job(self) -> None:
        j = Job(type="research.brief", trace_id=new_ulid())
        assert j.status is JobStatus.QUEUED
        assert j.attempts == 0

    def test_leased_requires_lease(self) -> None:
        with pytest.raises(ValidationError):
            Job(type="a.b", trace_id=new_ulid(), status=JobStatus.LEASED)

    def test_queued_forbids_lease(self) -> None:
        with pytest.raises(ValidationError):
            Job(type="a.b", trace_id=new_ulid(), lease=_lease())

    def test_awaiting_approval_requires_record(self) -> None:
        with pytest.raises(ValidationError):
            Job(type="a.b", trace_id=new_ulid(), status=JobStatus.AWAITING_APPROVAL)
        ok = Job(
            type="a.b", trace_id=new_ulid(),
            status=JobStatus.AWAITING_APPROVAL,
            approval=Approval(gate="outbound", requested_ts=utc_now()),
        )
        assert ok.approval is not None

    def test_failed_requires_error_and_finished(self) -> None:
        with pytest.raises(ValidationError):
            Job(type="a.b", trace_id=new_ulid(), status=JobStatus.FAILED)
        ok = Job(
            type="a.b", trace_id=new_ulid(), status=JobStatus.FAILED,
            error="provider exploded", finished_ts=utc_now(),
        )
        assert ok.error

    def test_succeeded_forbids_error(self) -> None:
        with pytest.raises(ValidationError):
            Job(
                type="a.b", trace_id=new_ulid(), status=JobStatus.SUCCEEDED,
                error="but also failed?", finished_ts=utc_now(),
            )

    def test_type_shape(self) -> None:
        for bad in ("nodots", "Upper.case", "a.", ".b"):
            with pytest.raises(ValidationError):
                Job(type=bad, trace_id=new_ulid())