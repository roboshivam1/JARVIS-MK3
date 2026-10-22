# =============================================================================
# tests/integration/test_brief.py - the morning brief
# =============================================================================
#
# What it promises: it finds the things that need the owner, leaves out
# what does not, never invents (the model only phrases), survives the
# model being down, and cannot be sent twice by a retried job.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

from jarvis.common.ids import new_ulid
from jarvis.common.jobs import Job, JobStatus
from jarvis.common.reels import EpisodeStatus, EpisodeType, ReelBrief, ReelEpisode
from jarvis.common.schedules import Schedule, ScheduleKind
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.people import PeopleRepo
from jarvis.core.db.repos.reels import ReelsRepo
from jarvis.core.db.repos.schedules import SchedulesRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.initiative.brief import BriefService
from jarvis.core.initiative.engine import next_cron_time

JAIPUR = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 21, 7, 35, tzinfo=JAIPUR)


class FakeLLM:
    def __init__(self, text: str = "Good morning, sir.", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.prompts: list[str] = []

    async def complete(self, tier: Any, system: str, messages: list[dict[str, Any]],
                       **kwargs: Any) -> Any:
        self.prompts.append(messages[0]["content"])
        if self.fail:
            raise RuntimeError("overloaded")
        return SimpleNamespace(text=self.text, stop_reason="end_turn")


class FakeApprovals:
    def __init__(self, summaries: list[str] | None = None) -> None:
        self.summaries = summaries or []

    async def all_pending(self) -> list[Any]:
        return [SimpleNamespace(summary=s) for s in self.summaries]


class FakeWatchers:
    def __init__(self, hits: list[tuple[str, datetime | None]] | None = None) -> None:
        self.hits = hits or []

    async def all(self) -> list[Any]:
        return [SimpleNamespace(name=n, note="", last_hit_ts=ts) for n, ts in self.hits]


def _service(db: Database, llm: Any = None, approvals: Any = None,
             watchers: Any = None) -> BriefService:
    return BriefService(
        db, people=PeopleRepo(db), approvals=approvals or FakeApprovals(),
        watchers=watchers or FakeWatchers(), sessions=SessionsRepo(db),
        notifications=NotificationsRepo(db), llm=llm, tz=JAIPUR,
    )


async def _backup_last_night(db: Database) -> None:
    await _finished_job(db, "system.backup", JobStatus.SUCCEEDED, hours_ago=3)


async def _finished_job(db: Database, kind: str, status: JobStatus, hours_ago: float,
                        error: str | None = None) -> None:
    at = NOW - timedelta(hours=hours_ago)
    await JobsRepo(db, EventsRepo(db)).create(Job(
        type=kind, status=status, error=error, trace_id=new_ulid(),
        created_ts=at, updated_ts=at, finished_ts=at,
        result={} if status is JobStatus.SUCCEEDED else None,
    ))


async def _person(db: Database, name: str) -> str:
    pid = new_ulid()
    stamp = NOW.isoformat()
    await db.execute(
        "INSERT INTO people (id, name, created_ts, updated_ts) VALUES (?, ?, ?, ?)",
        (pid, name, stamp, stamp))
    return pid


async def test_a_quiet_morning_is_one_line_and_costs_nothing(db: Database) -> None:
    await _backup_last_night(db)
    llm = FakeLLM()
    text, facts = await _service(db, llm).compose(new_ulid(), NOW)
    assert facts.empty
    assert text.startswith("Good morning, sir. Wednesday 21 October.")
    assert "Nothing needs you" in text
    assert llm.prompts == []


async def test_it_finds_what_needs_him(db: Database) -> None:
    riya = await _person(db, "Riya")
    stamp = NOW.isoformat()
    await db.execute(
        "INSERT INTO person_commitments (id, person_id, direction, what, due_date, "
        "created_ts) VALUES (?, ?, 'i_owe', 'send the CAD files', '2026-10-19', ?)",
        (new_ulid(), riya, stamp))
    await db.execute(
        "INSERT INTO person_commitments (id, person_id, direction, what, due_date, "
        "created_ts) VALUES (?, ?, 'they_owe', 'return the multimeter', '2026-12-01', ?)",
        (new_ulid(), riya, stamp))
    await db.execute(
        "INSERT INTO person_dates (id, person_id, label, date, recurring, created_ts) "
        "VALUES (?, ?, 'birthday', '10-24', 1, ?)", (new_ulid(), riya, stamp))

    await _finished_job(db, "research.brief", JobStatus.FAILED, 5, error="timed out\ntrace")
    await _finished_job(db, "watch.check", JobStatus.SUCCEEDED, 2)       # routine
    await _finished_job(db, "write.document", JobStatus.SUCCEEDED, 4)
    await _finished_job(db, "code.task", JobStatus.FAILED, 30, error="old news")

    await ReelsRepo(db).create(ReelEpisode(
        kit_id="k", type=EpisodeType.FIELD_NOTES, title="Five more minutes",
        request="five more minutes", status=EpisodeStatus.DELIVERED,
        brief=ReelBrief.model_validate({
            "episode_type": "field_notes", "premise": "Five more minutes means ninety.",
            "beats": ["a", "b", "c"], "facts": ["f"]}),
    ))
    await db.execute(
        "INSERT INTO llm_calls (id, ts, trace_id, actor, tier, model, latency_ms, "
        "tokens_in, tokens_out, cached_tokens, cost_usd, cost_inr, stop_reason) "
        "VALUES (?, ?, ?, 'x', 'utility', 'm', 1, 1, 1, 0, 0.5, 42.0, 'end_turn')",
        (new_ulid(), (NOW - timedelta(hours=12)).astimezone(timezone.utc).isoformat(),
         new_ulid()))

    service = _service(
        db, FakeLLM(),
        approvals=FakeApprovals(["Run the bench test script"]),
        watchers=FakeWatchers([("bambu firmware", NOW - timedelta(hours=6)),
                               ("old watcher", NOW - timedelta(days=3))]),
    )
    facts = await service.gather(NOW)
    text = facts.as_text()

    assert "approval waiting: Run the bench test script" in text
    assert "sir owes Riya: send the CAD files (OVERDUE since Mon 19 Oct)" in text
    assert "multimeter" not in text                      # due beyond the week
    assert "Riya: birthday (Sat 24 Oct)" in text
    assert "FAILED: research.brief - timed out" in text
    assert "old news" not in text                        # before the window
    assert "1 jobs finished in the last day" in text     # watch.check is routine
    assert "watcher 'bambu firmware' fired" in text
    assert "old watcher" not in text
    assert "no successful backup" in text
    assert "reel 'Five more minutes' is delivered but not marked posted" in text
    assert "yesterday: Rs 42 over 1 model calls" in text
    assert list(facts.sections)[0] == "Waiting on sir"


async def test_the_model_only_phrases_and_its_failure_is_survivable(db: Database) -> None:
    await _finished_job(db, "research.brief", JobStatus.FAILED, 2, error="boom")
    llm = FakeLLM(text="Good morning, sir. One failure overnight.")
    text, _ = await _service(db, llm).compose(new_ulid(), NOW)
    assert text == "Good morning, sir. One failure overnight."
    assert "Today is Wednesday 21 October." in llm.prompts[0]
    assert "FAILED: research.brief - boom" in llm.prompts[0]

    text, _ = await _service(db, FakeLLM(fail=True)).compose(new_ulid(), NOW)
    assert text.startswith("Good morning, sir. Wednesday 21 October.")
    assert "- FAILED: research.brief - boom" in text


async def test_a_retried_job_cannot_send_it_twice(db: Database) -> None:
    await _backup_last_night(db)
    service = _service(db, FakeLLM())
    job_id = new_ulid()
    await service.send(new_ulid(), job_id=job_id, now=NOW)
    await service.send(new_ulid(), job_id=job_id, now=NOW)

    pending = await NotificationsRepo(db).pending()
    assert len(pending) == 1
    note = pending[0]
    assert note.client_kind == "telegram" and note.priority == 3
    session = await SessionsRepo(db).get_or_create_default("telegram")
    assert note.session_id == session.id


async def test_on_demand_is_recorded_in_the_thread(db: Database) -> None:
    await _backup_last_night(db)
    text = await _service(db, FakeLLM()).on_demand(new_ulid())
    sessions = SessionsRepo(db)
    session = await sessions.get_or_create_default("telegram")
    turns = await sessions.recent_turns(session.id)
    assert [t.content for t in turns] == [text]


async def test_the_schedule_can_be_switched_and_moved(db: Database) -> None:
    repo = SchedulesRepo(db)
    await repo.ensure(Schedule(
        name="morning brief", kind=ScheduleKind.CRON, cron_expr="35 7 * * *",
        job_type="brief.morning", next_fire_ts=NOW - timedelta(days=5),
    ))
    assert await repo.set_enabled("morning brief", False)
    assert (await repo.by_name("morning brief")).enabled is False  # type: ignore[union-attr]

    fresh = next_cron_time("35 7 * * *", JAIPUR)
    assert await repo.set_enabled("morning brief", True, fresh)
    assert await repo.due() == []                # a stale time would fire at once

    assert await repo.set_cron("morning brief", "15 8 * * *", fresh)
    moved = await repo.by_name("morning brief")
    assert moved is not None and moved.cron_expr == "15 8 * * *"
    assert not await repo.set_enabled("no such thing", True)
    assert await repo.by_name("no such thing") is None
