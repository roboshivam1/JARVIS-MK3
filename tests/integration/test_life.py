# =============================================================================
# tests/integration/test_life.py - the life log
# =============================================================================
#
# The promises: a report lands as rows and ticks the right checklist
# items; started work can be stopped and timed; undo takes back exactly
# one report; finished days turn open items into misses; the check-in,
# the brief and the trends read it all back correctly; the page's API is
# behind the token.
# =============================================================================

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import httpx
import pytest

from jarvis.common.ids import new_ulid
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.life import LifeRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.projects import ProjectsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.initiative.brief import BriefService
from jarvis.core.life.service import LifeService
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.queue.registry import JobContext, JobTypeRegistry
from jarvis.jobs.life import register_life_jobs
from jarvis.llm.layer import ToolCall

JAIPUR = ZoneInfo("Asia/Kolkata")
MONDAY = datetime(2026, 10, 26, 9, 0, tzinfo=JAIPUR)
AUTH = {"Authorization": "Bearer test-token"}


class ScriptedLLM:
    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []
        self.prompts: list[str] = []

    async def complete(self, tier: Any, system: str, messages: list[dict[str, Any]],
                       **kwargs: Any) -> Any:
        self.prompts.append(system + "\n" + messages[0]["content"])
        return SimpleNamespace(stop_reason="tool_use", tool_calls=[
            ToolCall(id="t1", name="record_report", args=self.replies.pop(0))])


@pytest.fixture
def world(db: Database) -> SimpleNamespace:
    repo = LifeRepo(db)
    llm = ScriptedLLM()
    return SimpleNamespace(db=db, repo=repo, llm=llm,
                           life=LifeService(repo, llm, JAIPUR, projects=ProjectsRepo(db)))  # type: ignore[arg-type]


async def _say(world: SimpleNamespace, reply: dict[str, Any], at: datetime = MONDAY) -> str:
    world.llm.replies.append(reply)
    outcome = await world.life.report("...", trace_id=new_ulid(), moment=at)
    return str(outcome.summary())


async def test_gym_is_on_the_checklist_monday_to_saturday(world: SimpleNamespace) -> None:
    await world.life.prepare(MONDAY)
    assert [i["text"] for i in await world.repo.checklist("2026-10-26")] == ["Gym"]
    await world.repo.ensure_checklist("2026-11-01")                  # a Sunday
    assert await world.repo.checklist("2026-11-01") == []


async def test_a_report_logs_the_day_and_ticks_the_gym(world: SimpleNamespace) -> None:
    summary = await _say(world, {
        "wake": {"date": "2026-10-26", "time": "7:10"},
        "activities": [{"date": "2026-10-26", "kind": "gym", "subject": "legs", "minutes": 50},
                       {"date": "2026-10-26", "kind": "study", "subject": "OS", "minutes": 120},
                       {"date": "nonsense", "kind": "gym"}],
        "meals": [{"date": "2026-10-26", "slot": "lunch", "what": "dal rice", "place": "mess"},
                  {"date": "2026-10-26", "slot": "breakfast", "status": "skipped"}],
        "new_targets": [{"date": "2026-10-27", "text": "Finish one Coursera module"}],
    })
    assert "up at 07:10" in summary and "ticked Gym" in summary
    assert "target for 2026-10-27: Finish one Coursera module" in summary
    [gym] = await world.repo.checklist("2026-10-26")
    assert gym["status"] == "done"
    tomorrow = [i["text"] for i in await world.repo.checklist("2026-10-27")]
    assert tomorrow == ["Gym", "Finish one Coursera module"]
    status = await world.life.status_text("2026-10-26")
    assert "Woke: 07:10" in status and "study 2h00" in status
    assert "Meals: lunch; skipped breakfast" in status


async def test_targets_are_ticked_by_handle_and_misses_are_kept(world: SimpleNamespace) -> None:
    await _say(world, {"new_targets": [{"date": "2026-10-26", "text": "Coursera module"}]})
    await _say(world, {"done": ["C2", "C99"]})
    assert "C2 | today | Coursera module | open" in world.llm.prompts[-1]
    items = {i["text"]: i["status"] for i in await world.repo.checklist("2026-10-26")}
    assert items == {"Gym": "open", "Coursera module": "done"}

    await world.life.prepare(datetime(2026, 10, 27, 8, 0, tzinfo=JAIPUR))
    items = {i["text"]: i["status"] for i in await world.repo.checklist("2026-10-26")}
    assert items == {"Gym": "missed", "Coursera module": "done"}


async def test_start_then_stop_is_timed(world: SimpleNamespace) -> None:
    await _say(world, {"activities": [{"date": "2026-10-26", "kind": "learning",
                                       "action": "start", "subject": "Coursera"}]})
    assert len(await world.repo.running()) == 1
    later = datetime(2026, 10, 26, 10, 35, tzinfo=JAIPUR)
    summary = await _say(world, {"activities": [{"date": "2026-10-26", "kind": "learning",
                                                 "action": "stop"}]}, at=later)
    assert "learning finished (1h35)" in summary
    assert await world.repo.running() == []


async def test_undo_takes_back_exactly_the_last_report(world: SimpleNamespace) -> None:
    await _say(world, {"wake": {"date": "2026-10-26", "time": "07:00"}})
    await _say(world, {"wake": {"date": "2026-10-26", "time": "09:00"},
                       "activities": [{"date": "2026-10-26", "kind": "gym", "minutes": 40}],
                       "new_targets": [{"date": "2026-10-26", "text": "Wiring"}]})
    assert (await world.life.undo_last()).startswith("Undone: Logged: up at 09:00")
    assert (await world.repo.day("2026-10-26"))["wake_time"] is None   # field cleared
    assert await world.repo.activities("2026-10-26") == []
    assert [(i["text"], i["status"]) for i in await world.repo.checklist("2026-10-26")] == \
        [("Gym", "open")]


async def test_iphone_events(world: SimpleNamespace) -> None:
    assert await world.life.ping("wake", MONDAY) == "Up at 09:00."
    assert await world.life.ping("gym_start", datetime(2026, 10, 26, 18, 0, tzinfo=JAIPUR)) \
        == "Gym started at 18:00."
    assert await world.life.ping("gym_end", datetime(2026, 10, 26, 19, 5, tzinfo=JAIPUR)) \
        == "Gym done, 1h05."
    assert await world.life.ping("sleep", datetime(2026, 10, 27, 1, 30, tzinfo=JAIPUR)) \
        == "Asleep at 01:30."
    assert (await world.repo.day("2026-10-26"))["sleep_time"] == "01:30"   # the day that ended
    assert (await world.repo.checklist("2026-10-26"))[0]["status"] == "done"


async def test_the_evening_check_in_and_the_brief(world: SimpleNamespace) -> None:
    await _say(world, {"meals": [{"date": "2026-10-26", "slot": "lunch"}],
                       "new_targets": [{"date": "2026-10-27", "text": "Coursera module"}]})
    text = await world.life.checkin_text(datetime(2026, 10, 26, 22, 0, tzinfo=JAIPUR))
    assert "☐ Gym" in text
    assert "Not logged yet: when you woke, breakfast, dinner, what you worked on." in text
    assert "Tomorrow (Tue 27 Oct):\n- Gym\n- Coursera module" in text

    sections = await world.life.brief_sections(datetime(2026, 10, 27, 7, 35, tzinfo=JAIPUR))
    assert sections["Checklist for today"] == ["Gym", "Coursera module"]
    assert sections["Yesterday"] == ["0 of 1 checklist items done (missed: Gym)"]

    nobody = SimpleNamespace(all_pending=_empty, all=_empty)
    brief = BriefService(world.db, people=None, approvals=nobody, watchers=nobody,  # type: ignore[arg-type]
                         sessions=SessionsRepo(world.db), notifications=NotificationsRepo(world.db),
                         llm=None, tz=JAIPUR, life=world.life)
    facts = await brief.gather(datetime(2026, 10, 27, 7, 35, tzinfo=JAIPUR))
    assert list(facts.sections)[:2] == ["Checklist for today", "Yesterday"]


async def test_the_check_in_job_queues_one_message(world: SimpleNamespace) -> None:
    registry = JobTypeRegistry()
    register_life_jobs(registry, world.life, SessionsRepo(world.db), NotificationsRepo(world.db))
    spec = registry.get("life.checkin")
    assert spec is not None and spec.handler is not None

    async def nothing(*_: Any) -> None: ...

    async def no_file(*_: Any) -> str:
        return new_ulid()

    ctx = JobContext(job_id=new_ulid(), trace_id=new_ulid(), checkpoint=None,
                     save_checkpoint=nothing, progress=nothing, write_artifact=no_file)
    payload = spec.input_model.model_validate({})
    await spec.handler(payload, ctx)
    await spec.handler(payload, ctx)                              # a retry
    [note] = await NotificationsRepo(world.db).pending()
    assert note.text.startswith("Evening check-in, sir.") and note.priority == 3


async def test_trends(world: SimpleNamespace) -> None:
    await world.life.set_wake_by("7:00")
    await _say(world, {"wake": {"date": "2026-10-26", "time": "07:30"},
                       "activities": [{"date": "2026-10-26", "kind": "gym", "minutes": 45},
                                      {"date": "2026-10-26", "kind": "learning", "minutes": 30}]})
    stats = await world.life.stats(7, datetime(2026, 10, 27, 20, 0, tzinfo=JAIPUR))
    monday = next(d for d in stats["days"] if d["date"] == "2026-10-26")
    assert monday["gym_done"] and monday["minutes"]["learning"] == 30 and monday["done"] == 1
    assert stats["wake_by"] == "07:00"
    week = await world.life.week_text()
    assert "target 07:00" in week


async def _empty() -> list[Any]:
    return []


@pytest.fixture
async def client(settings: CoreSettings, world: SimpleNamespace) -> AsyncIterator[httpx.AsyncClient]:
    deps = GatewayDeps(settings=settings, db=world.db, traces=TracesRepo(world.db),
                       started_monotonic=time.monotonic(), life_repo=world.repo, life=world.life)
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_the_page_and_its_api(client: httpx.AsyncClient, world: SimpleNamespace) -> None:
    page = await client.get("/app/life")
    assert page.status_code == 200 and "Fixed targets" in page.text
    assert (await client.get("/life/day")).status_code == 401

    today = (await client.get("/life/day", headers=AUTH)).json()["today"]
    added = await client.post("/life/checklist", headers=AUTH,
                              json={"date": today, "text": "Read one paper"})
    item = added.json()["id"]
    assert (await client.post(f"/life/checklist/{item}", headers=AUTH,
                              json={"status": "done"})).status_code == 200
    summary = (await client.get("/life/summary", headers=AUTH)).json()
    assert summary["done"] >= 1 and "Read one paper" not in summary["open"]

    assert (await client.post("/life/ping", headers=AUTH, json={"event": "wake"})).status_code == 200
    assert (await client.post("/life/ping", headers=AUTH, json={"event": "nap"})).status_code == 422

    made = await client.post("/life/habits", headers=AUTH,
                             json={"name": "Read", "kind": "learning", "weekdays": ["sun"]})
    assert made.status_code == 200
    clash = await client.post("/life/habits", headers=AUTH, json={"name": "read"})
    assert clash.status_code == 409
    assert (await client.post("/life/settings", headers=AUTH,
                              json={"wake_by": "25:00"})).status_code == 422
    habits = (await client.get("/life/habits", headers=AUTH)).json()["habits"]
    assert [h["name"] for h in habits] == ["Gym", "Read"]
    assert (await client.get("/life/stats?days=28", headers=AUTH)).status_code == 200


async def test_home_page_links_every_section(client: httpx.AsyncClient) -> None:
    page = (await client.get("/app")).text
    for link in ('href="/app/life"', 'href="/app/projects"', 'href="/app/people"'):
        assert link in page
