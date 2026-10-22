# =============================================================================
# tests/integration/test_projects.py - the projects database
# =============================================================================
#
# The promises: a debrief only ever stages proposals; handles map back
# to the right rows and dangling ones are dropped; approving in any order
# works; history is kept rather than overwritten; the page's API is
# behind the token; and the master document is versioned.
# =============================================================================

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import httpx
import pytest

from jarvis.common.ids import new_ulid
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.projects import ProjectReviewError, ProjectsRepo
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.projects.service import ProjectService
from jarvis.core.queue.registry import JobContext, JobTypeRegistry
from jarvis.jobs.projects import register_project_jobs
from jarvis.llm.layer import ToolCall

JAIPUR = ZoneInfo("Asia/Kolkata")
AUTH = {"Authorization": "Bearer test-token"}

FIRST = {
    "new_projects": [{"tag": "N1", "name": "Rover", "kind": "hardware",
                      "pitch": "A six-wheel rocker-bogie rover"}],
    "aliases": [{"project_ref": "new:N1", "alias": "the rocker bogie"}],
    "decisions": [{"project_ref": "new:N1", "decision": "Use a Raspberry Pi 4",
                   "why": "camera support"}],
    "facts": [{"project_ref": "new:N1", "category": "constraint",
               "text": "Must fit in a 30 cm cube"}],
    "parts": [{"project_ref": "new:N1", "name": "N20 gear motor", "quantity": 6,
               "status": "needed", "cost_inr": 1800}],
    "tasks": [{"project_ref": "new:N1", "text": "Print the bogie joints",
               "due_date": "2026-10-23"}],
    "log": [{"project_ref": "new:N1", "text": "Sketched the suspension",
             "happened_on": "2026-10-20"}],
    "people": [{"project_ref": "new:N1", "person_id": "H1", "role": "mechanical"}],
    "links": [{"project_ref": "new:N9", "label": "dangling", "url": "https://x.dev"}],
}


class ScriptedLLM:
    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []
        self.prompts: list[str] = []
        self.text = "# Rover\n\n## Abstract\nA rover."

    async def complete(self, tier: Any, system: str, messages: list[dict[str, Any]],
                       **kwargs: Any) -> Any:
        self.prompts.append(messages[0]["content"])
        if "tool_choice" not in kwargs:
            return SimpleNamespace(text=self.text, stop_reason="end_turn")
        return SimpleNamespace(stop_reason="tool_use", tool_calls=[
            ToolCall(id="t1", name="record_proposals", args=self.replies.pop(0))])


@pytest.fixture
async def world(db: Database) -> SimpleNamespace:
    repo = ProjectsRepo(db)
    llm = ScriptedLLM()
    now = datetime.now(timezone.utc).isoformat()
    aarav = new_ulid()
    await db.execute(
        "INSERT INTO people (id, name, created_ts, updated_ts) VALUES (?, 'Aarav', ?, ?)",
        (aarav, now, now))
    return SimpleNamespace(db=db, repo=repo, llm=llm, aarav=aarav,
                           service=ProjectService(repo, llm, JAIPUR))  # type: ignore[arg-type]


async def _approve_all(repo: ProjectsRepo) -> None:
    for debrief in await repo.pending_debriefs():
        for p in debrief["proposals"]:
            try:
                await repo.approve(p["id"])
            except ProjectReviewError:
                pass        # already approved as a dependency


async def test_a_debrief_only_stages(world: SimpleNamespace) -> None:
    world.llm.replies.append(FIRST)
    outcome = await world.service.debrief("Started the rover today...", trace_id=new_ulid())
    assert outcome.staged == 8 and outcome.dropped == 1          # the dangling link
    assert await world.repo.directory() == []                    # nothing real yet
    [debrief] = await world.repo.pending_debriefs()
    assert {p["project"] for p in debrief["proposals"]} == {"Rover"}
    person = next(p for p in debrief["proposals"] if p["kind"] == "person")
    assert person["payload"]["person_id"] == world.aarav         # H1 -> the real id


async def test_approving_out_of_order_creates_the_project_first(world: SimpleNamespace) -> None:
    world.llm.replies.append(FIRST)
    await world.service.debrief("Started the rover today...", trace_id=new_ulid())
    [debrief] = await world.repo.pending_debriefs()
    fact = next(p for p in debrief["proposals"] if p["kind"] == "fact")
    await world.repo.approve(fact["id"])                         # before the project
    [rover] = await world.repo.find("the rocker bogie") or await world.repo.find("Rover")
    assert rover["name"] == "Rover"
    await _approve_all(world.repo)
    profile = await world.repo.profile(rover["id"])
    assert profile["aliases"] == ["the rocker bogie"]
    assert profile["decisions"][0]["why"] == "camera support"
    assert profile["parts"][0]["quantity"] == 6
    assert profile["people"] == [{"id": world.aarav, "name": "Aarav", "role": "mechanical"}]
    assert await world.repo.pending_debriefs() == []


async def test_rejecting_a_new_project_takes_its_records_with_it(world: SimpleNamespace) -> None:
    world.llm.replies.append(FIRST)
    await world.service.debrief("Started the rover today...", trace_id=new_ulid())
    [debrief] = await world.repo.pending_debriefs()
    new = next(p for p in debrief["proposals"] if p["kind"] == "new_project")
    rejected = await world.repo.reject(new["id"])
    assert len(rejected) == len(debrief["proposals"])
    assert await world.repo.pending_debriefs() == []


async def test_the_second_debrief_supersedes_and_finishes(world: SimpleNamespace) -> None:
    world.llm.replies.append(FIRST)
    await world.service.debrief("Started the rover today...", trace_id=new_ulid())
    await _approve_all(world.repo)

    world.llm.replies.append({
        "decisions": [{"project_ref": "existing:J1", "decision": "Use an ESP32-S3",
                       "why": "the Pi drew too much power", "alternatives": "Raspberry Pi 4",
                       "supersedes_decision_id": "D1"}],
        "part_updates": [{"part_id": "B1", "status": "ordered", "note": "Robu"}],
        "tasks_done": [{"task_id": "T1"}, {"task_id": "T7"}],
    })
    outcome = await world.service.debrief("On the rover: switched to the ESP32...",
                                          trace_id=new_ulid())
    assert outcome.staged == 3 and outcome.dropped == 1
    assert "KNOWN RECORDS" in world.llm.prompts[-1] and "D1 | J1 Rover" in world.llm.prompts[-1]
    await _approve_all(world.repo)

    [rover] = await world.repo.find("Rover")
    profile = await world.repo.profile(rover["id"])
    assert [d["decision"] for d in profile["decisions"]] == ["Use an ESP32-S3"]
    assert profile["parts"][0]["status"] == "ordered"
    assert profile["tasks"] == []
    history = await world.repo.history(rover["id"])
    assert history["decisions"][0]["status"] == "superseded"


async def test_reading_projects_back(world: SimpleNamespace) -> None:
    world.llm.replies.append(FIRST)
    await world.service.debrief("Started the rover today...", trace_id=new_ulid())
    await _approve_all(world.repo)

    context = await world.service.mention_context("how is the rocker bogie coming along?")
    assert context.startswith("Rover [active, hardware]")
    assert "Decided: Use a Raspberry Pi 4 - because camera support" in context
    found = await world.service.lookup("suspension")             # full text, not a name
    assert "these mention it" in found and "Sketched the suspension" in found
    assert "Rover [active" in await world.service.overview()
    assert await world.service.note("Rover", "Wheels printed") == "Logged on Rover."
    [rover] = await world.repo.find("Rover")
    assert (await world.repo.profile(rover["id"]))["log"][0]["text"] == "Wheels printed"
    assert [t["text"] for t in await world.repo.tasks_due("2026-10-25")] == \
        ["Print the bogie joints"]
    later = (datetime.now(timezone.utc) + timedelta(days=20)).isoformat()
    assert [p["name"] for p in await world.repo.stale(14, later)] == ["Rover"]


async def test_the_master_document_is_versioned(world: SimpleNamespace) -> None:
    world.llm.replies.append(FIRST)
    await world.service.debrief("Started the rover today...", trace_id=new_ulid())
    await _approve_all(world.repo)
    [rover] = await world.repo.find("Rover")

    registry = JobTypeRegistry()
    register_project_jobs(registry, world.llm, world.repo, world.service)
    spec = registry.get("project.document")
    assert spec is not None and spec.handler is not None
    written: list[str] = []

    async def write(name: str, mime: str, data: bytes) -> str:
        written.append(name)
        return new_ulid()

    async def nothing(*_: Any) -> None: ...

    ctx = JobContext(job_id=new_ulid(), trace_id=new_ulid(), checkpoint=None,
                     save_checkpoint=nothing, progress=nothing, write_artifact=write)
    payload = spec.input_model.model_validate({"project_id": rover["id"]})
    first = await spec.handler(payload, ctx)
    second = await spec.handler(payload, ctx)
    assert (first.version, second.version) == (1, 2)               # type: ignore[attr-defined]
    assert written == ["rover-master-v1.md", "rover-master-v2.md"]
    assert "Raspberry Pi 4" in world.llm.prompts[-1]              # written from the records


@pytest.fixture
async def client(settings: CoreSettings, world: SimpleNamespace) -> AsyncIterator[httpx.AsyncClient]:
    deps = GatewayDeps(settings=settings, db=world.db, traces=TracesRepo(world.db),
                       started_monotonic=time.monotonic(),
                       projects_repo=world.repo, projects=world.service)
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_the_page_and_its_api(client: httpx.AsyncClient, world: SimpleNamespace) -> None:
    page = await client.get("/app/projects")
    assert page.status_code == 200 and "Waiting for review" in page.text
    assert (await client.get("/projects/pending")).status_code == 401

    world.llm.replies.append(FIRST)
    sent = await client.post("/projects/debrief", headers=AUTH,
                             json={"text": "Started the rover today..."})
    assert sent.json()["staged"] == 8
    [debrief] = (await client.get("/projects/pending", headers=AUTH)).json()["debriefs"]
    new = next(p for p in debrief["proposals"] if p["kind"] == "new_project")
    edited = {**new["payload"], "name": "Mars Rover"}
    ok = await client.post(f"/projects/proposals/{new['id']}/approve", headers=AUTH,
                           json={"payload": edited})
    assert ok.status_code == 200
    again = await client.post(f"/projects/proposals/{new['id']}/approve", headers=AUTH, json={})
    assert again.status_code == 409

    [row] = (await client.get("/projects/directory", headers=AUTH)).json()["projects"]
    assert row["name"] == "Mars Rover"
    pid = row["id"]
    assert (await client.patch(f"/projects/project/{pid}", headers=AUTH,
                               json={"status": "paused"})).status_code == 200
    assert (await client.post(f"/projects/project/{pid}/log", headers=AUTH,
                              json={"text": "Shelved for exams"})).status_code == 200
    full = (await client.get(f"/projects/project/{pid}", headers=AUTH)).json()
    assert full["status"] == "paused" and full["log"][0]["text"] == "Shelved for exams"
    assert json.dumps(full["history"])
    assert (await client.post(f"/projects/project/{pid}/document", headers=AUTH)).status_code == 503


async def test_the_morning_brief_mentions_project_work(world: SimpleNamespace) -> None:
    from jarvis.core.db.repos.notifications import NotificationsRepo
    from jarvis.core.db.repos.sessions import SessionsRepo
    from jarvis.core.initiative.brief import BriefService

    world.llm.replies.append(FIRST)
    await world.service.debrief("Started the rover today...", trace_id=new_ulid())
    await _approve_all(world.repo)
    world.llm.replies.append({"tasks": [{"project_ref": "existing:J1", "text": "Wire it"}]})
    await world.service.debrief("Rover: still need to wire it", trace_id=new_ulid())

    nobody = SimpleNamespace(all_pending=_empty, all=_empty)
    brief = BriefService(world.db, people=None, approvals=nobody, watchers=nobody,  # type: ignore[arg-type]
                         sessions=SessionsRepo(world.db),
                         notifications=NotificationsRepo(world.db), llm=None, tz=JAIPUR,
                         projects=world.repo)
    facts = await brief.gather(datetime(2026, 10, 22, 7, 35, tzinfo=JAIPUR))
    assert facts.sections["Projects"] == [
        "Rover: Print the bogie joints (due Fri 23 Oct)",
        "1 projects-database proposals to review",
    ]


async def _empty() -> list[Any]:
    return []
