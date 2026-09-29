# =============================================================================
# tests/integration/test_people_directory.py - groups, profiles, owner edits
# =============================================================================
#
# What must hold:
#   - the owner's edits skip review but never destroy history: an edited
#     fact supersedes, a retired one stays readable
#   - merging two rows for one person loses nothing, and re-points
#     proposals still waiting on the absorbed row
#   - groups are many-to-many, and a group is something JARVIS can name
# =============================================================================

from __future__ import annotations

import time
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import httpx
import pytest

from jarvis.common.people import ExtractionResult
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.people import PeopleRepo
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.people.service import PeopleService

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def repo(db: Database) -> PeopleRepo:
    return PeopleRepo(db)


@pytest.fixture
async def client(settings: CoreSettings, db: Database,
                 repo: PeopleRepo) -> AsyncIterator[httpx.AsyncClient]:
    deps = GatewayDeps(
        settings=settings, db=db, traces=TracesRepo(db),
        started_monotonic=time.monotonic(), people_repo=repo,
        people=PeopleService(repo, None, ZoneInfo("Asia/Kolkata")),  # type: ignore[arg-type]
    )
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _person(repo: PeopleRepo, name: str, fact: str = "Studies ECE") -> str:
    """One approved person, through the real staging and review path."""
    result = ExtractionResult.model_validate({
        "new_people": [{"tag": "P1", "name": name, "relationship": "friend"}],
        "facts": [{"person_ref": "new:P1", "category": "work_study", "text": fact}],
    })
    await repo.stage(f"met {name}", result, "web")
    debrief = (await repo.pending_debriefs())[0]
    for p in debrief["proposals"]:
        await repo.approve(p["id"])
    return (await repo.find(name))[0]["id"]


async def test_editing_a_fact_keeps_the_old_one_readable(
    client: httpx.AsyncClient, repo: PeopleRepo
) -> None:
    person_id = await _person(repo, "Aarav Mehta")
    detail = (await client.get(f"/people/person/{person_id}", headers=AUTH)).json()
    [fact] = detail["facts"]
    assert detail["debriefs"][fact["debrief_id"]]["raw_text"] == "met Aarav Mehta"

    edited = await client.patch(f"/people/facts/{fact['id']}", headers=AUTH, json={
        "category": "work_study", "text": "Studies CSE", "sensitivity": "normal"})
    assert edited.status_code == 200

    detail = (await client.get(f"/people/person/{person_id}", headers=AUTH)).json()
    assert [f["text"] for f in detail["facts"]] == ["Studies CSE"]
    assert [f["text"] for f in detail["history"]] == ["Studies ECE"]
    assert detail["history"][0]["superseded_by"] == detail["facts"][0]["id"]

    # Retiring leaves it in history with nothing replacing it.
    await client.delete(f"/people/facts/{detail['facts'][0]['id']}", headers=AUTH)
    detail = (await client.get(f"/people/person/{person_id}", headers=AUTH)).json()
    assert detail["facts"] == []
    assert {f["text"] for f in detail["history"]} == {"Studies ECE", "Studies CSE"}
    assert detail["history"][0]["superseded_by"] is None

async def test_groups_are_many_to_many_and_reach_jarvis(
    client: httpx.AsyncClient, repo: PeopleRepo, settings: CoreSettings
) -> None:
    person_id = await _person(repo, "Aarav Mehta")
    college = (await client.post("/people/groups", headers=AUTH, json={
        "name": "LNMIIT", "kind": "college"})).json()["group_id"]
    team = (await client.post("/people/groups", headers=AUTH, json={
        "name": "Robofest team", "kind": "project"})).json()["group_id"]
    for group_id in (college, team):
        r = await client.post(f"/people/groups/{group_id}/members", headers=AUTH,
                              json={"person_id": person_id})
        assert r.status_code == 200

    clash = await client.post("/people/groups", headers=AUTH,
                              json={"name": "lnmiit", "kind": "college"})
    assert clash.status_code == 409          # names are case-insensitively unique

    directory = (await client.get("/people/directory", headers=AUTH)).json()
    assert {g["name"] for g in directory["groups"]} == {"LNMIIT", "Robofest team"}
    assert sorted(directory["people"][0]["groups"]) == sorted([college, team])

    # What JARVIS is handed: groups on the card, and a group he can name.
    service = PeopleService(repo, None, ZoneInfo("Asia/Kolkata"))  # type: ignore[arg-type]
    assert "Groups: LNMIIT, Robofest team" in await service.describe(person_id)
    assert "Aarav Mehta" in await service.lookup("Robofest team")

    # Removing the group leaves the person alone.
    await client.delete(f"/people/groups/{team}", headers=AUTH)
    detail = (await client.get(f"/people/person/{person_id}", headers=AUTH)).json()
    assert [g["name"] for g in detail["groups"]] == ["LNMIIT"]


async def test_merging_two_rows_for_one_person(
    client: httpx.AsyncClient, repo: PeopleRepo
) -> None:
    keep = await _person(repo, "Aarav Mehta", fact="Studies ECE")
    merge = await _person(repo, "Aarav M", fact="Builds drones")
    group = (await client.post("/people/groups", headers=AUTH, json={
        "name": "LNMIIT", "kind": "college"})).json()["group_id"]
    await client.post(f"/people/groups/{group}/members", headers=AUTH,
                      json={"person_id": merge})

    # A proposal still waiting on the row about to be absorbed.
    await repo.stage("more about Aarav", ExtractionResult.model_validate({
        "facts": [{"person_ref": f"existing:{merge}", "category": "skills",
                   "text": "Good with PCB design"}]}), "web")

    merged = await client.post("/people/merge", headers=AUTH,
                               json={"keep": keep, "merge": merge})
    assert merged.status_code == 200

    detail = (await client.get(f"/people/person/{keep}", headers=AUTH)).json()
    assert {f["text"] for f in detail["facts"]} == {"Studies ECE", "Builds drones"}
    assert "Aarav M" in detail["aliases"]
    assert [g["id"] for g in detail["groups"]] == [group]
    assert await repo.find("Aarav M") == [] or (await repo.find("Aarav M"))[0]["id"] == keep

    # The waiting proposal now points at the surviving row, and approving
    # it works rather than failing on a person that no longer exists.
    pending = [p for d in await repo.pending_debriefs() for p in d["proposals"]]
    [proposal] = [p for p in pending if p["payload"].get("text") == "Good with PCB design"]
    assert proposal["payload"]["person_ref"] == f"existing:{keep}"
    await repo.approve(proposal["id"])


async def test_owner_edits_a_person_and_its_bits(
    client: httpx.AsyncClient, repo: PeopleRepo
) -> None:
    person_id = await _person(repo, "Aarav Mehta")
    await client.patch(f"/people/person/{person_id}", headers=AUTH,
                       json={"closeness": 1, "summary": "ECE friend"})
    await client.post(f"/people/person/{person_id}/alias", headers=AUTH,
                      json={"alias": "AM"})
    await client.post(f"/people/person/{person_id}/facts", headers=AUTH,
                      json={"category": "preferences", "text": "Vegetarian"})

    detail = (await client.get(f"/people/person/{person_id}", headers=AUTH)).json()
    assert detail["closeness"] == 1 and detail["summary"] == "ECE friend"
    assert detail["aliases"] == ["AM"]
    assert "Vegetarian" in {f["text"] for f in detail["facts"]}

    await client.post(f"/people/person/{person_id}/alias", headers=AUTH,
                      json={"alias": "AM", "present": False})
    assert (await client.get(f"/people/person/{person_id}",
                             headers=AUTH)).json()["aliases"] == []

    assert (await client.get("/people/person/01ARZ3NDEKTSV4RRFFQ69G5FAV",
                             headers=AUTH)).status_code == 404
    assert (await client.get("/people/directory")).status_code == 401


async def test_summary_feeds_the_home_panel(client: httpx.AsyncClient, repo: PeopleRepo) -> None:
    person_id = await _person(repo, "Aarav Mehta")
    await repo.stage("someone new", ExtractionResult.model_validate({
        "new_people": [{"tag": "P1", "name": "Kabir Shah"}]}), "conversation")

    summary = (await client.get("/people/summary", headers=AUTH)).json()
    assert summary["pending_debriefs"] == 1 and summary["pending_proposals"] == 1
    assert summary["people"] == 1 and summary["groups"] == 0
    assert summary["commitments"] == [] and summary["dates"] == []

    await repo.stage("owes me", ExtractionResult.model_validate({
        "commitments": [{"person_ref": f"existing:{person_id}", "direction": "they_owe",
                         "what": "Schematics"}],
        "key_dates": [{"person_ref": f"existing:{person_id}", "label": "birthday",
                       "date": "02-14"}]}), "web")
    for d in await repo.pending_debriefs():
        for p in d["proposals"]:
            if p["kind"] in ("commitment", "key_date"):
                await repo.approve(p["id"])

    summary = (await client.get("/people/summary", headers=AUTH)).json()
    assert summary["commitments"][0]["name"] == "Aarav Mehta"
    assert summary["dates"][0]["date"] == "02-14"


async def test_the_page_carries_its_script(client: httpx.AsyncClient) -> None:
    page = await client.get("/app/people")
    assert 'src="/app/people-directory.js"' in page.text
    script = await client.get("/app/people-directory.js")
    assert script.status_code == 200
    assert "renderDirectory" in script.text
    assert script.headers["content-type"].startswith("application/javascript")
