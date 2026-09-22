# =============================================================================
# tests/integration/test_people_gateway.py - the review page's API
# =============================================================================
#
# The page is only as trustworthy as the routes behind it: every one of
# them behind the token, every failure mapped to a status the page can
# explain, and an edit made in the browser held to the same validation
# as the model's own output.
# =============================================================================

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import httpx
import pytest

from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.people import PeopleRepo
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.people.service import PeopleService
from jarvis.llm.layer import NoApiKeyError, ToolCall

AUTH = {"Authorization": "Bearer test-token"}

DEBRIEF = {
    "new_people": [{"tag": "P1", "name": "Aarav Mehta", "relationship": "friend"}],
    "facts": [{"person_ref": "new:P1", "category": "work_study", "text": "Studies ECE"}],
    "key_dates": [{"person_ref": "new:P1", "label": "birthday", "date": "02-14"}],
}


class ScriptedLLM:
    def __init__(self) -> None:
        self.reply: dict[str, Any] | Exception = DEBRIEF

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(stop_reason="tool_use", tool_calls=[
            ToolCall(id="t1", name="record_proposals", args=self.reply)])


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
async def client(settings: CoreSettings, db: Database,
                 llm: ScriptedLLM) -> AsyncIterator[httpx.AsyncClient]:
    repo = PeopleRepo(db)
    deps = GatewayDeps(
        settings=settings, db=db, traces=TracesRepo(db),
        started_monotonic=time.monotonic(), people_repo=repo,
        people=PeopleService(repo, llm, ZoneInfo("Asia/Kolkata")),  # type: ignore[arg-type]
    )
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_page_is_open_but_api_is_not(client: httpx.AsyncClient) -> None:
    page = await client.get("/app/people")
    assert page.status_code == 200 and "Waiting for review" in page.text
    assert (await client.get("/people/pending")).status_code == 401


async def test_debrief_review_round_trip(client: httpx.AsyncClient) -> None:
    sent = await client.post("/people/debrief", headers=AUTH,
                             json={"text": "Had coffee with Aarav today"})
    assert sent.status_code == 200 and sent.json()["staged"] == 3

    [debrief] = (await client.get("/people/pending", headers=AUTH)).json()["debriefs"]
    assert debrief["new_people"] == {"P1": {"name": "Aarav Mehta", "approved": False}}
    by_kind = {p["kind"]: p for p in debrief["proposals"]}

    # A browser edit is validated like a model proposal: 02-30 is refused.
    bad = {**by_kind["key_date"]["payload"], "date": "02-30"}
    refused = await client.post(f"/people/proposals/{by_kind['key_date']['id']}/approve",
                                headers=AUTH, json={"payload": bad})
    assert refused.status_code == 422 and "date" in refused.json()["detail"]

    fixed = {**by_kind["fact"]["payload"], "text": "Studies CSE now"}
    ok = await client.post(f"/people/proposals/{by_kind['fact']['id']}/approve",
                           headers=AUTH, json={"payload": fixed})
    assert ok.status_code == 200

    again = await client.post(f"/people/proposals/{by_kind['fact']['id']}/approve",
                              headers=AUTH, json={})
    assert again.status_code == 409

    [debrief] = (await client.get("/people/pending", headers=AUTH)).json()["debriefs"]
    assert debrief["new_people"]["P1"]["approved"] is True    # created by the fact
    people = (await client.get("/people/pending", headers=AUTH)).json()["people"]
    assert [p["name"] for p in people] == ["Aarav Mehta"]

    gone = await client.post(f"/people/proposals/{by_kind['key_date']['id']}/reject",
                             headers=AUTH, json={})
    assert gone.status_code == 200
    assert (await client.get("/people/pending", headers=AUTH)).json()["debriefs"] == []


async def test_failures_are_explained(client: httpx.AsyncClient, llm: ScriptedLLM) -> None:
    short = await client.post("/people/debrief", headers=AUTH, json={"text": "hi"})
    assert short.status_code == 422

    llm.reply = NoApiKeyError("no key configured")
    no_key = await client.post("/people/debrief", headers=AUTH,
                               json={"text": "Met someone new today at the lab"})
    assert no_key.status_code == 503
