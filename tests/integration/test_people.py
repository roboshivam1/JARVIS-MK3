# =============================================================================
# tests/integration/test_people.py - the people database
# =============================================================================
#
# The promises that matter:
#   - the model can only ever STAGE; nothing is real until approved
#   - handles (R1, F1) are translated to real ids, dangling ones dropped
#   - approving something about a new person creates that person
#   - rejecting a new person takes everything about them with it
#   - changed facts supersede, never overwrite
#   - sensitive facts never reach the prompt or the conversation context
# Real database, fake model.
# =============================================================================

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from jarvis.core.db.database import Database
from jarvis.core.db.repos.people import PeopleRepo, ReviewError
from jarvis.core.people.service import PeopleService
from jarvis.llm.layer import ToolCall


class FakeLLM:
    """Answers each call with the next queued tool arguments and keeps
    the prompt it was sent, so tests can check what left the machine."""

    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []
        self.prompts: list[str] = []

    async def complete(self, tier: Any, system: str, messages: list[dict[str, Any]],
                       **kwargs: Any) -> Any:
        assert kwargs["tool_choice"] == {"type": "tool", "name": "record_proposals"}
        self.prompts.append(messages[0]["content"])
        args = self.replies.pop(0)
        return SimpleNamespace(stop_reason="tool_use",
                               tool_calls=[ToolCall(id="t1", name="record_proposals", args=args)])


@pytest.fixture
def repo(db: Database) -> PeopleRepo:
    return PeopleRepo(db)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(repo: PeopleRepo, llm: FakeLLM) -> PeopleService:
    return PeopleService(repo, llm, ZoneInfo("Asia/Kolkata"))  # type: ignore[arg-type]


FIRST_DEBRIEF = {
    "new_people": [
        {"tag": "P1", "name": "Aarav Mehta", "relationship": "friend", "closeness": 1,
         "summary": "ECE friend, embedded systems"},
        {"tag": "P2", "name": "Kabir Shah", "relationship": "friend"},
    ],
    "aliases": [{"person_ref": "new:P1", "alias": "AM"}],
    "facts": [
        {"person_ref": "new:P1", "category": "work_study", "text": "Studies ECE at LNMIIT"},
        {"person_ref": "new:P1", "category": "friction", "sensitivity": "sensitive",
         "text": "Touchy about the club election"},
        {"person_ref": "new:P2", "category": "preferences", "text": "Vegetarian"},
        {"person_ref": "new:P9", "category": "trait", "text": "Tag that does not exist"},
        {"person_ref": "banana", "category": "trait", "text": "Malformed ref"},
    ],
    "commitments": [{"person_ref": "new:P1", "direction": "they_owe",
                     "what": "Motor controller schematics", "due_date": "2026-09-30"}],
    "key_dates": [{"person_ref": "new:P1", "label": "birthday", "date": "02-14"},
                  {"person_ref": "new:P1", "label": "impossible", "date": "02-30"}],
}


def _find(pending: list[dict[str, Any]], kind: str, **match: Any) -> dict[str, Any]:
    return next(p for p in pending if p["kind"] == kind
                and all(p["payload"].get(k) == v for k, v in match.items()))


async def _staged(repo: PeopleRepo) -> list[dict[str, Any]]:
    debriefs = await repo.pending_debriefs()
    return debriefs[0]["proposals"] if debriefs else []


async def test_debrief_stages_and_drops_bad_items(service: PeopleService, repo: PeopleRepo,
                                                  llm: FakeLLM) -> None:
    llm.replies.append(FIRST_DEBRIEF)
    outcome = await service.debrief("coffee with Aarav and Kabir", trace_id="T")
    assert outcome.staged == 8
    assert outcome.dropped == 3          # malformed ref, bad date, dangling tag
    assert await repo.roster() == []     # nothing real until approved

async def test_review_flow(service: PeopleService, repo: PeopleRepo, llm: FakeLLM) -> None:
    llm.replies.append(FIRST_DEBRIEF)
    await service.debrief("coffee with Aarav and Kabir", trace_id="T")
    pending = await _staged(repo)

    # Approving a fact about a new person creates the person.
    await repo.approve(_find(pending, "fact", text="Studies ECE at LNMIIT")["id"])
    [aarav] = await repo.find("aarav mehta")

    # Edits are validated and stored as approved.
    friction = _find(pending, "fact", category="friction")
    await repo.approve(friction["id"], edited={**friction["payload"],
                                               "text": "Sore about the club election"})
    for kind in ("alias", "commitment", "key_date"):
        await repo.approve(_find(pending, kind)["id"])

    # Rejecting a new person takes their fact with them.
    rejected = await repo.reject(_find(pending, "new_person", tag="P2")["id"])
    assert len(rejected) == 2
    assert await repo.pending_debriefs() == []        # debrief closed itself

    assert (await repo.find("am"))[0]["id"] == aarav["id"]
    profile = await repo.profile(aarav["id"])
    assert profile is not None
    assert {f["text"] for f in profile["facts"]} == {
        "Studies ECE at LNMIIT", "Sore about the club election"}
    assert profile["commitments"][0]["what"] == "Motor controller schematics"


async def test_handles_supersession_and_sensitivity(service: PeopleService, repo: PeopleRepo,
                                                    llm: FakeLLM) -> None:
    llm.replies.append(FIRST_DEBRIEF)
    await service.debrief("coffee with Aarav and Kabir", trace_id="T")
    for p in await _staged(repo):
        about_kabir = "P2" in (p["payload"].get("tag"), p["payload"].get("person_ref", "")[4:])
        if not about_kabir:
            await repo.approve(p["id"])
    [aarav] = await repo.find("Aarav Mehta")

    # Second debrief: the model speaks in handles, never ULIDs.
    llm.replies.append({"facts": [{"person_ref": "existing:R1", "category": "work_study",
                                   "text": "Switched to CSE", "supersedes_fact_id": "F1"}]})
    await service.debrief("Aarav switched to CSE", trace_id="T")
    prompt = llm.prompts[-1]
    assert "R1 | Aarav Mehta | AM" in prompt
    assert "F1 | R1 Aarav Mehta | Studies ECE at LNMIIT" in prompt
    assert "election" not in prompt.lower()            # sensitive never sent

    [proposal] = await _staged(repo)
    assert proposal["payload"]["person_ref"] == f"existing:{aarav['id']}"
    await repo.approve(proposal["id"])

    profile = await repo.profile(aarav["id"])
    assert profile is not None
    texts = {f["text"] for f in profile["facts"]}
    assert "Switched to CSE" in texts and "Studies ECE at LNMIIT" not in texts

    context = await service.mention_context("did Aarav reply?")
    assert "Switched to CSE" in context and "election" not in context.lower()
    assert "[sensitive]" in await service.describe(aarav["id"])


async def test_guard_rails(service: PeopleService, repo: PeopleRepo, llm: FakeLLM) -> None:
    llm.replies.append(FIRST_DEBRIEF)
    await service.debrief("coffee", trace_id="T")
    pending = await _staged(repo)
    person = _find(pending, "new_person", tag="P1")

    with pytest.raises(ReviewError):
        await repo.approve("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    with pytest.raises(ReviewError):
        await repo.approve(person["id"], edited={**person["payload"], "tag": "P7"})
    await repo.approve(person["id"])
    with pytest.raises(ReviewError):
        await repo.approve(person["id"])               # already decided

    # A failed approval writes nothing at all.
    fact = _find(pending, "fact", text="Studies ECE at LNMIIT")
    with pytest.raises(ReviewError):
        await repo.approve(fact["id"], edited={**fact["payload"],
                                               "supersedes_fact_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"})
    assert fact["id"] in {p["id"] for p in await _staged(repo)}
