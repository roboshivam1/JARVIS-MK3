# =============================================================================
# tests/integration/test_people_jarvis.py - JARVIS using the people database
# =============================================================================
#
# Through the real paths, not around them: the booted app, the orchestrator's
# own toolset with its Guard (an allowlist gap would show up here as a
# "denied"), the real SessionManager building a turn's context, and the
# sleep cycle's filing of person facts.
# =============================================================================

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import pytest

from jarvis.agentloop.loop import LoopResult
from jarvis.common.facts import FactCategory
from jarvis.common.people import ExtractionResult
from jarvis.common.settings import CoreSettings
from jarvis.core.app import CoreApp
from jarvis.core.sessionmgr import SessionManager
from jarvis.jobs.maintenance import _file_candidates
from jarvis.subagents.archivist import ExtractedFact


def _settings(tmp_path: Path) -> CoreSettings:
    return CoreSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        gateway_token="test-token",  # type: ignore[arg-type]
    )


@pytest.fixture
async def app(tmp_path: Path) -> AsyncIterator[CoreApp]:
    core = CoreApp(_settings(tmp_path))
    await core.boot()
    try:
        yield core
    finally:
        await core.shutdown()


async def _seed(core: CoreApp) -> str:
    """One person, approved through the real review path."""
    assert core.people_repo is not None
    soon = (datetime.now(ZoneInfo("Asia/Kolkata")).date() + timedelta(days=5)).strftime("%m-%d")
    result = ExtractionResult.model_validate({
        "new_people": [{"tag": "P1", "name": "Aarav Mehta", "relationship": "friend"}],
        "aliases": [{"person_ref": "new:P1", "alias": "AM"}],
        "facts": [
            {"person_ref": "new:P1", "category": "work_study", "text": "Studies CSE"},
            {"person_ref": "new:P1", "category": "friction", "sensitivity": "sensitive",
             "text": "Sore about the club election"},
        ],
        "commitments": [{"person_ref": "new:P1", "direction": "they_owe",
                         "what": "Motor controller schematics"}],
        "key_dates": [{"person_ref": "new:P1", "label": "birthday", "date": soon}],
    })
    debrief_id = await core.people_repo.stage("seed", result, "web")
    for p in (await core.people_repo.pending_debriefs())[0]["proposals"]:
        await core.people_repo.approve(p["id"])
    return debrief_id


async def test_tools_are_registered_and_allowed(app: CoreApp) -> None:
    await _seed(app)
    assert app.session_mgr is not None
    tools = app.session_mgr._orchestrator._build_toolset("S", "T")

    text, is_error = await tools.execute("people_lookup", {"name": "am"})
    assert not is_error, text
    assert "Aarav Mehta" in text and "Sore about the club election [sensitive]" in text

    text, is_error = await tools.execute("people_upcoming", {"days": 10})
    assert not is_error, text
    assert "Owed to the owner:" in text and "Motor controller schematics" in text
    assert "Aarav Mehta, birthday" in text

    text, _ = await tools.execute("people_lookup", {"name": "Nobody Here"})
    assert "No one called" in text

class CapturingOrchestrator:
    def __init__(self) -> None:
        self.retrieved: list[str] = []

    async def respond(self, messages: list[dict[str, Any]], *, session_id: str,
                      rolling_summary: str, trace_id: str, profile_doc: str = "",
                      retrieved_memory: str = "", on_text: Any = None) -> LoopResult:
        self.retrieved.append(retrieved_memory)
        return LoopResult(text="ok", llm_call_ids=[], iterations=1)


async def test_mentioned_people_reach_the_turn_without_sensitive_facts(app: CoreApp) -> None:
    await _seed(app)
    assert app.sessions is not None and app.events is not None
    fake = CapturingOrchestrator()
    mgr = SessionManager(app.sessions, app.events, fake,  # type: ignore[arg-type]
                         people=app.people)

    await mgr.handle_user_message("web", "did Aarav send the schematics yet?")
    context = fake.retrieved[-1]
    assert "People mentioned:" in context and "Studies CSE" in context
    assert "Motor controller schematics" in context
    assert "election" not in context.lower()

    await mgr.handle_user_message("web", "what's the weather like")
    assert "People mentioned" not in fake.retrieved[-1]


class FakeMemory:
    def __init__(self) -> None:
        self.stored: list[str] = []

    async def store(self, *, text: str, **_: Any) -> None:
        self.stored.append(text)


class FakePeople:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.debriefs: list[str] = []

    async def debrief(self, text: str, *, trace_id: str, source: str) -> None:
        assert source == "archivist"
        if self.fail:
            raise RuntimeError("model unavailable")
        self.debriefs.append(text)


CANDIDATES = [
    ExtractedFact(text="Shivam's friend Kabir got an internship", category=FactCategory.PERSON,
                  importance=0.5),
    ExtractedFact(text="Shivam prefers SQLite", category=FactCategory.PREFERENCE, importance=0.6),
]


async def test_sleep_cycle_routes_person_facts_to_review() -> None:
    memory, people = FakeMemory(), FakePeople()
    await _file_candidates(CANDIDATES, memory, people, [], "T")  # type: ignore[arg-type]
    assert people.debriefs == ["- Shivam's friend Kabir got an internship"]
    assert memory.stored == ["Shivam prefers SQLite"]


async def test_sleep_cycle_never_drops_facts() -> None:
    memory, people = FakeMemory(), FakePeople(fail=True)
    await _file_candidates(CANDIDATES, memory, people, [], "T")  # type: ignore[arg-type]
    assert sorted(memory.stored) == sorted(c.text for c in CANDIDATES)

    memory = FakeMemory()
    await _file_candidates(CANDIDATES, memory, None, [], "T")  # type: ignore[arg-type]
    assert len(memory.stored) == 2
