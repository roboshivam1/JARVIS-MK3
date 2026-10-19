# =============================================================================
# tests/integration/test_reels_jarvis.py - JARVIS knowing his own reels
# =============================================================================
#
# Through the booted app: the orchestrator's real toolset with its Guard
# (a tool missing from the allowlist would come back "denied"), and the
# real SessionManager building a turn's context.
#
# The promise: after a reel goes out, "what should the caption be?" reaches
# a JARVIS who can see which reel, its script, and that it is not posted.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from jarvis.agentloop.loop import LoopResult
from jarvis.common.reels import EpisodeStatus, EpisodeType, ReelBrief, ReelEpisode
from jarvis.common.settings import CoreSettings
from jarvis.core.app import CoreApp
from jarvis.core.db.repos.reels import ReelsRepo
from jarvis.core.orchestrator.prompts import assemble_system_prompt
from jarvis.core.sessionmgr import SessionManager

from tests.integration.test_reels_service import BRIEF, FakeLLM


@pytest.fixture
async def app(tmp_path: Path) -> AsyncIterator[CoreApp]:
    core = CoreApp(CoreSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        gateway_token="test-token",  # type: ignore[arg-type]
    ))
    await core.boot()
    try:
        yield core
    finally:
        await core.shutdown()


async def _delivered(core: CoreApp) -> ReelEpisode:
    assert core.db is not None
    episode = ReelEpisode(
        kit_id="meet-the-staff-abc123",
        type=EpisodeType.MEET_THE_STAFF,
        title="The Butler and His Basement",
        request="introducing JARVIS",
        brief=ReelBrief.model_validate({**BRIEF, "episode_type": "meet_the_staff"}),
        lines=["Good evening. I am JARVIS, and I live in the basement.",
               "One human. Forty pages of notes."],
        status=EpisodeStatus.DELIVERED,
        verification="1080x1920 47.30 s 13 shots",
    )
    await ReelsRepo(core.db).create(episode)
    return episode


class CapturingOrchestrator:
    def __init__(self) -> None:
        self.retrieved: list[str] = []

    async def respond(self, messages: list[dict[str, Any]], *, session_id: str,
                      rolling_summary: str, trace_id: str, profile_doc: str = "",
                      retrieved_memory: str = "", on_text: Any = None) -> LoopResult:
        self.retrieved.append(retrieved_memory)
        return LoopResult(text="ok", llm_call_ids=[], iterations=1)


async def test_every_turn_knows_the_latest_reel(app: CoreApp) -> None:
    assert app.sessions is not None and app.events is not None
    fake = CapturingOrchestrator()
    mgr = SessionManager(app.sessions, app.events, fake,  # type: ignore[arg-type]
                         reels=app.reels)

    await mgr.handle_user_message("telegram", "hello")
    assert "Your reels" not in fake.retrieved[-1]      # nothing to know yet

    episode = await _delivered(app)
    await mgr.handle_user_message("telegram", "Lovely reel. What should the caption be?")
    context = fake.retrieved[-1]
    assert "Your reels, newest first:" in context
    assert f"ref {episode.ref}" in context and "The Butler and His Basement" in context
    assert "One human. Forty pages of notes." in context
    assert "not yet posted" in context


async def test_reel_tools_are_registered_and_allowed(app: CoreApp) -> None:
    episode = await _delivered(app)
    assert app.session_mgr is not None and app.reels is not None
    tools = app.session_mgr._orchestrator._build_toolset("S", "T")

    text, is_error = await tools.execute("reels_recent", {})
    assert not is_error, text
    assert episode.ref in text

    text, is_error = await tools.execute("reel_details", {"ref": "last"})
    assert not is_error, text
    assert "Asked for: introducing JARVIS" in text and "Forty pages" in text

    text, is_error = await tools.execute("reel_mark_posted", {"ref": episode.ref})
    assert not is_error, text
    assert "is live" in text

    app.reels._llm = FakeLLM()  # type: ignore[assignment]
    app.reels._llm.replies.append(BRIEF)  # type: ignore[attr-defined]
    text, is_error = await tools.execute("reel_request", {"idea": "five more minutes"})
    assert not is_error, text
    assert text.startswith("Briefed reel") and "not connected" in text

    text, _ = await tools.execute("reel_request", {"idea": "another one, please"})
    assert text.startswith("Refused: one reel at a time")


def test_the_persona_knows_about_the_channel() -> None:
    prompt = assemble_system_prompt()
    assert "Your channel:" in prompt and "You never post anything" in prompt
    assert "Basement Supply" in prompt
