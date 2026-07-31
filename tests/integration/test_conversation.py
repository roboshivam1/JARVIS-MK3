# =============================================================================
# tests/integration/test_conversation.py - phase 1 machinery, tested
# without any network or API key
# =============================================================================
#
# The expensive unreliable edge (the model) is replaced by a scripted fake
# orchestrator, so these tests exercise OUR machinery deterministically:
#
#   - gateway auth and status, in-process via httpx (no port binding)
#   - turn persistence and ordering across a restart
#   - interruption: a second message cancels the first mid-generation
# =============================================================================

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from jarvis.agentloop.loop import LoopResult
from jarvis.common.settings import CoreSettings
from jarvis.common.sessions import TurnRole
from jarvis.core.app import CoreApp
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.sessionmgr import SessionManager
from jarvis.llm.layer import TextCallback



def _settings(tmp_path: Path) -> CoreSettings:
    # _env_file=None: no real .env leaks into tests (see test_boot.py).
    return CoreSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        gateway_token="test-token",  # type: ignore[arg-type]  # coerced to SecretStr
    )

class FakeOrchestrator:
    """Scripted stand-in for the real mind: instant, free, deterministic.
    reply_delay_s simulates thinking time so interruption can be tested."""

    def __init__(self, reply_delay_s: float = 0.0) -> None:
        self.reply_delay_s = reply_delay_s
        self.calls: list[list[dict[str, Any]]] = []

    async def respond(
        self,
        messages: list[dict[str, Any]],
        *,
        rolling_summary: str,
        trace_id: str,
        on_text: TextCallback | None = None,
    ) -> LoopResult:
        self.calls.append(messages)
        if self.reply_delay_s:
            await asyncio.sleep(self.reply_delay_s)
        text = f"reply to: {messages[-1]['content']}"
        if on_text is not None:
            await on_text(text)
        return LoopResult(text=text, llm_call_ids=[], iterations=1)


async def _make_mgr(app: CoreApp, delay: float = 0.0) -> tuple[SessionManager, FakeOrchestrator]:
    assert app.sessions is not None and app.events is not None
    fake = FakeOrchestrator(reply_delay_s=delay)
    return SessionManager(app.sessions, app.events, fake), fake  # type: ignore[arg-type]


# -- gateway ------------------------------------------------------------------

async def test_gateway_auth_and_status(tmp_path: Path) -> None:
    app = CoreApp(_settings(tmp_path))
    await app.boot()
    try:
        assert app.gateway_deps is not None
        api = create_app(app.gateway_deps)
        transport = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # health: open
            r = await client.get("/health")
            assert r.status_code == 200 and r.json()["status"] == "alive"

            # status: locked without / with wrong / with right token
            assert (await client.get("/status")).status_code == 401
            r = await client.get(
                "/status", headers={"Authorization": "Bearer wrong"}
            )
            assert r.status_code == 401
            r = await client.get(
                "/status", headers={"Authorization": "Bearer test-token"}
            )
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["sessions"] == 0 and body["llm_calls"] == 0
    finally:
        await app.shutdown()


# -- conversation persistence -------------------------------------------------

async def test_turns_persist_across_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    app1 = CoreApp(settings)
    await app1.boot()
    mgr, _ = await _make_mgr(app1)
    result = await mgr.handle_user_message("test", "hello there")
    assert result.reply is not None
    assert result.reply.content == "reply to: hello there"
    await app1.shutdown()

    # Second life: same database, history intact, context replayed.
    app2 = CoreApp(settings)
    await app2.boot()
    try:
        mgr2, fake2 = await _make_mgr(app2)
        result2 = await mgr2.handle_user_message("test", "still there?")
        assert result2.reply is not None
        # The fake saw the OLD turns as context - proof memory is the db.
        replayed = [m["content"] for m in fake2.calls[0]]
        assert any("hello there" in c for c in replayed)

        assert app2.sessions is not None
        turns = await app2.sessions.recent_turns(result2.session_id)
        roles = [t.role for t in turns]
        assert roles == [
            TurnRole.USER, TurnRole.ASSISTANT,
            TurnRole.USER, TurnRole.ASSISTANT,
        ]
    finally:
        await app2.shutdown()


# -- interruption -------------------------------------------------------------

async def test_new_message_interrupts_inflight_turn(tmp_path: Path) -> None:
    app = CoreApp(_settings(tmp_path))
    await app.boot()
    try:
        mgr, _ = await _make_mgr(app, delay=0.5)   # slow thinker

        first = asyncio.create_task(
            mgr.handle_user_message("test", "long question")
        )
        await asyncio.sleep(0.1)                   # let it start thinking
        second = await mgr.handle_user_message("test", "actually, this instead")
        first_result = await first

        assert first_result.interrupted and first_result.reply is None
        assert not second.interrupted and second.reply is not None

        # History: both user turns stored, exactly one assistant reply.
        assert app.sessions is not None
        turns = await app.sessions.recent_turns(second.session_id)
        assert [t.role for t in turns] == [
            TurnRole.USER, TurnRole.USER, TurnRole.ASSISTANT,
        ]

        # And the interruption is on the record.
        assert app.events is not None
        interrupts = await app.events.recent(kind="session.interrupted")
        assert len(interrupts) == 1
    finally:
        await app.shutdown()