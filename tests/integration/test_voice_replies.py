# =============================================================================
# tests/integration/test_voice_replies.py - spoken replies over the client socket
# =============================================================================
#
# The promise to voice clients (the web client, the arc reactor): every
# sentence of a reply arrives as core.audio BEFORE core.audio_done, so a
# client can treat audio_done as "that was the last one". It used to be
# sent while sentences were still being synthesised - for a reply ending
# on a full stop, before any audio at all.
# =============================================================================

from __future__ import annotations

import asyncio
import base64
import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from jarvis.agentloop.loop import LoopResult
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.gateway.clients import ClientRegistry
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.sessionmgr import SessionManager

websockets = pytest.importorskip("websockets")   # in the worker extra

REPLY = "Rajma tonight, sir. The paneer ran out early. Anything else?"


class Ears:
    available = True

    async def transcribe(self, audio: bytes, mime: str = "audio/webm",
                         language: str | None = None) -> str:
        return "what is for dinner"


class Brain:
    async def respond(self, messages: list[dict[str, Any]], on_text: Any = None,
                      **_: Any) -> LoopResult:
        for word in REPLY.split(" "):
            await on_text(word + " ")
        return LoopResult(text=REPLY, llm_call_ids=[], iterations=1)


class SlowVoice:
    """Longer sentences take longer, so they finish out of order."""
    available = True

    async def synthesise(self, text: str) -> bytes:
        await asyncio.sleep(0.02 * len(text))
        return b"RIFF" + text.encode()


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_audio_done_comes_after_every_sentence(db: Database, tmp_path: Path) -> None:
    settings = CoreSettings(_env_file=None, data_dir=tmp_path, gateway_token="tok")
    sessions = SessionsRepo(db)
    deps = GatewayDeps(settings=settings, db=db, traces=TracesRepo(db),
                       started_monotonic=time.monotonic(), clients=ClientRegistry(),
                       session_mgr=SessionManager(sessions, EventsRepo(db), Brain()),  # type: ignore[arg-type]
                       sessions=sessions, transcriber=Ears(), speaker=SlowVoice())  # type: ignore[arg-type]
    port = _port()
    server = uvicorn.Server(uvicorn.Config(create_app(deps), host="127.0.0.1", port=port,
                                           log_level="warning"))
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws/client") as ws:
            await ws.send(json.dumps({"kind": "client.hello", "payload": {
                "token": "tok", "client_kind": "voice.rpi", "label": "test"}}))
            assert json.loads(await ws.recv())["kind"] == "core.ready"
            await ws.send(json.dumps({"kind": "client.audio_chunk", "payload": {
                "seq": 0, "data_b64": base64.b64encode(b"RIFF....").decode(),
                "mime": "audio/wav"}}))
            await ws.send(json.dumps({"kind": "client.audio_end", "payload": {}}))
            kinds: list[str] = []
            sentences: list[tuple[int, bytes]] = []
            while "core.assistant_done" not in kinds:
                frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                kinds.append(frame["kind"])
                if frame["kind"] == "core.audio":
                    sentences.append((frame["payload"]["seq"],
                                      base64.b64decode(frame["payload"]["data_b64"])))
    finally:
        server.should_exit = True
        await serving

    assert kinds.index("core.audio_done") > max(i for i, k in enumerate(kinds) if k == "core.audio")
    assert [seq for seq, _ in sentences] != [0, 1, 2]          # they did race
    assert [text for _, text in sorted(sentences)] == [
        b"RIFFRajma tonight, sir.", b"RIFFThe paneer ran out early.", b"RIFFAnything else?"]
    assert kinds[-2:] == ["core.audio_done", "core.assistant_done"]


@pytest.fixture(autouse=True)
def _quiet_uvicorn() -> None:
    import logging
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
