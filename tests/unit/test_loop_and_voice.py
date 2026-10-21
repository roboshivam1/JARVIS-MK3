# =============================================================================
# tests/unit/test_loop_and_voice.py - what JARVIS says, and what he hears
# =============================================================================
#
# Two bugs from the first night on the VPS:
#   - asked how it felt to move, JARVIS answered, stored a fact, closed
#     with "Noted, sir" - and the closing line replaced the answer
#   - voice notes in English came back labelled Arabic or Gujarati, and
#     he replied in kind
# =============================================================================

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from jarvis.agentloop.loop import run_agent_loop
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.common.settings import CoreSettings
from jarvis.core.orchestrator.prompts import assemble_system_prompt
from jarvis.llm.layer import LLMResponse, ToolCall
from jarvis.llm.tiers import Tier
from jarvis.llm.transcription import Transcriber


class _Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact: str


class ScriptedLLM:
    """Answers, stores a fact, then closes - streaming as it goes."""

    def __init__(self) -> None:
        self.turns = [
            ("It feels roomier, sir. Europe is quiet at this hour.",
             [ToolCall(id="t1", name="memory_store", args={"fact": "JARVIS runs on a VPS"})]),
            ("Noted and remembered, sir.", []),
        ]

    async def complete(self, tier: Any, system: str, messages: list[Any], *,
                       on_text: Any = None, **_: Any) -> LLMResponse:
        text, calls = self.turns.pop(0)
        if on_text is not None:
            for word in text.split(" "):
                await on_text(word + " ")
        return LLMResponse(
            text=text, tool_calls=calls, stop_reason="tool_use" if calls else "end_turn",
            model="m", tokens_in=1, tokens_out=1, cached_tokens=0, call_id="c",
            cost_usd=0, cost_inr=0, latency_ms=1,
        )


async def test_the_answer_survives_a_tool_call() -> None:
    stored: list[str] = []

    async def remember(args: _Fact) -> str:
        stored.append(args.fact)
        return "stored"

    tools = Toolset()
    tools.register(InlineTool(name="memory_store", description="remember",
                              args_model=_Fact, handler=remember))
    streamed: list[str] = []

    async def on_text(chunk: str) -> None:
        streamed.append(chunk)

    result = await run_agent_loop(
        ScriptedLLM(), Tier.REASONER, "system",  # type: ignore[arg-type]
        [{"role": "user", "content": "How does the move feel?"}],
        tools, actor="test", trace_id="t", on_text=on_text,
    )
    assert stored == ["JARVIS runs on a VPS"]
    assert result.text == ("It feels roomier, sir. Europe is quiet at this hour."
                           "\n\nNoted and remembered, sir.")
    live = "".join(streamed)
    assert "hour. \n\nNoted" in live                 # a break between the two, live too


async def test_voice_is_transcribed_as_english(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    async def post(self: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
        sent.update(kwargs["files"])
        return httpx.Response(200, json={"text": "aap kaise ho"},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    settings = CoreSettings(_env_file=None, groq_api_key="k")  # type: ignore[arg-type]
    assert await Transcriber(settings).transcribe(b"audio") == "aap kaise ho"
    assert sent["language"] == (None, "en")
    assert "Aap kaise ho?" in sent["prompt"][1]

    sent.clear()
    auto = CoreSettings(_env_file=None, groq_api_key="k",  # type: ignore[arg-type]
                        transcribe_language="")
    await Transcriber(auto).transcribe(b"audio")
    assert "language" not in sent


def test_the_persona_sets_the_reply_language() -> None:
    prompt = assemble_system_prompt()
    assert "Reply in English" in prompt and "Hinglish" in prompt
    assert "Your channel:" in prompt                  # nothing earlier was lost
