# =============================================================================
# src/jarvis/llm/layer.py - the one doorway to model intelligence
# =============================================================================
#
# Every model call in JARVIS goes through LLMLayer.complete(). No caller
# ever touches an adapter or a model name. The doorway's three duties, on
# every call without exception:
#
#   1. resolve the requested TIER to a concrete model (config decides)
#   2. compute the call's COST from token usage
#   3. report a trace record to the TRACE SINK - success or failure alike
#
# The trace sink is deliberately just "an async function that accepts a
# record". Today's default writes a log line; the observability layer will
# hand in one that writes a database row instead. This file will not
# change when that happens - that is the point of the design.
#
# A call that fails is STILL traced, with its error recorded and zero
# tokens. Invisible failures are how systems rot.
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings
from jarvis.llm.anthropic import (
    AnthropicAdapter,
    ProviderError,
    ProviderResult,
    TextCallback,
)
from jarvis.llm.pricing import compute_cost
from jarvis.llm.tiers import Tier, resolve_model

log = get_logger("llm.layer")


class NoApiKeyError(RuntimeError):
    """Model calls need a key; the daemon does not. This error surfaces at
    the moment of the call, never at boot."""


# -- normalised shapes the rest of JARVIS uses --------------------------------

class ToolSpec(BaseModel):
    """A tool offered to the model: name, human description, and a JSON
    schema for its arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_provider(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolCall(BaseModel):
    """The model asked for a tool run."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """What a caller gets back from complete()."""

    model_config = ConfigDict(extra="forbid")

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    model: str
    tokens_in: int
    tokens_out: int
    cached_tokens: int
    call_id: str
    cost_usd: float
    cost_inr: float
    latency_ms: int

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0


# -- trace sink ---------------------------------------------------------------

@dataclass(frozen=True)
class LLMCallRecord:
    """One row of accounting truth about one model call - the shape the
    llm_calls table will store."""

    id: str
    ts: str                  # UTC ISO-8601
    trace_id: str
    actor: str               # who called: core.orchestrator, subagent.researcher, ...
    tier: str
    model: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cached_tokens: int
    cost_usd: float
    cost_inr: float
    stop_reason: str
    error: str | None


TraceSink = Callable[[LLMCallRecord], Awaitable[None]]


async def logging_trace_sink(record: LLMCallRecord) -> None:
    """Default sink until the database writer exists: one structured log
    line per call, same fields the table will hold."""
    log.info("llm call", extra={
        "trace_id": record.trace_id,
        "actor": record.actor,
        "tier": record.tier,
        "model": record.model,
        "latency_ms": record.latency_ms,
        "tokens_in": record.tokens_in,
        "tokens_out": record.tokens_out,
        "cached_tokens": record.cached_tokens,
        "cost_usd": record.cost_usd,
        "error": record.error,
    })


# -- the doorway --------------------------------------------------------------

class LLMLayer:
    """The single entry point for model intelligence. One instance on the
    Core, handed to whoever needs to think."""

    def __init__(
        self,
        settings: CoreSettings,
        trace_sink: TraceSink = logging_trace_sink,
    ) -> None:
        self._settings = settings
        self._trace = trace_sink
        self._adapter: AnthropicAdapter | None = None
        if settings.anthropic_api_key is not None:
            key = settings.anthropic_api_key.get_secret_value().strip()
            if key:
                self._adapter = AnthropicAdapter(api_key=key)

    @property
    def available(self) -> bool:
        """Whether model calls can be made at all in this deployment."""
        return self._adapter is not None

    async def complete(
        self,
        tier: Tier,
        system: str,
        messages: list[dict[str, Any]],
        *,
        actor: str,
        trace_id: str,
        tools: list[ToolSpec] | None = None,
        provider_tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        """One model call: resolve tier, call provider, price it, trace it,
        return the normalised response. actor and trace_id are mandatory -
        an unattributable call is not allowed to exist."""
        if self._adapter is None:
            raise NoApiKeyError(
                "no JARVIS_ANTHROPIC_API_KEY configured - the daemon runs, "
                "but model calls cannot be made until a key is set"
            )

        model = resolve_model(tier, self._settings)
        call_id = new_ulid() 
        started = time.monotonic()

        try:
            result = await self._adapter.complete(
                model=model,
                system=system,
                messages=messages,
                tools=[t.to_provider() for t in tools] if tools else None,
                provider_tools=provider_tools,
                max_tokens=max_tokens,
                on_text=on_text,
            )
        except ProviderError as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            await self._trace(LLMCallRecord(
                id=call_id,
                ts=utc_now().isoformat(),
                trace_id=trace_id,
                actor=actor,
                tier=tier.value,
                model=model,
                latency_ms=latency_ms,
                tokens_in=0,
                tokens_out=0,
                cached_tokens=0,
                cost_usd=0.0,
                cost_inr=0.0,
                stop_reason="error",
                error=str(exc),
            ))
            raise

        latency_ms = int((time.monotonic() - started) * 1000)
        cost = compute_cost(
            model=result.model,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cached_tokens=result.cached_tokens,
        )
        await self._trace(LLMCallRecord(
            id=call_id,
            ts=utc_now().isoformat(),
            trace_id=trace_id,
            actor=actor,
            tier=tier.value,
            model=result.model,
            latency_ms=latency_ms,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cached_tokens=result.cached_tokens,
            cost_usd=cost.rounded_usd,
            cost_inr=cost.rounded_inr,
            stop_reason=result.stop_reason,
            error=None,
        ))

        return LLMResponse(
            text=result.text,
            tool_calls=[
                ToolCall(id=c.id, name=c.name, args=c.args)
                for c in result.tool_calls
            ],
            stop_reason=result.stop_reason,
            model=result.model,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cached_tokens=result.cached_tokens,
            call_id=call_id,
            cost_usd=cost.rounded_usd,
            cost_inr=cost.rounded_inr,
            latency_ms=latency_ms,
        )


# -- message construction helpers ---------------------------------------------
# The orchestrator builds conversations with these instead of hand-writing
# provider dictionaries. If the wire shape ever changes, it changes here.

def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant_message(response: LLMResponse) -> dict[str, Any]:
    """Re-encode a model response (text and any tool requests) as a
    conversation message, so the next call sees what the model said."""
    content: list[dict[str, Any]] = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for call in response.tool_calls:
        content.append({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.args,
        })
    return {"role": "assistant", "content": content}


def tool_result_message(call_id: str, result_text: str, is_error: bool = False) -> dict[str, Any]:
    """The answer to one tool call, matched back by the provider's id."""
    return {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": result_text,
            "is_error": is_error,
        }],
    }