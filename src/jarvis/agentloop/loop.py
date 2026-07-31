# =============================================================================
# src/jarvis/agentloop/loop.py - THE agentic loop
# =============================================================================
#
# The engine shared by the orchestrator (JARVIS conversing) and, from the
# next phase, every subagent. One cycle:
#
#   1. call the model with the conversation so far + the tool contracts
#   2. model returns text and/or tool call requests
#   3. no tool calls -> done, return the text
#   4. otherwise run each tool, append the results to the conversation,
#      and go to 1 - the model now KNOWS what the tools found
#
# The iteration budget is a circuit breaker: a confused model cannot
# ping-pong tools forever on your money. Hitting the cap stops cleanly
# and says so in the result; the caller decides how to present that.
#
# The loop owns NOTHING but the cycle: no personality, no storage, no
# client I/O. Identity comes in as the system prompt, capability as the
# toolset, quality as the tier. That is why one engine serves many minds.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.common.log import get_logger
from jarvis.llm.layer import (
    LLMLayer,
    TextCallback,
    assistant_message,
    tool_result_message,
)
from jarvis.llm.tiers import Tier
from jarvis.agentloop.toolset import Toolset

log = get_logger("agentloop.loop")


@dataclass
class LoopResult:
    """What one full loop run produced."""

    text: str                                  # the final answer text
    llm_call_ids: list[str] = field(default_factory=list)
    iterations: int = 0
    tool_calls_made: int = 0
    hit_iteration_budget: bool = False


async def run_agent_loop(
    llm: LLMLayer,
    tier: Tier,
    system: str,
    messages: list[dict[str, Any]],
    toolset: Toolset,
    *,
    actor: str,
    trace_id: str,
    on_text: TextCallback | None = None,
    provider_tools: list[dict[str, Any]] | None = None,
    max_iterations: int = 8,
    max_tokens: int = 2048,
) -> LoopResult:
    """Run the think-act-observe cycle to completion.

    `messages` is the conversation so far (newest last); the loop extends
    a private copy with its own tool traffic and never mutates the
    caller's list.
    """
    convo = list(messages)
    specs = None if toolset.is_empty() else toolset.specs()
    result = LoopResult(text="")

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration

        response = await llm.complete(
            tier,
            system,
            convo,
            actor=actor,
            trace_id=trace_id,
            tools=specs,
            provider_tools=provider_tools,
            max_tokens=max_tokens,
            on_text=on_text,
        )
        result.llm_call_ids.append(response.call_id)
        if response.text:
            # Keep the LAST text the model produced; with tool use the
            # model often says a little, acts, then says the real answer.
            result.text = response.text

        if not response.wants_tools:
            return result   # the model is done talking

        # The model asked for tools: run them, show it the results, loop.
        convo.append(assistant_message(response))
        for call in response.tool_calls:
            output, is_error = await toolset.execute(call.name, call.args)
            result.tool_calls_made += 1
            log.debug("tool executed", extra={
                "tool": call.name, "is_error": is_error, "trace_id": trace_id,
            })
            convo.append(tool_result_message(call.id, output, is_error=is_error))

    # Budget exhausted with the model still asking for tools.
    result.hit_iteration_budget = True
    log.warning("agent loop hit iteration budget", extra={
        "actor": actor, "trace_id": trace_id, "iterations": max_iterations,
    })
    if not result.text:
        result.text = (
            "I ran out of thinking budget before finishing that, sir - "
            "the trail is in my logs if you want the details."
        )
    return result