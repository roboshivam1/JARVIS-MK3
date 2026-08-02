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

# Tool results longer than this are candidates for pruning once they age
# out of the window. Short results (a number, a confirmation) cost
# nothing to keep and are often the thing the model needs to remember.
_PRUNE_THRESHOLD_CHARS = 800


# Tool-result blocks whose content we have already replaced. Tracked
# HERE rather than stamped onto the block, because message dicts go
# verbatim to the provider and it rejects fields it does not recognise.
_PRUNED_MARKER = "[earlier tool output,"


def _prune_stale_tool_results(
    messages: list[dict[str, Any]],
    keep_recent: int,
) -> int:
    """Replace old bulky tool results with a placeholder. Returns how
    many were pruned.

    WHY: every step resends the whole conversation, so a browser
    snapshot taken at step 2 is paid for again at steps 3, 4, 5 and so
    on. Cost grows with the SQUARE of the step count, which is why a
    twenty-step browse costs far more than five times a four-step one.

    A stale snapshot is also not worth keeping on its own merits: the
    accessibility tree of a page the agent left three steps ago is
    stale, and having it in context invites confusion about which page
    is actually in front of it. The agent needs the CURRENT page plus a
    memory of what it did, not a transcript of every DOM it has seen.

    Short results are left alone - they are cheap, and a one-line
    confirmation is often exactly what the model needs to recall.
    """
    # Find bulky tool results, oldest first.
    candidates: list[tuple[int, int, int]] = []   # (msg index, block index, size)
    for m_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for b_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = block.get("content")
            if not isinstance(text, str):
                continue
            if text.startswith(_PRUNED_MARKER):
                continue          # already pruned on an earlier pass
            if len(text) > _PRUNE_THRESHOLD_CHARS:
                candidates.append((m_index, b_index, len(text)))

    prunable = candidates[:-keep_recent] if keep_recent else candidates
    for m_index, b_index, size in prunable:
        block = messages[m_index]["content"][b_index]
        block["content"] = (
            f"{_PRUNED_MARKER} {size} characters, pruned to save "
            f"context - re-run the tool if this is needed again]"
        )

    return len(prunable)


@dataclass
class LoopResult:
    """What one full loop run produced."""

    text: str                                  # the final answer text
    llm_call_ids: list[str] = field(default_factory=list)
    iterations: int = 0
    tool_calls_made: int = 0
    hit_iteration_budget: bool = False
    results_pruned: int = 0                    # bulky tool outputs trimmed


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
    keep_recent_results: int | None = None,
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

        # Trim what the next call has to pay for. Loops with bulky tool
        # output (browser snapshots above all) opt in by setting
        # keep_recent_results; loops with small results leave it off.
        if keep_recent_results is not None:
            pruned = _prune_stale_tool_results(convo, keep_recent_results)
            if pruned:
                result.results_pruned += pruned
                log.debug("pruned stale tool results", extra={
                    "count": pruned, "trace_id": trace_id,
                })

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