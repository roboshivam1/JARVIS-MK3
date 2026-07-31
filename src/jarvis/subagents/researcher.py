# =============================================================================
# src/jarvis/subagents/researcher.py - ATHENA, the researcher
# =============================================================================
#
# A subagent is a CONFIGURATION of the shared agent loop: its own persona,
# its own tools, its own tier - executed inside a job, with no direct
# line to the owner. Results flow back through the job system.
#
# ATHENA's toolkit is the provider's server-side web search: the provider
# runs searches mid-thought inside the API call, so this file has no
# client-side tools at all. Iteration budget is small because the search
# loop happens on the provider's side, not ours.
#
# The persona demands a fixed document shape (summary, findings, sources)
# because a brief is read COLD, later, possibly on the phone - it must
# stand alone without the conversation that requested it.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from jarvis.agentloop.loop import run_agent_loop
from jarvis.agentloop.toolset import Toolset
from jarvis.llm.layer import LLMLayer, user_message
from jarvis.llm.tiers import Tier

ACTOR = "subagent.researcher"

ATHENA_PERSONA_V1 = """\
You are ATHENA, the research specialist of a personal AI system. You are
given one self-contained research brief and you produce one finished
document. You have web search available - use it; claims about the
current state of anything must be grounded in what you find, not in
recollection. Prefer primary sources.

Your output is a markdown document with exactly this shape:

# <title reflecting the brief>

## Summary
Three to six sentences a busy person can read on a phone. The direct
answer first, nuance after.

## Findings
The substance, organised under short subheadings. Concrete numbers,
names, and dates over generalities. Note disagreements between sources
where they exist rather than smoothing them over.

## Sources
A bulleted list of the sources you actually used: title and URL.

Rules: no preamble before the title, no closing remarks after sources.
If the brief asks something the web genuinely cannot settle, say so
plainly in the summary instead of manufacturing confidence.
"""

# The provider-side web search tool. max_uses caps runaway search bills:
# a brief that cannot be grounded in eight searches needs a better brief,
# not a bigger budget.
_WEB_SEARCH = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 8,
}


@dataclass(frozen=True)
class ResearchOutcome:
    """What one research run produced."""

    brief_markdown: str
    llm_call_ids: list[str]


async def run_researcher(
    llm: LLMLayer,
    brief: str,
    *,
    trace_id: str,
) -> ResearchOutcome:
    """Execute one research brief to a finished document."""
    result = await run_agent_loop(
        llm,
        Tier.REASONER,
        ATHENA_PERSONA_V1,
        [user_message(brief)],
        Toolset(),                      # no client-side tools
        actor=ACTOR,
        trace_id=trace_id,
        provider_tools=[_WEB_SEARCH],
        max_iterations=2,               # search happens provider-side
        max_tokens=4096,                # documents need room
    )
    return ResearchOutcome(
        brief_markdown=result.text,
        llm_call_ids=result.llm_call_ids,
    )