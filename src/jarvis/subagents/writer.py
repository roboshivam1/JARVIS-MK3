# =============================================================================
# src/jarvis/subagents/writer.py - CALLIOPE, the writer
# =============================================================================
#
# The odd subagent out. ATHENA has search, PROTEUS has a browser,
# DAEDALUS has a sandbox - each defined by a capability. A writer's
# capability is writing, which the model does natively, so what is the
# loop for?
#
# Three things one model call cannot do:
#
#   LENGTH. A three-thousand-word document exceeds what a single
#     response comfortably produces. Section by section, with the
#     outline in view, works far better than one enormous generation
#     that drifts and repeats.
#   REVISION. The interesting part of writing is rereading and fixing.
#     A loop can produce a draft, look at it, and improve it. A single
#     call cannot look at its own output.
#   MATERIAL. Documents are written FROM something - a research brief,
#     notes, a dataset someone analysed. Reading inputs is a tool call.
#
# So the tools manage a document in progress; they do not produce prose.
# The prose is the model's own work, which is the point.
#
# RUNS ON THE CORE, not a worker. Writing is API-bound: no browser, no
# sandbox, nothing a laptop provides. It should work with the laptop
# shut, which is the "useful with zero workers" principle applied to the
# last subagent.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from jarvis.agentloop.loop import run_agent_loop
from jarvis.agentloop.toolset import Toolset
from jarvis.llm.layer import LLMLayer, user_message
from jarvis.llm.tiers import Tier

ACTOR = "subagent.writer"

CALLIOPE_PERSONA_V1 = """\
You are CALLIOPE, the writer of a personal AI system. You are given a
brief and you produce a finished document.

How you work:
- Read your source material FIRST, before planning anything. What you
  can actually support determines what you can actually write.
- Outline before drafting. A document without a shape wanders.
- Write section by section. Finish one, then the next. Do not try to
  produce the whole thing in a single burst - it drifts, repeats
  itself, and thins out toward the end.
- Reread what you have written before declaring it done. You will find
  a paragraph that says nothing, a claim you did not support, and a
  sentence you wrote twice. Fix them.

How you write:
- Say the thing. Then support it. Not the reverse, and not a throat-
  clearing paragraph before either.
- Concrete over abstract. A number, a name, an example beats a
  characterisation every time.
- Cut what does not earn its place. "It is important to note that" is
  never doing work. Neither is a summary of what you are about to say.
- Vary sentence length or the reader stops seeing individual sentences.
- No filler transitions - "moreover", "furthermore", "in conclusion".
  If the connection between two paragraphs is not obvious, the problem
  is the paragraphs.
- Match the register to the purpose. A school newsletter and a
  technical proposal are not the same document with different words.

On honesty:
- Write only what your sources support. If the brief asks for a claim
  you cannot back, say so in the document rather than inventing
  support for it.
- Do not pad to reach a length. A tight document that is short is
  better than a long one that repeats.

Your finished document is markdown, and it is the whole output - no
preamble to the owner, no notes about your process. If you have
something to tell him, that goes in your final message, not in the
document.
"""


@dataclass(frozen=True)
class WriterOutcome:
    """What one writing task produced."""

    document: str
    report: str
    steps_taken: int
    hit_step_budget: bool
    llm_call_ids: list[str]


async def run_writer(
    llm: LLMLayer,
    brief: str,
    toolset: Toolset,
    *,
    trace_id: str,
    max_steps: int = 15,
) -> WriterOutcome:
    """Execute one writing brief to a finished document.

    The document itself is accumulated through the toolset's document
    buffer rather than returned as text: a long document produced
    section by section never exists in one model response, so it cannot
    be a return value.
    """
    result = await run_agent_loop(
        llm,
        Tier.REASONER,
        CALLIOPE_PERSONA_V1,
        [user_message(brief)],
        toolset,
        actor=ACTOR,
        trace_id=trace_id,
        max_iterations=max_steps,
        max_tokens=4096,
        # Source material is bulky and read once; the document itself
        # lives in the buffer, not in the conversation. Keeping the two
        # most recent tool results is enough to know what just happened.
        keep_recent_results=2,
    )
    return WriterOutcome(
        document="",        # filled by the caller from the buffer
        report=result.text,
        steps_taken=result.iterations,
        hit_step_budget=result.hit_iteration_budget,
        llm_call_ids=result.llm_call_ids,
    )
