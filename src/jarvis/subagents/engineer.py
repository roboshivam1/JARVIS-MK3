# =============================================================================
# src/jarvis/subagents/engineer.py - DAEDALUS, the engineer
# =============================================================================
#
# The subagent that computes. Writes Python, runs it in a kernel jail,
# reads what happened, fixes it, repeats - the write-run-observe cycle
# IS the capability. The tools are deliberately thin because the loop is
# where the intelligence lives.
#
# WHAT IT IS FOR, and what it is not: this is a computation companion,
# not an autonomous replacement for a person at a keyboard. It shines on
# work that is well-specified, verifiable by running it, and tedious -
# data analysis, calculations, format conversion, checking whether a
# claim survives contact with the numbers. It is deliberately not aimed
# at building large projects unsupervised, where an architectural
# mistake at step four gets faithfully built upon through step forty.
#
# ERRORS ARE DATA. A traceback comes back as tool output like any other
# result; the model reads it and fixes the code. That is how a subagent
# that cannot write perfect code first time still produces working
# results - it iterates against a real interpreter rather than guessing
# in one shot.
#
# THE SANDBOX IS NOT NEGOTIABLE FROM IN HERE. No network, no filesystem
# beyond a disposable directory, no environment secrets. The prompt says
# so mostly to save the model from wasting steps discovering it.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from jarvis.agentloop.loop import run_agent_loop
from jarvis.agentloop.toolset import Toolset
from jarvis.llm.layer import LLMLayer, user_message
from jarvis.llm.tiers import Tier

ACTOR = "subagent.engineer"

DAEDALUS_PERSONA_V1 = """\
You are DAEDALUS, the engineer of a personal AI system. You are given a
task, you write Python to accomplish it, you run that code, and you
report what you found or built.

How you work:
- Write code, run it, READ THE OUTPUT, then decide. Do not write three
  scripts before running any of them.
- Errors are information, not failure. A traceback tells you exactly
  what to fix. Fix it and run again.
- Start by looking at your inputs. Print the shape of a dataframe, the
  first few rows, the column names - before writing analysis that
  assumes a structure you have not verified.
- Small steps beat large ones. A script that does one thing and prints
  its result is easier to correct than one that does everything and
  fails somewhere inside.
- KEEP EACH SCRIPT UNDER ABOUT 60 LINES. If a task needs more, split it
  across runs - files persist between them, so step two can load what
  step one saved. Long scripts get truncated mid-generation, fail on a
  syntax error you cannot see, and cost the owner real money producing
  code that never runs.
- Do not write defensive code for problems you have not seen. No
  try/except around everything, no handling of edge cases the data may
  not contain. Run it, see what actually breaks, fix that.
- Files you create persist between runs within this task, so you can
  clean data in one step and analyse it in the next.

Your environment:
- Python with pandas, numpy, matplotlib, and scipy available.
- NO NETWORK. You cannot download anything, call any API, or install
  packages. Everything you need is either already installed or was
  given to you as an input file.
- A fresh directory that is destroyed when this task ends. You cannot
  see or touch anything else on the machine.
- Use save_artifact for anything the owner should actually receive:
  charts, cleaned datasets, generated documents. Files you do not save
  are working scratch and vanish.

Reporting:
- State what you did, what the result was, and how confident you are in
  it. If the data was messier than expected, say so.
- Numbers with units and context, not bare figures.
- If you could not finish, say what stopped you and what you tried. A
  clear account of failure is worth more than a vague success.
- Do not paste your code into the report unless the owner asked to see
  it; they asked for the answer.
"""


@dataclass(frozen=True)
class EngineerOutcome:
    """What one code task produced."""

    report: str
    steps_taken: int
    runs: int
    hit_step_budget: bool
    llm_call_ids: list[str]


async def run_engineer(
    llm: LLMLayer,
    task: str,
    toolset: Toolset,
    *,
    trace_id: str,
    max_steps: int = 20,
) -> EngineerOutcome:
    """Execute one code task to a report.

    The toolset arrives already built and already bound to the guard.
    DAEDALUS receives capabilities; it does not choose them.
    """
    result = await run_agent_loop(
        llm,
        Tier.REASONER,
        DAEDALUS_PERSONA_V1,
        [user_message(task)],
        toolset,
        actor=ACTOR,
        trace_id=trace_id,
        max_iterations=max_steps,
        # Deliberately tight. Hitting this cap truncates code
        # mid-statement, producing a syntax error the model then has to
        # debug - having already paid for 4,000 output tokens at 5x the
        # input rate. A lower ceiling pushes it toward the small scripts
        # the persona asks for, and makes overrun cheap when it happens.
        max_tokens=2048,
        # Code output is compact next to browser snapshots, but a long
        # debugging session still accumulates. Keep more history than
        # PROTEUS does: earlier errors are often the context for
        # understanding a later one.
        keep_recent_results=4,
    )
    return EngineerOutcome(
        report=result.text,
        steps_taken=result.iterations,
        runs=result.tool_calls_made,
        hit_step_budget=result.hit_iteration_budget,
        llm_call_ids=result.llm_call_ids,
    )
