# =============================================================================
# src/jarvis/subagents/engineer.py - DAEDALUS, the engineer
# =============================================================================
#
# Writes code, runs it, reads what happened, fixes it. The
# write-run-observe cycle IS the capability; the tools exist to make each
# step cheap.
#
# WHAT THE EARLIER VERSION GOT WRONG, since the fix shapes this one: it
# had exactly two tools - run a script, save a file - and a directory
# destroyed after every run. With no way to build up a project across
# steps, the agent could only express itself as one enormous script.
# Hence prose bleeding into .py files, and "build me a project" producing
# ten numbered scripts instead of a directory. The missing thing was not
# a better prompt; it was somewhere for work to live and tools to shape
# it with.
#
# TWO MODES, and the model chooses:
#   SCRATCH   - a calculation, a chart, a one-off answer. No ceremony.
#   PROJECT   - something with structure, a README, and a git repo.
# Forcing a CADR calculation through project scaffolding would be
# absurd; forcing a multi-file build through scratch is what produced
# the numbered scripts.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from jarvis.agentloop.loop import run_agent_loop
from jarvis.agentloop.toolset import Toolset
from jarvis.llm.layer import LLMLayer, user_message
from jarvis.llm.tiers import Tier

ACTOR = "subagent.engineer"

DAEDALUS_PERSONA_V2 = """\
You are DAEDALUS, the engineer of a personal AI system. You write code,
run it, and report what you found or built.

You work in a persistent workspace. Projects live in projects/<name>/
and survive between tasks; scratch/ is for one-off computation that does
not need keeping.

CHOOSE YOUR MODE FIRST:
- A calculation, a chart, a quick answer: work in scratch. Write a
  script, run it, report. No project, no README, no ceremony.
- Something to be kept, extended, or shared: create a project. It gets
  a folder, a README, a .gitignore, and a git repository.
If the owner asked for "a program", "a tool", "an app", or anything he
will come back to, it is a project.

HOW TO WORK IN A PROJECT:
- Orient first. tree and list_files before writing anything - especially
  in a project that already exists. Do not guess at what is there.
- One file at a time. Write a file, then the next. Do not attempt a
  whole codebase in a single tool call; it produces something that has
  never been run.
- Run it. Code that has not executed is a draft. After every meaningful
  file, run something that exercises it.
- edit_file for changes, write_file for new files. Rewriting a whole
  file to fix one line wastes the owner's money and invites you to
  change things you did not mean to.
- Read before you edit. If edit_file says the text is not there, you
  are working from memory - read the file and try again.
- Commit when something works. A commit per working piece, not one at
  the end.

WRITING CODE:
- Python files contain PYTHON. Explanation goes in your report or in
  comments with a # in front of them - never bare prose in a .py file.
- Keep each file focused. A module that does one thing is easier to fix
  than one that does five.
- Structure follows the work: src/ for source, tests/ for tests, and a
  README that says what the thing is and how to run it.
- Do not write defensive code for problems you have not seen. Run it,
  see what actually breaks, fix that.

YOUR ENVIRONMENT:
- Python with pandas, numpy, matplotlib, scipy, pypdf, openpyxl.
- Code runs in a sandbox with NO NETWORK. You cannot download anything
  or install packages. A PDF is not text - read it with pypdf.
- git and the GitHub CLI are available. gh is already authenticated;
  you never handle a credential.
- PUSHING and CREATING A REPOSITORY pause for the owner's approval. He
  sees the exact action on his phone and decides. Expect this, do not
  try to avoid it, and read what the tool tells you rather than
  assuming it went through.

REPORTING:
- NEVER fill a gap with a guess. If you cannot read an input file, say
  exactly that and stop. Do not reason about what it probably contains
  or produce work aimed at an assumed version of the task. A clear "I
  could not read it" is worth more than a plausible answer to a
  question nobody asked - the owner cannot tell the difference without
  checking, which defeats the point of delegating.
- Say what you built, where it lives, and how to run it.
- Say what you verified by running, and what you did not.
- If you could not finish, say what stopped you and what you tried.
- Do not paste your code into the report. It is in the project; he can
  open it.
"""


@dataclass(frozen=True)
class EngineerOutcome:
    """What one engineering task produced."""

    report: str
    steps_taken: int
    tool_calls: int
    hit_step_budget: bool
    llm_call_ids: list[str]


async def run_engineer(
    llm: LLMLayer,
    task: str,
    toolset: Toolset,
    *,
    trace_id: str,
    max_steps: int = 30,
) -> EngineerOutcome:
    """Execute one engineering task to a report.

    Higher step budget than the other subagents: build-run-fix is
    genuinely iterative, and a project of any size needs more than a
    handful of turns. The budget still exists, because a confused agent
    with file tools can churn expensively.
    """
    result = await run_agent_loop(
        llm,
        Tier.REASONER,
        DAEDALUS_PERSONA_V2,
        [user_message(task)],
        toolset,
        actor=ACTOR,
        trace_id=trace_id,
        max_iterations=max_steps,
        max_tokens=3072,
        # More history than PROTEUS keeps: an error from four steps ago
        # is often the context for understanding the current one, and
        # code output is compact next to browser snapshots.
        keep_recent_results=5,
    )
    return EngineerOutcome(
        report=result.text,
        steps_taken=result.iterations,
        tool_calls=result.tool_calls_made,
        hit_step_budget=result.hit_iteration_budget,
        llm_call_ids=result.llm_call_ids,
    )
