# =============================================================================
# src/jarvis/jobs/claude_code.py - the Claude Code shift, Core side
# =============================================================================
#
# Three subagents can run two ways. The API way is the original: the
# Core (or, for DAEDALUS, the Mac's sandbox) drives the model through
# the API, billed per token. The Claude Code way hands the same brief to
# `claude -p` on the Mac, billed to the owner's subscription.
#
#   engineer   (DAEDALUS)  code.task       <->  code.cc
#   researcher (ATHENA)    research.brief  <->  research.cc
#   writer     (CALLIOPE)  write.document  <->  write.cc
#
# The choice is made when the job is created (see core/queue/
# claude_code.py): Claude Code when a worker that advertises it is
# online, the API otherwise. Claude Code jobs that cannot finish - the
# plan is out of usage, or the Mac went away with the job still queued -
# are handed to the API version rather than left to rot.
#
# Metadata only here. The handlers live on the worker, which is the only
# place `claude` is installed and logged in.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.queue.registry import JobTypeRegistry, JobTypeSpec

CLAUDE_CODE = "claude-code"          # the capability tag a worker advertises

CODE_CC = "code.cc"
RESEARCH_CC = "research.cc"
WRITE_CC = "write.cc"

# agent -> (Claude Code job type, API job type)
ROUTES: dict[str, tuple[str, str]] = {
    "engineer": (CODE_CC, "code.task"),
    "researcher": (RESEARCH_CC, "research.brief"),
    "writer": (WRITE_CC, "write.document"),
}


class ClaudeCodeIn(BaseModel):
    """One brief, plus any files it needs. input_artifacts is the name
    the worker gateway already ships files for."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=10)
    input_artifacts: list[str] = Field(default_factory=list)


class ClaudeCodeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    report: str = ""
    filename: str = ""
    completed: bool = True
    # True when Claude Code could not do the work (usage limit) and the
    # Core should run the API version instead. The summary says so.
    fallback: bool = False
    cost_usd: float | None = None     # what the API would have charged


def register_claude_code_job_metadata(registry: JobTypeRegistry) -> None:
    for job_type, timeout_s in ((CODE_CC, 1800), (RESEARCH_CC, 900), (WRITE_CC, 900)):
        registry.register(JobTypeSpec(
            type=job_type,
            input_model=ClaudeCodeIn,
            output_model=ClaudeCodeOut,
            # A retry starts the brief over. Nothing outward happens
            # without the owner: pushes are denied outright.
            execution="idempotent",
            requires=[CLAUDE_CODE],
            default_priority=5,
            timeout_s=timeout_s,
            handler=None,
        ))
