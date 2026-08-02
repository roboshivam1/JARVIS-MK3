# =============================================================================
# src/jarvis/jobs/code.py - the code.task job type
# =============================================================================
#
# Core-side metadata for DAEDALUS. The Core never runs this: it requires
# the sandbox-exec capability, which only a worker advertises, so the
# built-in executor passes it by and the job waits for a capable worker.
#
# IDEMPOTENT, not resumable: sandbox state is destroyed between runs by
# design, so there is nothing meaningful to checkpoint. A retry starts
# the task over, which is wasteful but harmless - the sandbox has no
# outward side effects to repeat.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.queue.registry import JobTypeRegistry, JobTypeSpec


class CodeTaskIn(BaseModel):
    """Input: a self-contained task, plus any files it needs.

    input_artifacts names artifacts the Core should hand to the sandbox.
    That is the ONLY way data reaches DAEDALUS - it cannot go looking
    for files, so anything it should analyse must be given to it.
    """

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=10)
    input_artifacts: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=20, ge=1, le=40)


class CodeTaskOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    report: str
    runs: int
    completed: bool


def register_code_job_metadata(registry: JobTypeRegistry) -> None:
    """Core-side registration: metadata only, no handler."""
    registry.register(JobTypeSpec(
        type="code.task",
        input_model=CodeTaskIn,
        output_model=CodeTaskOut,
        execution="idempotent",
        requires=["sandbox-exec"],
        default_priority=5,
        timeout_s=900,
        handler=None,
    ))
