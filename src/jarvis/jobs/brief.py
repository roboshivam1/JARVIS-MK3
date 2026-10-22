# =============================================================================
# src/jarvis/jobs/brief.py - the morning brief, as a job
# =============================================================================
#
# A schedule fires this once a morning. The work lives in BriefService;
# the job is only the handle the scheduler can pull, so the brief gets
# retries, a trace, and a row in the job log like everything else.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.initiative.brief import BRIEF_JOB_TYPE, BriefService, describe_facts
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec


class BriefIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="scheduled")


class BriefOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: dict[str, int]
    chars: int


def register_brief_jobs(registry: JobTypeRegistry, brief: BriefService) -> None:
    """Core-side: everything it reads is in the Core's database."""

    async def handle(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, BriefIn)
        await ctx.progress("writing the morning brief")
        text, facts = await brief.send(ctx.trace_id, job_id=ctx.job_id)
        return BriefOut(sections=describe_facts(facts), chars=len(text))

    registry.register(JobTypeSpec(
        type=BRIEF_JOB_TYPE,
        input_model=BriefIn,
        output_model=BriefOut,
        # Safe to retry: the notification is keyed to this job, so a
        # second attempt cannot queue a second brief.
        execution="idempotent",
        requires=[],
        default_priority=4,
        timeout_s=180,
        handler=handle,
    ))
