# =============================================================================
# src/jarvis/jobs/research.py - the research.brief job type
# =============================================================================
#
# Binds ATHENA into the queue: typed input (a self-contained brief),
# typed output, and the execution contract.
#
# The document is now stored as an ARTIFACT, and the result carries the
# artifact id plus a chat-sized summary - not the document text. This is
# the result-size rule in practice: results are pointers and summaries;
# bulk content is a file. It keeps job rows small, makes the brief
# deliverable as a real file, and means the same document can be sent to
# any client without re-reading a database row.
#
# Declared IDEMPOTENT: re-running research after a crash is wasteful but
# harmless - no outward side effects. A retry writes a second artifact,
# which is the honest outcome (two research runs, two documents).
#
# requires: [] - API-bound work, so the Core runs it via the built-in
# core-worker. Useful with zero workers online.
# =============================================================================

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec
from jarvis.llm.layer import LLMLayer
from jarvis.subagents.researcher import run_researcher


class ResearchBriefIn(BaseModel):
    """Input: one brief that survives being read cold, hours later,
    with no conversation attached."""

    model_config = ConfigDict(extra="forbid")

    brief: str = Field(min_length=10)


class ResearchBriefOut(BaseModel):
    """Output: a chat-sized summary and a pointer to the document."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    artifact_id: str
    filename: str


def register_research_jobs(registry: JobTypeRegistry, llm: LLMLayer) -> None:
    """Register research job types. Called once at boot, after the LLM
    layer exists (the handler closes over it)."""

    async def handle(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, ResearchBriefIn)
        await ctx.progress("ATHENA researching")
        outcome = await run_researcher(llm, payload.brief, trace_id=ctx.trace_id)

        markdown = outcome.brief_markdown
        filename = _filename_for(markdown, ctx.job_id)
        artifact_id = await ctx.write_artifact(
            filename, "text/markdown", markdown.encode("utf-8")
        )
        await ctx.progress("brief written")

        return ResearchBriefOut(
            summary=_extract_summary(markdown),
            artifact_id=artifact_id,
            filename=filename,
        )

    registry.register(JobTypeSpec(
        type="research.brief",
        input_model=ResearchBriefIn,
        output_model=ResearchBriefOut,
        execution="idempotent",
        requires=[],
        default_priority=5,
        timeout_s=420,          # searches plus a long document take minutes
        handler=handle,
    ))


def _extract_summary(markdown: str) -> str:
    """The Summary section's text, or a truncated head as fallback. A
    missing section means ATHENA broke format - we still deliver; a
    finished document is not failed over a heading."""
    collecting = False
    collected: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## summary"):
            collecting = True
            continue
        if collecting and stripped.startswith("#"):
            break
        if collecting and stripped:
            collected.append(stripped)
    return " ".join(collected) if collected else markdown[:500]


def _filename_for(markdown: str, job_id: str) -> str:
    """A readable filename from the document's own title, so what lands
    in the owner's chat is 'solid-state-batteries.md', not a ULID. Falls
    back to the job id when there is no usable title."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            slug = re.sub(r"[^a-z0-9]+", "-", line[2:].strip().lower())
            slug = slug.strip("-")[:60]
            if slug:
                return f"{slug}.md"
    return f"brief-{job_id[-6:]}.md"