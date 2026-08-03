# =============================================================================
# src/jarvis/jobs/writing.py - the write.document job type
# =============================================================================
#
# CALLIOPE bound into the queue, and unlike the other subagents this one
# runs ON THE CORE: requires:[] means the built-in executor picks it up,
# so documents get written whether or not a worker is online.
#
# The document lives in a BUFFER the tools write into, not in the
# model's responses. A three-thousand-word document produced section by
# section never exists in any single response, so it cannot be a return
# value - the tools accumulate it and the handler saves the result.
#
# IDEMPOTENT: rerunning produces a fresh document. Wasteful after a
# crash, harmless otherwise - and the second attempt is often better.
# =============================================================================

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from jarvis.agentloop.policies import WRITER, create_guard
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec
from jarvis.llm.layer import LLMLayer
from jarvis.subagents.writer import run_writer


class WriteDocumentIn(BaseModel):
    """Input: a brief, and the material to write from."""

    model_config = ConfigDict(extra="forbid")

    brief: str = Field(min_length=10)
    # Artifact ids the writer may read - a research brief, notes, a
    # dataset summary. This is how material reaches CALLIOPE; it cannot
    # go looking for anything.
    source_artifacts: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=15, ge=1, le=30)


class WriteDocumentOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    filename: str
    word_count: int
    completed: bool


def register_writing_jobs(
    registry: JobTypeRegistry,
    llm: LLMLayer,
    artifacts: ArtifactsRepo,
) -> None:
    """Register writing job types. Core-side, with a real handler."""
    guard = create_guard()

    async def handle(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, WriteDocumentIn)

        # The document under construction. Sections are appended and
        # revised here rather than being carried in the conversation,
        # which is what lets a document outgrow any single response.
        sections: list[str] = []

        class _NoArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class _ReadSourceArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            artifact_id: str

        class _AppendArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            markdown: str = Field(min_length=1)

        class _ReviseArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            section_index: int = Field(ge=0)
            markdown: str = Field(min_length=1)

        async def list_sources(_: _NoArgs) -> str:
            if not payload.source_artifacts:
                return "No source material was provided for this brief."
            lines: list[str] = []
            for artifact_id in payload.source_artifacts:
                artifact = await artifacts.get(artifact_id)
                if artifact is not None:
                    lines.append(
                        f"{artifact.id} | {artifact.name} | {artifact.size} bytes"
                    )
            return "\n".join(lines) or "No readable source material."

        async def read_source(args: _ReadSourceArgs) -> str:
            if args.artifact_id not in payload.source_artifacts:
                return (
                    f"error: {args.artifact_id} is not among this task's "
                    f"sources. Use list_sources to see what is available."
                )
            artifact = await artifacts.get(args.artifact_id)
            if artifact is None:
                return f"error: no artifact {args.artifact_id}"
            try:
                content = artifacts.read_bytes(artifact).decode(
                    "utf-8", errors="replace"
                )
            except Exception as exc:
                return f"error reading {artifact.name}: {exc}"
            # Generous but bounded: a source large enough to blow the
            # context window helps nobody.
            return content[:40_000]

        async def append_section(args: _AppendArgs) -> str:
            sections.append(args.markdown.strip())
            words = sum(len(s.split()) for s in sections)
            await ctx.progress(f"section {len(sections)} written")
            return (
                f"Appended as section {len(sections) - 1}. "
                f"Document is now {words} words across {len(sections)} sections."
            )

        async def read_document(_: _NoArgs) -> str:
            """Rereading is how revision happens - the model cannot
            improve what it cannot see."""
            if not sections:
                return "The document is empty."
            return "\n\n".join(
                f"--- section {i} ---\n{text}"
                for i, text in enumerate(sections)
            )

        async def revise_section(args: _ReviseArgs) -> str:
            if args.section_index >= len(sections):
                return (
                    f"error: no section {args.section_index}. "
                    f"The document has {len(sections)}."
                )
            sections[args.section_index] = args.markdown.strip()
            return f"Section {args.section_index} replaced."

        toolset = Toolset(guard=guard, actor=WRITER)
        for name, description, model, handler in (
            ("artifact_list_sources",
             "List the source material available for this brief.",
             _NoArgs, list_sources),
            ("artifact_read_source",
             "Read one source document in full. Do this before planning "
             "what to write.",
             _ReadSourceArgs, read_source),
            ("artifact_append_section",
             "Add a finished section to the document. Write one section "
             "at a time rather than the whole document at once.",
             _AppendArgs, append_section),
            ("artifact_read_document",
             "Read back everything written so far. Do this before "
             "finishing, to catch repetition and thin passages.",
             _NoArgs, read_document),
            ("artifact_revise_section",
             "Replace a section with an improved version.",
             _ReviseArgs, revise_section),
        ):
            toolset.register(InlineTool(
                name=name, description=description,
                args_model=model, handler=handler,
            ))

        await ctx.progress("CALLIOPE starting")
        outcome = await run_writer(
            llm, payload.brief, toolset,
            trace_id=ctx.trace_id,
            max_steps=payload.max_steps,
        )

        document = "\n\n".join(sections)
        if not document.strip():
            # The model talked instead of writing. Salvage its final
            # message rather than delivering nothing.
            document = outcome.report

        filename = _filename_for(document, ctx.job_id)
        await ctx.write_artifact(
            filename, "text/markdown", document.encode("utf-8")
        )

        word_count = len(document.split())
        summary = outcome.report.strip().split("\n\n")[0][:400] or (
            f"Written: {filename}, {word_count} words."
        )

        return WriteDocumentOut(
            summary=summary,
            filename=filename,
            word_count=word_count,
            completed=not outcome.hit_step_budget,
        )

    registry.register(JobTypeSpec(
        type="write.document",
        input_model=WriteDocumentIn,
        output_model=WriteDocumentOut,
        execution="idempotent",
        requires=[],          # Core-side: writing needs no worker
        default_priority=5,
        timeout_s=900,
        handler=handle,
    ))


def _filename_for(document: str, job_id: str) -> str:
    """A readable filename from the document's own title."""
    for line in document.splitlines():
        if line.startswith("# "):
            slug = re.sub(r"[^a-z0-9]+", "-", line[2:].strip().lower())
            slug = slug.strip("-")[:60]
            if slug:
                return f"{slug}.md"
    return f"document-{job_id[-6:]}.md"
