# =============================================================================
# src/jarvis/jobs/projects.py - the master document
# =============================================================================
#
# One document per project that says everything: what it is, how it
# works, what it is made of, every design choice with the reason and the
# road not taken, and how it got here. Written in the manner of a patent
# specification - abstract, field, background, summary, detailed
# description - because that form forces the two things a project
# write-up usually lacks: a precise statement of the problem, and a
# description complete enough that someone else could build it.
#
# Written ONLY from the database. The model is told to invent nothing;
# where the records are thin the document says so, which is itself
# useful - it shows the owner what he never wrote down.
#
# VERSIONED. Each run is a new artifact and a new row in
# project_documents; earlier versions stay downloadable.
# =============================================================================

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.db.repos.projects import ProjectsRepo
from jarvis.core.projects.service import ProjectService
from jarvis.core.queue.registry import (
    JobContext,
    JobTypeRegistry,
    JobTypeSpec,
    PermanentJobError,
)
from jarvis.llm.layer import LLMLayer, user_message
from jarvis.llm.tiers import Tier

DOCUMENT_JOB_TYPE = "project.document"
ACTOR = "core.projects.document"

_SYSTEM = """You write the master document for one of the owner's projects, \
in the form of a patent specification, from the project records given. \
Markdown, with exactly these sections:

# <Title of the invention or work>
## Abstract
One paragraph, under 150 words: what it is and what it does.
## Field
One or two sentences: the technical area.
## Background
The problem it solves and why existing options fall short - only as far \
as the records support.
## Summary
What the project is, its main parts, and how they fit together.
## Detailed Description
How it works, in enough detail that a capable person could rebuild it. \
Use subsections per subsystem when there are several.
### Bill of Materials
A table of parts (quantity, part, spec, status, cost) when there are any.
### Design Decisions
Every recorded decision: what was chosen, why, and the alternatives \
turned down. Reversed and superseded decisions in a short "Earlier \
choices" list, with what replaced them.
## Development History
The timeline, dated, from the log.
## Status and Open Work
Where it stands now, open tasks, risks.
## People
Who is involved and in what role, if recorded.
## References
Links, if any.

Rules: use ONLY the records. Invent no numbers, parts, reasons or dates. \
Where a section has nothing to go on, write one line saying the records \
do not cover it yet. No preamble, no closing remarks."""


class ProjectDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=10)


class ProjectDocumentOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    filename: str
    version: int
    artifact_id: str


def register_project_jobs(registry: JobTypeRegistry, llm: LLMLayer,
                          repo: ProjectsRepo, service: ProjectService) -> None:

    async def handle(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, ProjectDocumentIn)
        profile = await repo.profile(payload.project_id)
        if profile is None:
            raise PermanentJobError("that project is no longer in the database")
        await ctx.progress("gathering the records")
        records = await service.describe(payload.project_id, full=True)
        history = await repo.history(payload.project_id)
        earlier = [f"- Earlier decision ({d['status']}): {d['decision']}"
                   + (f" - because {d['why']}" if d["why"] else "")
                   for d in history["decisions"]]
        earlier += [f"- Earlier fact ({f['status']}): {f['text']}" for f in history["facts"]]
        full_log = [f"- {e['happened_on']}: {e['text']}" for e in reversed(profile["log"])]
        material = "\n".join([
            records,
            "", "FULL LOG, oldest first:", *(full_log or ["(none)"]),
            "", "SUPERSEDED AND REVERSED RECORDS:", *(earlier or ["(none)"]),
        ])

        await ctx.progress("writing the master document")
        response = await llm.complete(
            Tier.REASONER, _SYSTEM, [user_message(material)],
            actor=ACTOR, trace_id=ctx.trace_id, max_tokens=8000,
        )
        markdown = response.text.strip()
        if not markdown:
            raise RuntimeError("the model returned an empty document")

        version = len(profile["documents"]) + 1
        slug = re.sub(r"[^a-z0-9]+", "-", profile["name"].lower()).strip("-")[:50] or "project"
        filename = f"{slug}-master-v{version}.md"
        artifact_id = await ctx.write_artifact(filename, "text/markdown",
                                               markdown.encode("utf-8"))
        version = await repo.record_document(payload.project_id, artifact_id, ctx.job_id)
        return ProjectDocumentOut(
            summary=f"The master document for {profile['name']}, version {version}.",
            filename=filename, version=version, artifact_id=artifact_id,
        )

    registry.register(JobTypeSpec(
        type=DOCUMENT_JOB_TYPE,
        input_model=ProjectDocumentIn,
        output_model=ProjectDocumentOut,
        execution="idempotent",      # a retry writes the next version; harmless
        requires=[],
        default_priority=5,
        timeout_s=600,
        handler=handle,
    ))
