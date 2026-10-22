# =============================================================================
# src/jarvis/core/gateway/projects_routes.py - the projects page's API
# =============================================================================
#
# Same bearer token as everything else. Two halves, like the people page:
# the review queue (what the model proposed) and the records themselves
# (which the owner edits directly - review gates the model, not him).
#
#   GET    /projects/pending                    debriefs awaiting review
#   POST   /projects/debrief                    type a debrief
#   POST   /projects/proposals/{id}/approve     approve, optionally edited
#   POST   /projects/proposals/{id}/reject      reject (cascades for new projects)
#   GET    /projects/directory                  every project, with counts
#   GET    /projects/project/{id}               one project in full, with history
#   PATCH  /projects/project/{id}               correct a field
#   DELETE /projects/project/{id}               remove it and everything under it
#   POST   /projects/project/{id}/log           add a log line yourself
#   POST   /projects/project/{id}/document      write a new master document
#   POST   /projects/project/{id}/relate        link two projects
#   POST   /projects/tasks/{id}                 done / dropped / open
#   POST   /projects/parts/{id}                 needed / ordered / have / used / dropped
#   POST   /projects/facts/{id}/retire          a fact that is simply wrong
#   POST   /projects/decisions/{id}/reverse     a decision that no longer stands
# =============================================================================

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jarvis.common.ids import new_ulid
from jarvis.common.jobs import Job
from jarvis.common.projects import ProjectUpdate
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.projects import ProjectReviewError, ProjectsRepo
from jarvis.core.gateway.auth import BearerAuth
from jarvis.core.projects.service import ExtractionFailed, ProjectService
from jarvis.jobs.projects import DOCUMENT_JOB_TYPE
from jarvis.llm.anthropic import ProviderError
from jarvis.llm.layer import NoApiKeyError

_MAX_DEBRIEF_CHARS = 20_000


class _DebriefBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=10, max_length=_MAX_DEBRIEF_CHARS)


class _ApproveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payload: dict[str, Any] | None = None


class _LogBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=2, max_length=2000)
    happened_on: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class _StatusBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


class _RelateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    other_id: str
    kind: Literal["part_of", "depends_on", "spun_off_from", "related"] = "related"
    present: bool = True


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    where = ".".join(str(p) for p in err["loc"]) or "proposal"
    return f"{where}: {err['msg']}"


def projects_router(repo: ProjectsRepo, service: ProjectService, auth: BearerAuth,
                    jobs: JobsRepo | None = None) -> APIRouter:
    router = APIRouter(prefix="/projects", dependencies=[Depends(auth)])

    def _fail(exc: ProjectReviewError) -> HTTPException:
        return HTTPException(status_code=409, detail=str(exc))

    async def _exists(project_id: str) -> dict[str, Any]:
        profile = await repo.profile(project_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="no such project")
        return profile

    # -- review ---------------------------------------------------------------

    @router.get("/pending")
    async def pending() -> dict[str, Any]:
        return {"debriefs": await repo.pending_debriefs(),
                "projects": await repo.directory()}

    @router.post("/debrief")
    async def debrief(body: _DebriefBody) -> dict[str, Any]:
        try:
            outcome = await service.debrief(body.text, trace_id=new_ulid(), source="web")
        except NoApiKeyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ExtractionFailed as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=f"The model call failed: {exc}") from exc
        return {"debrief_id": outcome.debrief_id, "staged": outcome.staged,
                "dropped": outcome.dropped, "summary": outcome.summary()}

    @router.post("/proposals/{proposal_id}/approve")
    async def approve(proposal_id: str, body: _ApproveBody) -> dict[str, Any]:
        try:
            return {"result_id": await repo.approve(proposal_id, body.payload)}
        except ProjectReviewError as exc:
            raise _fail(exc) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=_first_error(exc)) from exc

    @router.post("/proposals/{proposal_id}/reject")
    async def reject(proposal_id: str) -> dict[str, Any]:
        try:
            return {"rejected": await repo.reject(proposal_id)}
        except ProjectReviewError as exc:
            raise _fail(exc) from exc

    # -- records --------------------------------------------------------------

    @router.get("/directory")
    async def directory() -> dict[str, Any]:
        return {"projects": await repo.directory(),
                "pending": await repo.pending_count()}

    @router.get("/project/{project_id}")
    async def project(project_id: str) -> dict[str, Any]:
        profile = await _exists(project_id)
        profile["history"] = await repo.history(project_id)
        return profile

    @router.patch("/project/{project_id}")
    async def update(project_id: str, body: ProjectUpdate) -> dict[str, Any]:
        await _exists(project_id)
        try:
            await repo.update_project(project_id, body)
        except ProjectReviewError as exc:
            raise _fail(exc) from exc
        return {"ok": True}

    @router.delete("/project/{project_id}")
    async def delete(project_id: str) -> dict[str, Any]:
        await _exists(project_id)
        await repo.delete_project(project_id)
        return {"ok": True}

    @router.post("/project/{project_id}/log")
    async def add_log(project_id: str, body: _LogBody) -> dict[str, Any]:
        await _exists(project_id)
        return {"id": await repo.add_log(project_id, body.text, "owner", body.happened_on)}

    @router.post("/project/{project_id}/relate")
    async def relate(project_id: str, body: _RelateBody) -> dict[str, Any]:
        await _exists(project_id)
        await _exists(body.other_id)
        try:
            await repo.relate(project_id, body.other_id, body.kind, body.present)
        except ProjectReviewError as exc:
            raise _fail(exc) from exc
        return {"ok": True}

    @router.post("/project/{project_id}/document")
    async def document(project_id: str) -> dict[str, Any]:
        await _exists(project_id)
        if jobs is None:
            raise HTTPException(status_code=503, detail="the job queue is not available")
        job = Job(type=DOCUMENT_JOB_TYPE, payload={"project_id": project_id},
                  trace_id=new_ulid())
        await jobs.create(job)
        return {"job_id": job.id}

    @router.post("/tasks/{task_id}")
    async def task(task_id: str, body: _StatusBody) -> dict[str, Any]:
        try:
            await repo.resolve_task(task_id, body.status)
        except ProjectReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    @router.post("/parts/{part_id}")
    async def part(part_id: str, body: _StatusBody) -> dict[str, Any]:
        try:
            await repo.set_part_status(part_id, body.status)
        except ProjectReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"ok": True}

    @router.post("/facts/{fact_id}/retire")
    async def retire_fact(fact_id: str) -> dict[str, Any]:
        await repo.retire("facts", fact_id)
        return {"ok": True}

    @router.post("/decisions/{decision_id}/reverse")
    async def reverse(decision_id: str) -> dict[str, Any]:
        await repo.retire("decisions", decision_id)
        return {"ok": True}

    return router
