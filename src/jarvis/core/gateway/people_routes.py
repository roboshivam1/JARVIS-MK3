# =============================================================================
# src/jarvis/core/gateway/people_routes.py - the review page's API
# =============================================================================
#
# Everything the people review page needs, behind the same bearer token
# as every other authed route:
#
#   GET  /people/pending                    debriefs awaiting review, plus
#                                           the directory for the "about"
#                                           dropdown when editing
#   POST /people/debrief                    type a debrief instead of saying it
#   POST /people/proposals/{id}/approve     approve, optionally with edits
#   POST /people/proposals/{id}/reject      reject (cascades for new people)
#
# Errors are mapped to status codes the page can explain in plain words:
#   409 - the proposal cannot be decided as asked (already decided,
#         depends on a rejected person, ...)
#   422 - an edit or a debrief failed validation
#   502 - the model call failed; 503 - no API key configured
# =============================================================================

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jarvis.common.ids import new_ulid
from jarvis.core.db.repos.people import PeopleRepo, ReviewError
from jarvis.core.gateway.auth import BearerAuth
from jarvis.core.people.service import ExtractionFailed, PeopleService
from jarvis.llm.anthropic import ProviderError
from jarvis.llm.layer import NoApiKeyError

# Generous for a spoken ramble, bounded so one paste cannot become a
# very expensive extraction call.
_MAX_DEBRIEF_CHARS = 20_000


class _DebriefBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=10, max_length=_MAX_DEBRIEF_CHARS)


class _ApproveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any] | None = None   # the edited proposal, if edited


def _first_error(exc: ValidationError) -> str:
    """One readable line from a pydantic error: which field, what is wrong."""
    err = exc.errors()[0]
    where = ".".join(str(p) for p in err["loc"]) or "proposal"
    return f"{where}: {err['msg']}"


def people_router(repo: PeopleRepo, service: PeopleService, auth: BearerAuth) -> APIRouter:
    router = APIRouter(prefix="/people", dependencies=[Depends(auth)])

    @router.get("/pending")
    async def pending() -> dict[str, Any]:
        return {
            "debriefs": await repo.pending_debriefs(),
            "people": await repo.roster(limit=1000),
        }

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
        return {
            "debrief_id": outcome.debrief_id,
            "staged": outcome.staged,
            "dropped": outcome.dropped,
            "summary": outcome.summary(),
        }

    @router.post("/proposals/{proposal_id}/approve")
    async def approve(proposal_id: str, body: _ApproveBody) -> dict[str, Any]:
        try:
            result_id = await repo.approve(proposal_id, body.payload)
        except ReviewError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=_first_error(exc)) from exc
        return {"result_id": result_id}

    @router.post("/proposals/{proposal_id}/reject")
    async def reject(proposal_id: str) -> dict[str, Any]:
        try:
            rejected = await repo.reject(proposal_id)
        except ReviewError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"rejected": rejected}

    return router
