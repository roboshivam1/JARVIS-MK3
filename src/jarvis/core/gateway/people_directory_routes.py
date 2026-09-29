# =============================================================================
# src/jarvis/core/gateway/people_directory_routes.py - the People section
# =============================================================================
#
# Reading and editing the people database directly, as opposed to
# reviewing what the model proposed (people_routes.py). Same bearer
# token, same prefix; a separate module because the two halves of the
# page have nothing to do with each other.
#
#   GET    /people/directory              everyone and every group
#   GET    /people/person/{id}            one person in full, with history
#   PATCH  /people/person/{id}            correct a field
#   DELETE /people/person/{id}            remove them entirely
#   POST   /people/person/{id}/alias      add or drop a nickname
#   POST   /people/person/{id}/facts      write a fact yourself
#   POST   /people/merge                  two rows, one person
#   PATCH  /people/facts/{id}             correct a fact (supersedes it)
#   DELETE /people/facts/{id}             retire a fact, nothing replaces it
#   POST   /people/commitments/{id}       mark done or dropped
#   DELETE /people/dates/{id}             remove a date
#   GET    /people/groups                 list
#   POST   /people/groups                 create
#   PATCH  /people/groups/{id}            rename or recategorise
#   DELETE /people/groups/{id}            remove the group, keep the people
#   POST   /people/groups/{id}/members    add or remove someone
#
# These edits skip the review queue deliberately. Review exists to gate
# the extraction model; the owner correcting his own records does not
# need his own approval.
# =============================================================================

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.people import GroupIn, OwnerFact, PersonUpdate
from jarvis.core.db.repos.people import PeopleRepo, ReviewError
from jarvis.core.gateway.auth import BearerAuth


class _AliasBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str = Field(min_length=1, max_length=60)
    present: bool = True          # false removes it


class _MemberBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_id: str
    member: bool = True


class _MergeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep: str
    merge: str


class _ResolveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["done", "dropped"]


def people_directory_router(repo: PeopleRepo, auth: BearerAuth) -> APIRouter:
    router = APIRouter(prefix="/people", dependencies=[Depends(auth)])

    def _fail(exc: ReviewError) -> HTTPException:
        return HTTPException(status_code=409, detail=str(exc))

    # -- reading --------------------------------------------------------------

    @router.get("/directory")
    async def directory() -> dict[str, Any]:
        return await repo.directory()

    @router.get("/summary")
    async def summary() -> dict[str, Any]:
        """One call for the home page panel: what is waiting, what is
        owed, what is coming up. Dates are returned raw and the page
        works out which are near - it knows the owner's clock."""
        debriefs = await repo.pending_debriefs()
        directory = await repo.directory()
        return {
            "pending_debriefs": len(debriefs),
            "pending_proposals": sum(len(d["proposals"]) for d in debriefs),
            "people": len(directory["people"]),
            "groups": len(directory["groups"]),
            "commitments": await repo.open_commitments(),
            "dates": await repo.all_dates(),
        }

    @router.get("/person/{person_id}")
    async def person(person_id: str) -> dict[str, Any]:
        detail = await repo.person_detail(person_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="No such person.")
        return detail

    # -- a person -------------------------------------------------------------

    @router.patch("/person/{person_id}")
    async def update_person(person_id: str, body: PersonUpdate) -> dict[str, bool]:
        await repo.update_person(person_id, body)
        return {"ok": True}

    @router.delete("/person/{person_id}")
    async def delete_person(person_id: str) -> dict[str, bool]:
        await repo.delete_person(person_id)
        return {"ok": True}

    @router.post("/person/{person_id}/alias")
    async def alias(person_id: str, body: _AliasBody) -> dict[str, bool]:
        await repo.set_alias(person_id, body.alias, body.present)
        return {"ok": True}

    @router.post("/person/{person_id}/facts")
    async def add_fact(person_id: str, body: OwnerFact) -> dict[str, str]:
        return {"fact_id": await repo.add_fact(person_id, body)}

    @router.post("/merge")
    async def merge(body: _MergeBody) -> dict[str, bool]:
        try:
            await repo.merge_people(body.keep, body.merge)
        except ReviewError as exc:
            raise _fail(exc) from exc
        return {"ok": True}

    # -- what hangs off a person ----------------------------------------------

    @router.patch("/facts/{fact_id}")
    async def edit_fact(fact_id: str, body: OwnerFact) -> dict[str, str]:
        try:
            return {"fact_id": await repo.edit_fact(fact_id, body)}
        except ReviewError as exc:
            raise _fail(exc) from exc

    @router.delete("/facts/{fact_id}")
    async def retire_fact(fact_id: str) -> dict[str, bool]:
        await repo.retire_fact(fact_id)
        return {"ok": True}

    @router.post("/commitments/{commitment_id}")
    async def resolve(commitment_id: str, body: _ResolveBody) -> dict[str, bool]:
        try:
            await repo.resolve_commitment(commitment_id, body.status)
        except ReviewError as exc:
            raise _fail(exc) from exc
        return {"ok": True}

    @router.delete("/dates/{date_id}")
    async def delete_date(date_id: str) -> dict[str, bool]:
        await repo.delete_date(date_id)
        return {"ok": True}

    # -- groups ---------------------------------------------------------------

    @router.get("/groups")
    async def groups() -> list[dict[str, Any]]:
        return await repo.groups()

    @router.post("/groups")
    async def create_group(body: GroupIn) -> dict[str, str]:
        try:
            return {"group_id": await repo.create_group(body)}
        except ReviewError as exc:
            raise _fail(exc) from exc

    @router.patch("/groups/{group_id}")
    async def update_group(group_id: str, body: GroupIn) -> dict[str, bool]:
        await repo.update_group(group_id, body)
        return {"ok": True}

    @router.delete("/groups/{group_id}")
    async def delete_group(group_id: str) -> dict[str, bool]:
        await repo.delete_group(group_id)
        return {"ok": True}

    @router.post("/groups/{group_id}/members")
    async def member(group_id: str, body: _MemberBody) -> dict[str, bool]:
        await repo.set_membership(group_id, body.person_id, body.member)
        return {"ok": True}

    return router
