# =============================================================================
# src/jarvis/core/gateway/money_routes.py - the money page's API
# =============================================================================
#
# Behind the bearer token like everything else.
#
#   GET    /money/period?period=2026-10     summary + every transaction
#   GET    /money/summary                   the home page's panel
#   GET    /money/status                    is the mailbox being read
#   GET    /money/categories
#   POST   /money/categories                {"name": "...", "is_spend": true}
#   GET    /money/to-sort                   payees without a category
#   GET    /money/rules
#   POST   /money/rules                     {"payee", "category", "scope", "label"}
#   DELETE /money/rules/{payee_key}
#   PATCH  /money/transactions/{id}         one payment: category / scope / note
#   DELETE /money/transactions/{id}
#   POST   /money/cash                      {"amount", "what", "category", "date"}
#   POST   /money/alert                     paste an alert's text by hand
#   POST   /money/poll                      read the mailbox now
# =============================================================================

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.db.repos.money import MoneyError
from jarvis.core.gateway.auth import BearerAuth
from jarvis.core.money.service import MoneyService

Scope = Literal["personal", "basement_supply"]


class _Category(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=40)
    is_spend: bool = True


class _Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payee: str = Field(min_length=1)
    category: str
    scope: Scope = "personal"
    label: str | None = Field(None, max_length=60)


class _Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str | None = None
    scope: Scope | None = None
    note: str | None = Field(None, max_length=200)


class _Cash(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: str
    what: str = Field(min_length=1, max_length=120)
    category: str | None = None
    date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    scope: Scope = "personal"


class _Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=20, max_length=20000)


def money_router(money: MoneyService, auth: BearerAuth) -> APIRouter:
    router = APIRouter(prefix="/money", dependencies=[Depends(auth)])
    repo = money.repo

    def _fail(exc: MoneyError) -> HTTPException:
        return HTTPException(status_code=422, detail=str(exc))

    @router.get("/period")
    async def period(period: str = "month") -> dict[str, Any]:
        try:
            start, end, label = money.period(period)
        except MoneyError as exc:
            raise _fail(exc) from exc
        data = await money.summary(start, end)
        data["label"] = label
        data["today"] = money.today().isoformat()
        return data

    @router.get("/summary")
    async def summary() -> dict[str, Any]:
        start, end, _ = money.period("month")
        month = await money.summary(start, end)
        today = await money.summary(end, end)
        return {"month_paise": month["total_paise"], "today_paise": today["total_paise"],
                "to_sort": len(await repo.to_sort()), **await money.status()}

    @router.get("/status")
    async def status() -> dict[str, Any]:
        return await money.status()

    @router.get("/categories")
    async def categories() -> dict[str, Any]:
        return {"categories": await repo.categories()}

    @router.post("/categories")
    async def add_category(body: _Category) -> dict[str, Any]:
        try:
            return {"name": await repo.add_category(body.name, body.is_spend)}
        except MoneyError as exc:
            raise _fail(exc) from exc

    @router.get("/to-sort")
    async def to_sort() -> dict[str, Any]:
        return {"payees": await repo.to_sort()}

    @router.get("/rules")
    async def rules() -> dict[str, Any]:
        return {"rules": await repo.rules()}

    @router.post("/rules")
    async def set_rule(body: _Rule) -> dict[str, Any]:
        try:
            return {"summary": await money.categorise_payee(
                body.payee, body.category, scope=body.scope, label=body.label or None)}
        except MoneyError as exc:
            raise _fail(exc) from exc

    @router.delete("/rules/{key}")
    async def delete_rule(key: str) -> dict[str, Any]:
        await repo.delete_rule(key)
        return {"ok": True}

    @router.patch("/transactions/{txn_id}")
    async def edit(txn_id: str, body: _Edit) -> dict[str, Any]:
        try:
            await money.edit(txn_id, category=body.category, scope=body.scope, note=body.note)
        except MoneyError as exc:
            raise _fail(exc) from exc
        return {"ok": True}

    @router.delete("/transactions/{txn_id}")
    async def delete(txn_id: str) -> dict[str, Any]:
        await repo.delete(txn_id)
        return {"ok": True}

    @router.post("/cash")
    async def cash(body: _Cash) -> dict[str, Any]:
        try:
            txn = await money.add_cash(body.amount, body.what, category=body.category,
                                       day=body.date, scope=body.scope, source="web")
        except MoneyError as exc:
            raise _fail(exc) from exc
        return {"id": txn["id"]}

    @router.post("/alert")
    async def alert(body: _Alert) -> dict[str, Any]:
        try:
            txn = await money.ingest_text(body.text, source="web")
        except MoneyError as exc:
            raise _fail(exc) from exc
        return {"recorded": txn is not None}

    @router.post("/poll")
    async def poll() -> dict[str, Any]:
        return await money.poll()

    return router
