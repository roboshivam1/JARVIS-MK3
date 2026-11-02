# =============================================================================
# src/jarvis/core/gateway/health_routes.py - the health page's API
# =============================================================================
#
#   GET    /health-data/day?date=YYYY-MM-DD   meals with nutrients, totals, targets
#   GET    /health-data/trend?days=14         per-day totals and sleep
#   GET    /health-data/summary               the home page's panel
#   POST   /health-data/meals/{id}/fix        {"correction": "half a plate"}
#   GET    /health-data/foods?place=mess
#   POST   /health-data/foods                 teach a food's values
#   DELETE /health-data/foods/{key}
#   GET    /health-data/profile               profile, targets, limits
#   POST   /health-data/profile               {"key": "weight_kg", "value": "68"}
#   GET    /health-data/menu
#   PUT    /health-data/menu                  {"weekday": 0-6 | null, "slot", "items"}
#
# The prefix is /health-data because /health is the gateway's liveness check.
# =============================================================================

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.ids import new_ulid
from jarvis.core.gateway.auth import BearerAuth
from jarvis.core.health.service import HealthService


class _Fix(BaseModel):
    model_config = ConfigDict(extra="forbid")
    correction: str = Field(min_length=3, max_length=300)


class _Food(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=80)
    place: Literal["mess", "campus", "outside", "room", "home"]
    serving: str = Field(min_length=1, max_length=60)
    kcal: float = Field(ge=0, le=3000)
    protein_g: float = Field(ge=0, le=200)
    carbs_g: float = Field(0, ge=0, le=500)
    fat_g: float = Field(0, ge=0, le=300)
    fibre_g: float = Field(0, ge=0, le=100)
    sodium_mg: float = Field(0, ge=0, le=10000)
    sugar_g: float = Field(0, ge=0, le=300)
    junk: bool = False


class _Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    value: str | None = None


class _Menu(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weekday: int | None = Field(None, ge=0, le=6)
    slot: Literal["breakfast", "lunch", "snack", "dinner"]
    items: str = Field(max_length=1000)


def health_router(health: HealthService, auth: BearerAuth) -> APIRouter:
    router = APIRouter(prefix="/health-data", dependencies=[Depends(auth)])
    repo = health.repo

    def _day(value: str | None) -> str:
        if not value:
            return health.today().isoformat()
        try:
            return Date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD") from exc

    @router.get("/day")
    async def day(date: str | None = None) -> dict[str, Any]:
        d = await health.day_totals(_day(date))
        d["targets"] = (await health.targets()).as_dict()
        d["today"] = health.today().isoformat()
        d["standing"] = await health.standing_line(d["date"]) if d["logged"] else ""
        return d

    @router.get("/trend")
    async def trend(days: int = Query(14, ge=7, le=60)) -> dict[str, Any]:
        end = health.today()
        start = end - timedelta(days=days - 1)
        out = []
        for i in range(days):
            d = await health.day_totals((start + timedelta(days=i)).isoformat())
            out.append({"date": d["date"], "logged": d["logged"], **d["totals"]})
        nights = await health.sleep(start - timedelta(days=1), end - timedelta(days=1))
        return {"days": out, "sleep": nights, "targets": (await health.targets()).as_dict()}

    @router.get("/summary")
    async def summary() -> dict[str, Any]:
        d = await health.day_totals(health.today().isoformat())
        return {"logged": d["logged"], **d["totals"],
                "targets": (await health.targets()).as_dict()}

    @router.post("/meals/{meal_id}/fix")
    async def fix(meal_id: str, body: _Fix) -> dict[str, Any]:
        if await repo.meal(meal_id) is None:
            raise HTTPException(status_code=404, detail="no such meal")
        try:
            standing = await health.fix_meal(meal_id, body.correction, trace_id=new_ulid())
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"standing": standing}

    @router.get("/foods")
    async def foods(place: str | None = None) -> dict[str, Any]:
        return {"foods": await repo.foods(place)}

    @router.post("/foods")
    async def teach(body: _Food) -> dict[str, Any]:
        data = body.model_dump()
        await repo.remember_food({"name": body.name, "serving": body.serving,
                                  "junk": body.junk, "per_serving": data},
                                 body.place, source="owner")
        return {"ok": True}

    @router.delete("/foods/{key}")
    async def forget(key: str) -> dict[str, Any]:
        await repo.delete_food(key)
        return {"ok": True}

    @router.get("/profile")
    async def profile() -> dict[str, Any]:
        return {"profile": await repo.profile(), "targets": (await health.targets()).as_dict()}

    @router.post("/profile")
    async def set_profile(body: _Profile) -> dict[str, Any]:
        result = await health.set_profile(body.key, body.value)
        if not result.startswith("Set "):
            raise HTTPException(status_code=422, detail=result)
        return {"summary": result}

    @router.get("/menu")
    async def menu() -> dict[str, Any]:
        return {"menu": await repo.menu_rows()}

    @router.put("/menu")
    async def set_menu(body: _Menu) -> dict[str, Any]:
        await repo.set_menu(body.weekday, body.slot, body.items)
        return {"ok": True}

    return router
