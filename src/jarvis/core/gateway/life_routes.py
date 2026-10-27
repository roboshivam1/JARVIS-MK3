# =============================================================================
# src/jarvis/core/gateway/life_routes.py - the life page's API
# =============================================================================
#
# Behind the same bearer token as everything else.
#
#   GET    /life/day?date=YYYY-MM-DD     one day: checklist, activities, meals
#   GET    /life/stats?days=28           per-day numbers for the trends
#   GET    /life/summary                 the home page's panel
#   POST   /life/report                  log something in plain words
#   POST   /life/undo                    take back the last report
#   POST   /life/ping                    one event from an iPhone Shortcut:
#                                        wake | sleep | gym_start | gym_end
#   POST   /life/checklist               add a one-off target for a day
#   POST   /life/checklist/{id}          set open | done | missed
#   DELETE /life/checklist/{id}
#   DELETE /life/activities/{id}
#   DELETE /life/meals/{id}
#   GET    /life/habits                  the fixed targets
#   POST   /life/habits                  add one
#   PUT    /life/habits/{id}             edit one
#   DELETE /life/habits/{id}
#   POST   /life/settings                {"wake_by": "07:00" | null}
# =============================================================================

from __future__ import annotations

from datetime import date as Date
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.ids import new_ulid
from jarvis.common.life import HabitIn
from jarvis.core.db.repos.life import LifeError, LifeRepo
from jarvis.core.gateway.auth import BearerAuth
from jarvis.core.life.service import LifeService, ReportFailed
from jarvis.llm.anthropic import ProviderError
from jarvis.llm.layer import NoApiKeyError


class _ReportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=2, max_length=4000)


class _PingBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: Literal["wake", "sleep", "gym_start", "gym_end"]


class _TargetBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    text: str = Field(min_length=2, max_length=200)


class _StatusBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["open", "done", "missed"]


class _SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    wake_by: str | None = None


def _check_date(value: str) -> str:
    try:
        Date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD") from exc
    return value


def life_router(repo: LifeRepo, service: LifeService, auth: BearerAuth) -> APIRouter:
    router = APIRouter(prefix="/life", dependencies=[Depends(auth)])

    @router.get("/day")
    async def day(date: str | None = None) -> dict[str, Any]:
        today = await service.prepare()
        return {"today": today, **await service.day_detail(_check_date(date or today))}

    @router.get("/stats")
    async def stats(days: int = Query(28, ge=7, le=120)) -> dict[str, Any]:
        return await service.stats(days)

    @router.get("/summary")
    async def summary() -> dict[str, Any]:
        today = await service.prepare()
        items = await repo.checklist(today)
        info = await repo.day(today)
        return {"done": sum(1 for i in items if i["status"] == "done"),
                "total": len(items), "wake": info["wake_time"],
                "open": [i["text"] for i in items if i["status"] == "open"]}

    @router.post("/report")
    async def report(body: _ReportBody) -> dict[str, Any]:
        try:
            outcome = await service.report(body.text, trace_id=new_ulid(), source="web")
        except NoApiKeyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ReportFailed as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=f"The model call failed: {exc}") from exc
        return {"summary": outcome.summary()}

    @router.post("/undo")
    async def undo() -> dict[str, Any]:
        return {"summary": await service.undo_last()}

    @router.post("/ping")
    async def ping(body: _PingBody) -> dict[str, Any]:
        return {"summary": await service.ping(body.event)}

    @router.post("/checklist")
    async def add_target(body: _TargetBody) -> dict[str, Any]:
        day = _check_date(body.date)
        await repo.ensure_checklist(day)
        return {"id": await repo.add_target(day, body.text.strip(), None)}

    @router.post("/checklist/{item_id}")
    async def set_item(item_id: str, body: _StatusBody) -> dict[str, Any]:
        await repo.set_item(item_id, body.status)
        return {"ok": True}

    @router.delete("/checklist/{item_id}")
    async def delete_item(item_id: str) -> dict[str, Any]:
        await repo.delete_item(item_id)
        return {"ok": True}

    @router.delete("/activities/{activity_id}")
    async def delete_activity(activity_id: str) -> dict[str, Any]:
        await repo.delete_activity(activity_id)
        return {"ok": True}

    @router.delete("/meals/{meal_id}")
    async def delete_meal(meal_id: str) -> dict[str, Any]:
        await repo.delete_meal(meal_id)
        return {"ok": True}

    @router.get("/habits")
    async def habits() -> dict[str, Any]:
        return {"habits": await repo.habits(), "wake_by": await repo.setting("wake_by")}

    @router.post("/habits")
    async def add_habit(body: HabitIn) -> dict[str, Any]:
        try:
            return {"id": await repo.save_habit(body)}
        except LifeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.put("/habits/{habit_id}")
    async def edit_habit(habit_id: str, body: HabitIn) -> dict[str, Any]:
        try:
            await repo.save_habit(body, habit_id)
        except LifeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @router.delete("/habits/{habit_id}")
    async def delete_habit(habit_id: str) -> dict[str, Any]:
        await repo.delete_habit(habit_id)
        return {"ok": True}

    @router.post("/settings")
    async def settings(body: _SettingsBody) -> dict[str, Any]:
        try:
            return {"wake_by": await service.set_wake_by(body.wake_by)}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
