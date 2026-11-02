# =============================================================================
# src/jarvis/core/db/repos/health.py - HYGIEIA's storage
# =============================================================================

from __future__ import annotations

import json
import re
from typing import Any

from jarvis.common.ids import utc_now
from jarvis.core.db.database import Database

NUTRIENTS = ("kcal", "protein_g", "carbs_g", "fat_g", "fibre_g", "sodium_mg", "sugar_g")


def food_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


class HealthRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._db.query(sql, params)]

    # -- profile ------------------------------------------------------------------

    async def profile(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in await self._rows("SELECT * FROM health_profile")}

    async def set_profile(self, key: str, value: str | None) -> None:
        if value is None or value == "":
            await self._db.execute("DELETE FROM health_profile WHERE key = ?", (key,))
        else:
            await self._db.execute(
                "INSERT INTO health_profile (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))

    # -- meals and their nutrition -------------------------------------------------

    async def meal(self, meal_id: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM life_meals WHERE id = ?", (meal_id,))
        return rows[0] if rows else None

    async def save_nutrition(self, meal_id: str, items: list[dict[str, Any]],
                             confidence: str, source: str = "estimate") -> None:
        totals = {n: round(sum(float(i.get(n) or 0) for i in items), 1) for n in NUTRIENTS}
        junk = sum(1 for i in items if i.get("junk"))
        await self._db.execute(
            "INSERT INTO health_meal_nutrition (meal_id, items, kcal, protein_g, carbs_g, "
            "fat_g, fibre_g, sodium_mg, sugar_g, junk, confidence, source, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (meal_id) DO UPDATE SET "
            "items = excluded.items, kcal = excluded.kcal, protein_g = excluded.protein_g, "
            "carbs_g = excluded.carbs_g, fat_g = excluded.fat_g, fibre_g = excluded.fibre_g, "
            "sodium_mg = excluded.sodium_mg, sugar_g = excluded.sugar_g, junk = excluded.junk, "
            "confidence = excluded.confidence, source = excluded.source",
            (meal_id, json.dumps(items), *(totals[n] for n in NUTRIENTS), junk, confidence,
             source, utc_now().isoformat()))

    async def nutrition(self, meal_id: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM health_meal_nutrition WHERE meal_id = ?",
                                (meal_id,))
        if not rows:
            return None
        row = rows[0]
        row["items"] = json.loads(row["items"])
        return row

    async def meals_between(self, start: str, end: str) -> list[dict[str, Any]]:
        """Every logged meal between two dates, with its nutrition if known."""
        rows = await self._rows(
            "SELECT m.id, m.date, m.slot, m.status, m.what, m.place, m.created_ts, "
            "n.items, n.kcal, n.protein_g, n.carbs_g, n.fat_g, n.fibre_g, n.sodium_mg, "
            "n.sugar_g, n.junk, n.confidence, n.source FROM life_meals m "
            "LEFT JOIN health_meal_nutrition n ON n.meal_id = m.id "
            "WHERE m.date BETWEEN ? AND ? ORDER BY m.date, CASE m.slot WHEN 'breakfast' THEN 0 "
            "WHEN 'lunch' THEN 1 WHEN 'snack' THEN 2 ELSE 3 END, m.created_ts", (start, end))
        for r in rows:
            r["items"] = json.loads(r["items"]) if r["items"] else []
        return rows

    # -- the food memory -----------------------------------------------------------

    async def foods_like(self, names: list[str], limit: int = 30) -> list[dict[str, Any]]:
        """Known foods whose names share a word with any of these."""
        words = {w for n in names for w in food_key(n).split() if len(w) > 2}
        if not words:
            return []
        clauses = " OR ".join("key LIKE ?" for _ in words)
        return await self._rows(
            f"SELECT * FROM health_foods WHERE {clauses} ORDER BY times_eaten DESC LIMIT ?",
            (*(f"%{w}%" for w in words), limit))

    async def remember_food(self, item: dict[str, Any], place: str | None,
                            source: str = "learned", eaten_on: str | None = None) -> None:
        """Learn (or refresh) one food. Values per ONE serving."""
        key = food_key(item["name"])
        if not key:
            return
        existing = await self._rows("SELECT source FROM health_foods WHERE key = ?", (key,))
        bump = 1 if eaten_on else 0
        if existing:
            # The owner's own numbers are never overwritten by an estimate.
            if existing[0]["source"] == "owner" and source != "owner":
                await self._db.execute(
                    "UPDATE health_foods SET times_eaten = times_eaten + ?, "
                    "last_eaten = COALESCE(?, last_eaten), place = COALESCE(place, ?) "
                    "WHERE key = ?", (bump, eaten_on, place, key))
                return
        per = item.get("per_serving") or item
        await self._db.execute(
            "INSERT INTO health_foods (key, name, place, serving, kcal, protein_g, carbs_g, "
            "fat_g, fibre_g, sodium_mg, sugar_g, junk, source, times_eaten, last_eaten, "
            "updated_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (key) DO UPDATE SET place = COALESCE(excluded.place, health_foods.place), "
            "serving = excluded.serving, kcal = excluded.kcal, protein_g = excluded.protein_g, "
            "carbs_g = excluded.carbs_g, fat_g = excluded.fat_g, fibre_g = excluded.fibre_g, "
            "sodium_mg = excluded.sodium_mg, sugar_g = excluded.sugar_g, junk = excluded.junk, "
            "source = CASE WHEN health_foods.source = 'owner' THEN 'owner' ELSE excluded.source END, "
            "times_eaten = health_foods.times_eaten + ?, "
            "last_eaten = COALESCE(excluded.last_eaten, health_foods.last_eaten), "
            "updated_ts = excluded.updated_ts",
            (key, item["name"], place, str(item.get("serving") or "1 serving"),
             *(float(per.get(n) or 0) for n in NUTRIENTS), int(bool(item.get("junk"))),
             source, bump, eaten_on, utc_now().isoformat(), bump))

    async def foods(self, place: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if place:
            return await self._rows(
                "SELECT * FROM health_foods WHERE place = ? ORDER BY times_eaten DESC, name "
                "LIMIT ?", (place, limit))
        return await self._rows(
            "SELECT * FROM health_foods ORDER BY times_eaten DESC, name LIMIT ?", (limit,))

    async def food(self, key: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM health_foods WHERE key = ?", (key,))
        return rows[0] if rows else None

    async def delete_food(self, key: str) -> None:
        await self._db.execute("DELETE FROM health_foods WHERE key = ?", (key,))

    # -- the mess menu ------------------------------------------------------------

    async def menu(self, weekday: int, slot: str) -> list[str]:
        rows = await self._rows(
            "SELECT items FROM health_menu WHERE slot = ? AND (weekday = ? OR weekday IS NULL) "
            "ORDER BY weekday IS NULL", (slot, weekday))
        return [part.strip() for r in rows for part in r["items"].split(",") if part.strip()]

    async def menu_rows(self) -> list[dict[str, Any]]:
        return await self._rows("SELECT * FROM health_menu ORDER BY slot, weekday")

    async def set_menu(self, weekday: int | None, slot: str, items: str) -> None:
        if weekday is None:
            await self._db.execute(
                "DELETE FROM health_menu WHERE weekday IS NULL AND slot = ?", (slot,))
        else:
            await self._db.execute(
                "DELETE FROM health_menu WHERE weekday = ? AND slot = ?", (weekday, slot))
        if items.strip():
            await self._db.execute(
                "INSERT INTO health_menu (weekday, slot, items) VALUES (?, ?, ?)",
                (weekday, slot, items.strip()))

    # -- sleep, from the life log ---------------------------------------------------

    async def days(self, start: str, end: str) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT date, wake_time, sleep_time FROM life_days WHERE date BETWEEN ? AND ? "
            "ORDER BY date", (start, end))
