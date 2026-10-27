# =============================================================================
# src/jarvis/core/db/repos/life.py - the life log's storage
# =============================================================================
#
# Plain reads and writes. Every write made on behalf of a report carries
# that report's id, so undo() can take back exactly one message's worth.
# =============================================================================

from __future__ import annotations

from datetime import date as Date
from typing import Any

from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.life import WEEKDAYS, HabitIn
from jarvis.core.db.database import Database


class LifeError(Exception):
    """Shown to the owner as-is."""


class LifeRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._db.query(sql, params)]

    # -- reports ----------------------------------------------------------------

    async def new_report(self, raw_text: str, source: str) -> str:
        report_id = new_ulid()
        await self._db.execute(
            "INSERT INTO life_reports (id, raw_text, source, created_ts) VALUES (?, ?, ?, ?)",
            (report_id, raw_text, source, utc_now().isoformat()))
        return report_id

    async def set_report_summary(self, report_id: str, summary: str) -> None:
        await self._db.execute(
            "UPDATE life_reports SET summary = ? WHERE id = ?", (summary, report_id))

    async def last_report(self) -> dict[str, Any] | None:
        rows = await self._rows(
            "SELECT * FROM life_reports WHERE undone = 0 ORDER BY id DESC LIMIT 1")
        return rows[0] if rows else None

    async def undo(self, report_id: str) -> None:
        async with self._db.transaction() as tx:
            await tx.execute("DELETE FROM life_activities WHERE report_id = ?", (report_id,))
            await tx.execute("DELETE FROM life_meals WHERE report_id = ?", (report_id,))
            await tx.execute("DELETE FROM life_checklist WHERE report_id = ?", (report_id,))
            await tx.execute(
                "UPDATE life_checklist SET status = 'open', done_ts = NULL, done_report = NULL "
                "WHERE done_report = ?", (report_id,))
            for field in ("wake", "sleep", "note"):
                column = "wake_time" if field == "wake" else (
                    "sleep_time" if field == "sleep" else "note")
                await tx.execute(
                    f"UPDATE life_days SET {column} = NULL, {field}_report = NULL "
                    f"WHERE {field}_report = ?", (report_id,))
            await tx.execute("UPDATE life_reports SET undone = 1 WHERE id = ?", (report_id,))

    # -- days -------------------------------------------------------------------

    async def set_day(self, day: str, field: str, value: str, report_id: str | None) -> None:
        if field not in ("wake", "sleep", "note"):
            raise LifeError(f"Unknown day field {field}")
        column = {"wake": "wake_time", "sleep": "sleep_time", "note": "note"}[field]
        now = utc_now().isoformat()
        await self._db.execute(
            f"INSERT INTO life_days (date, {column}, {field}_report, updated_ts) "
            f"VALUES (?, ?, ?, ?) ON CONFLICT (date) DO UPDATE SET "
            f"{column} = excluded.{column}, {field}_report = excluded.{field}_report, "
            f"updated_ts = excluded.updated_ts",
            (day, value, report_id, now))

    async def day(self, day: str) -> dict[str, Any]:
        rows = await self._rows("SELECT * FROM life_days WHERE date = ?", (day,))
        return rows[0] if rows else {"date": day, "wake_time": None,
                                     "sleep_time": None, "note": None}

    # -- activities -------------------------------------------------------------

    async def add_activity(self, day: str, kind: str, *, subject: str | None,
                           project_id: str | None, started_at: str | None,
                           ended_at: str | None, minutes: int | None, note: str | None,
                           report_id: str | None) -> str:
        row_id = new_ulid()
        await self._db.execute(
            "INSERT INTO life_activities (id, date, kind, subject, project_id, started_at, "
            "ended_at, minutes, note, report_id, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row_id, day, kind, subject, project_id, started_at, ended_at, minutes, note,
             report_id, utc_now().isoformat()))
        return row_id

    async def running(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Activities started and not yet stopped, newest first."""
        if kind:
            return await self._rows(
                "SELECT * FROM life_activities WHERE ended_at IS NULL AND minutes IS NULL "
                "AND started_at IS NOT NULL AND kind = ? ORDER BY id DESC", (kind,))
        return await self._rows(
            "SELECT * FROM life_activities WHERE ended_at IS NULL AND minutes IS NULL "
            "AND started_at IS NOT NULL ORDER BY id DESC")

    async def finish_activity(self, activity_id: str, ended_at: str, minutes: int | None) -> None:
        await self._db.execute(
            "UPDATE life_activities SET ended_at = ?, minutes = ? WHERE id = ?",
            (ended_at, minutes, activity_id))

    async def activities(self, day: str) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT a.*, p.name AS project FROM life_activities a "
            "LEFT JOIN projects p ON p.id = a.project_id WHERE a.date = ? "
            "ORDER BY COALESCE(a.started_at, '99'), a.created_ts", (day,))

    async def delete_activity(self, activity_id: str) -> None:
        await self._db.execute("DELETE FROM life_activities WHERE id = ?", (activity_id,))

    # -- meals ------------------------------------------------------------------

    async def add_meal(self, day: str, slot: str, status: str, what: str | None,
                       place: str | None, report_id: str | None) -> str:
        row_id = new_ulid()
        await self._db.execute(
            "INSERT INTO life_meals (id, date, slot, status, what, place, report_id, "
            "created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row_id, day, slot, status, what, place, report_id, utc_now().isoformat()))
        return row_id

    async def meals(self, day: str) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT * FROM life_meals WHERE date = ? ORDER BY CASE slot "
            "WHEN 'breakfast' THEN 0 WHEN 'lunch' THEN 1 WHEN 'snack' THEN 2 ELSE 3 END",
            (day,))

    async def delete_meal(self, meal_id: str) -> None:
        await self._db.execute("DELETE FROM life_meals WHERE id = ?", (meal_id,))

    # -- habits and the checklist -----------------------------------------------

    async def habits(self, active_only: bool = False) -> list[dict[str, Any]]:
        where = "WHERE active = 1" if active_only else ""
        rows = await self._rows(f"SELECT * FROM life_habits {where} ORDER BY created_ts")
        for r in rows:
            r["weekdays"] = [d for d in r["weekdays"].split(",") if d]
            r["active"] = bool(r["active"])
        return rows

    async def save_habit(self, data: HabitIn, habit_id: str | None = None) -> str:
        days = ",".join(d for d in WEEKDAYS if d in data.weekdays)
        clash = await self._rows(
            "SELECT id FROM life_habits WHERE name = ? COLLATE NOCASE", (data.name,))
        if clash and clash[0]["id"] != habit_id:
            raise LifeError(f"There is already a fixed target called {data.name!r}.")
        if habit_id is None:
            habit_id = new_ulid()
            await self._db.execute(
                "INSERT INTO life_habits (id, name, kind, weekdays, active, created_ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (habit_id, data.name, data.kind, days, int(data.active), utc_now().isoformat()))
        else:
            await self._db.execute(
                "UPDATE life_habits SET name = ?, kind = ?, weekdays = ?, active = ? "
                "WHERE id = ?", (data.name, data.kind, days, int(data.active), habit_id))
        return habit_id

    async def delete_habit(self, habit_id: str) -> None:
        await self._db.execute("DELETE FROM life_habits WHERE id = ?", (habit_id,))

    async def ensure_checklist(self, day: str) -> None:
        """Copy the day's fixed targets onto its checklist. Idempotent."""
        weekday = WEEKDAYS[Date.fromisoformat(day).weekday()]
        now = utc_now().isoformat()
        for habit in await self.habits(active_only=True):
            if weekday in habit["weekdays"]:
                await self._db.execute(
                    "INSERT OR IGNORE INTO life_checklist (id, date, text, habit_id, "
                    "status, created_ts) VALUES (?, ?, ?, ?, 'open', ?)",
                    (new_ulid(), day, habit["name"], habit["id"], now))

    async def close_past(self, today: str) -> int:
        """Open items on days already over become missed."""
        return await self._db.execute_returning_changes(
            "UPDATE life_checklist SET status = 'missed' WHERE status = 'open' AND date < ?",
            (today,))

    async def checklist(self, day: str) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT c.*, h.kind AS habit_kind FROM life_checklist c "
            "LEFT JOIN life_habits h ON h.id = c.habit_id WHERE c.date = ? "
            "ORDER BY c.habit_id IS NULL, c.created_ts", (day,))

    async def add_target(self, day: str, text: str, report_id: str | None) -> str:
        existing = await self._rows(
            "SELECT id FROM life_checklist WHERE date = ? AND text = ? COLLATE NOCASE",
            (day, text))
        if existing:
            return str(existing[0]["id"])
        row_id = new_ulid()
        await self._db.execute(
            "INSERT INTO life_checklist (id, date, text, status, report_id, created_ts) "
            "VALUES (?, ?, ?, 'open', ?, ?)", (row_id, day, text, report_id, utc_now().isoformat()))
        return row_id

    async def set_item(self, item_id: str, status: str, report_id: str | None = None) -> None:
        if status not in ("open", "done", "missed"):
            raise LifeError(f"Not a checklist status: {status}")
        done_ts = utc_now().isoformat() if status == "done" else None
        await self._db.execute(
            "UPDATE life_checklist SET status = ?, done_ts = ?, done_report = ? WHERE id = ?",
            (status, done_ts, report_id if status == "done" else None, item_id))

    async def tick_habits_of_kind(self, day: str, kind: str, report_id: str | None) -> list[str]:
        """An activity of a habit's kind on that day ticks the habit's item."""
        rows = await self._rows(
            "SELECT c.id, c.text FROM life_checklist c JOIN life_habits h ON h.id = c.habit_id "
            "WHERE c.date = ? AND h.kind = ? AND c.status != 'done'", (day, kind))
        for r in rows:
            await self.set_item(r["id"], "done", report_id)
        return [r["text"] for r in rows]

    async def delete_item(self, item_id: str) -> None:
        await self._db.execute("DELETE FROM life_checklist WHERE id = ?", (item_id,))

    # -- settings ---------------------------------------------------------------

    async def setting(self, key: str) -> str | None:
        rows = await self._rows("SELECT value FROM life_settings WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    async def set_setting(self, key: str, value: str | None) -> None:
        if value is None:
            await self._db.execute("DELETE FROM life_settings WHERE key = ?", (key,))
        else:
            await self._db.execute(
                "INSERT INTO life_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))

    # -- ranges, for insights ---------------------------------------------------

    async def range_rows(self, start: str, end: str) -> dict[str, list[dict[str, Any]]]:
        """Everything between two dates inclusive, by table."""
        return {
            "days": await self._rows(
                "SELECT date, wake_time, sleep_time, note FROM life_days "
                "WHERE date BETWEEN ? AND ?", (start, end)),
            "activities": await self._rows(
                "SELECT a.date, a.kind, a.minutes, a.subject, p.name AS project "
                "FROM life_activities a LEFT JOIN projects p ON p.id = a.project_id "
                "WHERE a.date BETWEEN ? AND ?", (start, end)),
            "meals": await self._rows(
                "SELECT date, slot, status FROM life_meals WHERE date BETWEEN ? AND ?",
                (start, end)),
            "checklist": await self._rows(
                "SELECT c.date, c.status, c.text, c.habit_id, h.kind AS habit_kind "
                "FROM life_checklist c LEFT JOIN life_habits h ON h.id = c.habit_id "
                "WHERE c.date BETWEEN ? AND ?", (start, end)),
        }
