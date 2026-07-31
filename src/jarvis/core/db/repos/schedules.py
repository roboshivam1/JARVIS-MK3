# =============================================================================
# src/jarvis/core/db/repos/schedules.py - the schedules table
# =============================================================================
#
# The only code touching schedules.
#
# ensure() is the seeding entry point: create this schedule if a
# schedule of that name does not already exist, otherwise leave the
# existing one alone. Boot runs it every time; the unique index on name
# makes that safe, and leaving existing rows untouched means the owner's
# edits (a changed hour, a disabled schedule) survive restarts instead of
# being reset by code.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite

from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.common.schedules import Schedule, ScheduleKind
from jarvis.core.db.database import Database

log = get_logger("core.db.schedules")


class SchedulesRepo:
    """Create, find, and advance schedules."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def ensure(self, schedule: Schedule) -> bool:
        """Create the schedule unless one with that name exists. Returns
        True if it was created. Existing rows are left exactly as they
        are - the owner's changes outrank the code's defaults."""
        existing = await self._db.query_one(
            "SELECT id FROM schedules WHERE name = ?", (schedule.name,)
        )
        if existing is not None:
            return False
        await self._db.execute(
            "INSERT INTO schedules "
            "(id, name, kind, cron_expr, interval_s, job_type, job_payload, "
            " priority, enabled, next_fire_ts, last_fired_ts, fire_count, "
            " created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                schedule.id, schedule.name, schedule.kind.value,
                schedule.cron_expr, schedule.interval_s, schedule.job_type,
                json.dumps(schedule.job_payload), schedule.priority,
                1 if schedule.enabled else 0,
                schedule.next_fire_ts.isoformat(),
                schedule.last_fired_ts.isoformat() if schedule.last_fired_ts else None,
                schedule.fire_count, schedule.created_ts.isoformat(),
            ),
        )
        log.info("schedule created", extra={
            "schedule": schedule.name, "kind": schedule.kind.value,
            "next_fire_ts": schedule.next_fire_ts.isoformat(),
        })
        return True

    async def due(self, now: datetime | None = None) -> list[Schedule]:
        """Enabled schedules whose time has come or passed."""
        moment = (now or utc_now()).isoformat()
        rows = await self._db.query(
            "SELECT * FROM schedules WHERE enabled = 1 AND next_fire_ts <= ? "
            "ORDER BY next_fire_ts ASC",
            (moment,),
        )
        return [self._to_schedule(r) for r in rows]

    async def mark_fired(self, schedule_id: str, next_fire_ts: datetime) -> None:
        """Record a firing and set the next one."""
        await self._db.execute(
            "UPDATE schedules SET last_fired_ts = ?, next_fire_ts = ?, "
            "fire_count = fire_count + 1 WHERE id = ?",
            (utc_now().isoformat(), next_fire_ts.isoformat(), schedule_id),
        )

    async def disable(self, schedule_id: str) -> None:
        """Retire a schedule - what a `once` schedule does after firing."""
        await self._db.execute(
            "UPDATE schedules SET enabled = 0 WHERE id = ?", (schedule_id,)
        )

    async def all(self) -> list[Schedule]:
        rows = await self._db.query("SELECT * FROM schedules ORDER BY name ASC")
        return [self._to_schedule(r) for r in rows]

    @staticmethod
    def _to_schedule(row: aiosqlite.Row) -> Schedule:
        return Schedule.model_validate({
            "id": row["id"],
            "name": row["name"],
            "kind": ScheduleKind(row["kind"]),
            "cron_expr": row["cron_expr"],
            "interval_s": row["interval_s"],
            "job_type": row["job_type"],
            "job_payload": json.loads(row["job_payload"]),
            "priority": row["priority"],
            "enabled": bool(row["enabled"]),
            "next_fire_ts": row["next_fire_ts"],
            "last_fired_ts": row["last_fired_ts"],
            "fire_count": row["fire_count"],
            "created_ts": row["created_ts"],
        })