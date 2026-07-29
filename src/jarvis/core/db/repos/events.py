# =============================================================================
# src/jarvis/core/db/repos/events.py - repository for the events table
# =============================================================================
#
# The ONLY code that reads or writes the events table. Callers deal in
# Event models exclusively; SQL and row shapes stay inside this module.
#
# Write path notes:
#   - append() can join a caller's transaction. This matters later: writing
#     a job and its job.enqueued event must be one atomic unit, so the repo
#     must be able to participate in a larger write instead of always
#     committing alone.
#   - The table's triggers forbid UPDATE and DELETE, so this repo does not
#     even offer such methods. Append and read: that is the whole API.
#
# Read path notes:
#   - Rows are re-validated through the Event model on the way out. A
#     corrupted or hand-edited row fails loudly here, at the boundary,
#     instead of flowing onward as unvalidated data.
# =============================================================================

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from jarvis.common.events import Event
from jarvis.core.db.database import Database, _Transaction


class EventsRepo:
    """Append and query the event log."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- writes ---------------------------------------------------------------

    async def append(self, event: Event, tx: _Transaction | None = None) -> None:
        """Persist one event. If tx is given, the write joins that
        transaction and commits (or rolls back) with it; otherwise it
        commits immediately on its own."""
        sql = (
            "INSERT INTO events "
            "(id, ts, kind, source, session_id, job_id, trace_id, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            event.id,
            event.ts.isoformat(),
            event.kind,
            event.source,
            event.session_id,
            event.job_id,
            event.trace_id,
            json.dumps(event.payload, ensure_ascii=False, default=str),
        )
        if tx is not None:
            await tx.execute(sql, params)
        else:
            await self._db.execute(sql, params)

    # -- reads ----------------------------------------------------------------

    async def by_trace(self, trace_id: str) -> list[Event]:
        """Every event in one causal chain, in chronological (= id) order."""
        rows = await self._db.query(
            "SELECT * FROM events WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        )
        return [self._to_event(r) for r in rows]

    async def recent(self, kind: str | None = None, limit: int = 50) -> list[Event]:
        """The latest events, optionally filtered to one kind, newest first.
        Uses the (kind, id) index when kind is given."""
        if kind is not None:
            rows = await self._db.query(
                "SELECT * FROM events WHERE kind = ? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            )
        else:
            rows = await self._db.query(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        return [self._to_event(r) for r in rows]

    async def count(self) -> int:
        row = await self._db.query_one("SELECT COUNT(*) AS n FROM events")
        assert row is not None  # COUNT always returns one row
        return int(row["n"])

    # -- mapping --------------------------------------------------------------

    @staticmethod
    def _to_event(row: aiosqlite.Row) -> Event:
        """Row -> validated Event model. Validation on read is deliberate:
        the model is the contract, even against our own disk."""
        data: dict[str, Any] = {
            "id": row["id"],
            "ts": row["ts"],
            "kind": row["kind"],
            "source": row["source"],
            "session_id": row["session_id"],
            "job_id": row["job_id"],
            "trace_id": row["trace_id"],
            "payload": json.loads(row["payload"]),
        }
        return Event.model_validate(data)