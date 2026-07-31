# =============================================================================
# src/jarvis/core/db/repos/sessions.py - repository for sessions and turns
# =============================================================================
#
# The only code that touches the sessions and turns tables. Callers deal
# in Session and Turn models; SQL stays inside.
#
# Two behaviours to know about:
#   - get_or_create_default(client_kind): the Telegram pattern - one
#     long-lived session per client kind, created on first ever contact,
#     reused forever after (until a /new command archives it).
#   - append_turn() updates the session's last_active_ts in the SAME
#     transaction as the turn insert. Those two facts move together or
#     not at all - a crash between them cannot make them disagree.
# =============================================================================

from __future__ import annotations

import json

import aiosqlite

from jarvis.common.sessions import Session, SessionStatus, Turn
from jarvis.core.db.database import Database


class SessionsRepo:
    """Store and fetch conversation threads and their messages."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- sessions -------------------------------------------------------------

    async def create(self, session: Session) -> None:
        await self._db.execute(
            "INSERT INTO sessions "
            "(id, client_kind, title, status, rolling_summary, "
            " created_ts, last_active_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session.id,
                session.client_kind,
                session.title,
                session.status.value,
                session.rolling_summary,
                session.created_ts.isoformat(),
                session.last_active_ts.isoformat(),
            ),
        )

    async def get(self, session_id: str) -> Session | None:
        row = await self._db.query_one(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        return self._to_session(row) if row else None

    async def get_or_create_default(self, client_kind: str) -> Session:
        """The newest active session for this client kind, creating one if
        none exists. This is how Telegram gets its one long-lived thread."""
        row = await self._db.query_one(
            "SELECT * FROM sessions "
            "WHERE client_kind = ? AND status = ? "
            "ORDER BY id DESC LIMIT 1",
            (client_kind, SessionStatus.ACTIVE.value),
        )
        if row is not None:
            return self._to_session(row)
        session = Session(client_kind=client_kind)
        await self.create(session)
        return session

    async def archive(self, session_id: str) -> None:
        """Mark a session archived (the /new command path: archive the old
        thread, a fresh default gets created on next contact)."""
        await self._db.execute(
            "UPDATE sessions SET status = ? WHERE id = ?",
            (SessionStatus.ARCHIVED.value, session_id),
        )

    async def set_rolling_summary(self, session_id: str, summary: str) -> None:
        """Written by the summary refresher when that feature arrives; the
        repo method exists now so the seam is visible."""
        await self._db.execute(
            "UPDATE sessions SET rolling_summary = ? WHERE id = ?",
            (summary, session_id),
        )

    # -- turns ----------------------------------------------------------------

    async def append_turn(self, turn: Turn) -> None:
        """Insert a turn and bump the session's last-active time, atomically."""
        async with self._db.transaction() as tx:
            await tx.execute(
                "INSERT INTO turns "
                "(id, session_id, role, content, attachments, job_refs, "
                " llm_call_ids, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    turn.id,
                    turn.session_id,
                    turn.role.value,
                    turn.content,
                    json.dumps(turn.attachments),
                    json.dumps(turn.job_refs),
                    json.dumps(turn.llm_call_ids),
                    turn.ts.isoformat(),
                ),
            )
            await tx.execute(
                "UPDATE sessions SET last_active_ts = ? WHERE id = ?",
                (turn.ts.isoformat(), turn.session_id),
            )

    async def turns_since(
        self, after_turn_id: str | None, limit: int = 120
    ) -> list[Turn]:
        """Turns across ALL sessions after a given turn id, oldest first.

        ULIDs sort by time, so "after this id" is exactly "since this
        moment" - the high-water mark the sleep cycle carries between
        runs. A None marker (first ever cycle) reads the most recent
        `limit` turns instead of the oldest, so a fresh cycle on an old
        database starts from recent history rather than crawling forward
        from the beginning.
        """
        if after_turn_id is not None:
            rows = await self._db.query(
                "SELECT * FROM turns WHERE id > ? ORDER BY id ASC LIMIT ?",
                (after_turn_id, limit),
            )
            return [self._to_turn(r) for r in rows]

        rows = await self._db.query(
            "SELECT * FROM turns ORDER BY id DESC LIMIT ?", (limit,)
        )
        turns = [self._to_turn(r) for r in rows]
        turns.reverse()
        return turns

    async def recent_turns(self, session_id: str, limit: int = 20) -> list[Turn]:
        """The last N turns of a session, oldest first (ready to feed the
        model as conversation history)."""
        rows = await self._db.query(
            "SELECT * FROM turns WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        turns = [self._to_turn(r) for r in rows]
        turns.reverse()  # fetched newest-first for the LIMIT; serve oldest-first
        return turns

    # -- mapping --------------------------------------------------------------

    @staticmethod
    def _to_session(row: aiosqlite.Row) -> Session:
        return Session.model_validate({
            "id": row["id"],
            "client_kind": row["client_kind"],
            "title": row["title"],
            "status": row["status"],
            "rolling_summary": row["rolling_summary"],
            "created_ts": row["created_ts"],
            "last_active_ts": row["last_active_ts"],
        })

    @staticmethod
    def _to_turn(row: aiosqlite.Row) -> Turn:
        return Turn.model_validate({
            "id": row["id"],
            "session_id": row["session_id"],
            "role": row["role"],
            "content": row["content"],
            "attachments": json.loads(row["attachments"]),
            "job_refs": json.loads(row["job_refs"]),
            "llm_call_ids": json.loads(row["llm_call_ids"]),
            "ts": row["ts"],
        })