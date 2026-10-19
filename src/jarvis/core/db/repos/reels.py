# =============================================================================
# src/jarvis/core/db/repos/reels.py - the reel episode log
# =============================================================================
#
# The only code touching reel_episodes. Status changes go through
# set_status() with the statuses the caller expects the row to be in, the
# same optimistic check the job queue uses: a Telegram tap and the
# service loop racing over one episode resolve to exactly one winner.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

import aiosqlite

from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.common.reels import (
    ACTIVE_STATUSES,
    EpisodeStatus,
    ReelBrief,
    ReelEpisode,
)
from jarvis.core.db.database import Database

log = get_logger("core.db.reels")

# Columns update() may touch. Status is NOT here: it moves only through
# set_status(), where the optimistic check lives.
_UPDATABLE = frozenset({
    "job_id", "title", "lines", "round", "review_sent_round", "approval_id",
    "session_id", "sheet_artifact_id", "voice_artifact_id", "mp4_artifact_id",
    "notes", "error", "verification", "approved_ts", "delivered_ts",
    "posted_ts",
})


class ReelsRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    # -- writes ---------------------------------------------------------------

    async def create(self, episode: ReelEpisode) -> None:
        await self._db.execute(
            "INSERT INTO reel_episodes (id, kit_id, job_id, type, title, request, "
            "brief, lines, status, round, review_sent_round, created_ts, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode.id, episode.kit_id, episode.job_id, episode.type.value,
                episode.title, episode.request, episode.brief.model_dump_json(),
                json.dumps(episode.lines, ensure_ascii=False),
                episode.status.value, episode.round, episode.review_sent_round,
                episode.created_ts.isoformat(), episode.updated_ts.isoformat(),
            ),
        )

    async def update(self, episode_id: str, **fields: Any) -> None:
        """Set plain columns. Lists become JSON, datetimes ISO strings."""
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"not updatable: {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [_to_column(value) for value in fields.values()]
        await self._db.execute(
            f"UPDATE reel_episodes SET {assignments}, updated_ts = ? WHERE id = ?",
            (*values, utc_now().isoformat(), episode_id),
        )

    async def set_status(
        self,
        episode_id: str,
        status: EpisodeStatus,
        expected: Iterable[EpisodeStatus] | None = None,
        **fields: Any,
    ) -> bool:
        """Move an episode, optionally only from the given statuses, and
        set any other columns in the same statement. Returns whether it
        moved: False means someone else got there first."""
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"not updatable: {sorted(unknown)}")
        assignments = ["status = ?", "updated_ts = ?"]
        values: list[Any] = [status.value, utc_now().isoformat()]
        for name, value in fields.items():
            assignments.append(f"{name} = ?")
            values.append(_to_column(value))
        sql = f"UPDATE reel_episodes SET {', '.join(assignments)} WHERE id = ?"
        values.append(episode_id)
        if expected is not None:
            allowed = [s.value for s in expected]
            sql += f" AND status IN ({', '.join('?' * len(allowed))})"
            values.extend(allowed)
        changed = await self._db.execute_returning_changes(sql, values)
        return changed > 0

    # -- reads ----------------------------------------------------------------

    async def get(self, episode_id: str) -> ReelEpisode | None:
        row = await self._db.query_one(
            "SELECT * FROM reel_episodes WHERE id = ?", (episode_id,)
        )
        return self._to_episode(row) if row else None

    async def for_job(self, job_id: str) -> ReelEpisode | None:
        row = await self._db.query_one(
            "SELECT * FROM reel_episodes WHERE job_id = ?", (job_id,)
        )
        return self._to_episode(row) if row else None

    async def by_ref(self, ref: str) -> list[ReelEpisode]:
        """Episodes matching what the owner typed: the six-character ref,
        any longer suffix of the id, or the kit id. More than one match
        means the ref was ambiguous and the caller should say so."""
        ref = ref.strip().lower()
        if len(ref) < 4:
            return []
        rows = await self._db.query(
            "SELECT * FROM reel_episodes WHERE lower(id) LIKE ? OR kit_id = ? "
            "ORDER BY created_ts DESC",
            (f"%{ref}", ref),
        )
        return [self._to_episode(r) for r in rows]

    async def recent(self, limit: int = 10) -> list[ReelEpisode]:
        rows = await self._db.query(
            "SELECT * FROM reel_episodes ORDER BY created_ts DESC LIMIT ?", (limit,)
        )
        return [self._to_episode(r) for r in rows]

    async def active(self) -> list[ReelEpisode]:
        """Episodes holding the kit folder right now."""
        statuses = [s.value for s in ACTIVE_STATUSES]
        rows = await self._db.query(
            f"SELECT * FROM reel_episodes WHERE status IN "
            f"({', '.join('?' * len(statuses))}) ORDER BY created_ts",
            statuses,
        )
        return [self._to_episode(r) for r in rows]

    async def needing_attention(self) -> list[ReelEpisode]:
        """What the service loop watches: anything with a job still in
        motion. Delivered, posted, cancelled and failed rows are settled."""
        return await self.active()

    # -- mapping --------------------------------------------------------------

    @staticmethod
    def _to_episode(row: aiosqlite.Row) -> ReelEpisode:
        data = dict(row)
        data["brief"] = ReelBrief.model_validate_json(data["brief"])
        data["lines"] = json.loads(data["lines"] or "[]")
        for name in ("created_ts", "updated_ts", "approved_ts",
                     "delivered_ts", "posted_ts"):
            if data.get(name):
                data[name] = datetime.fromisoformat(data[name])
        return ReelEpisode.model_validate(data)


def _to_column(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value
