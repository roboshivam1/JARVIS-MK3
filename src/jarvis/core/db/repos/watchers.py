# =============================================================================
# src/jarvis/core/db/repos/watchers.py - the watchers table
# =============================================================================
#
# record_check() writes state back after EVERY check, hit or not. That
# is what makes the next comparison possible - a watcher that only saved
# state when something changed would keep re-reporting the same change.
# =============================================================================

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.common.watchers import Watcher, WatcherKind
from jarvis.core.db.database import Database

log = get_logger("core.db.watchers")


class WatchersRepo:
    """Create, list, and update watchers."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(self, watcher: Watcher) -> bool:
        """Add a watcher. False if the name is taken."""
        try:
            await self._db.execute(
                "INSERT INTO watchers "
                "(id, name, kind, config, state, priority, note, enabled, "
                " hit_count, last_checked_ts, last_hit_ts, created_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    watcher.id, watcher.name, watcher.kind.value,
                    json.dumps(watcher.config), json.dumps(watcher.state),
                    watcher.priority, watcher.note,
                    1 if watcher.enabled else 0, watcher.hit_count,
                    None, None, watcher.created_ts.isoformat(),
                ),
            )
            log.info("watcher created", extra={
                "watcher": watcher.name, "kind": watcher.kind.value,
            })
            return True
        except aiosqlite.IntegrityError:
            return False

    async def enabled(self) -> list[Watcher]:
        rows = await self._db.query(
            "SELECT * FROM watchers WHERE enabled = 1 ORDER BY id ASC"
        )
        return [self._to_watcher(r) for r in rows]

    async def all(self) -> list[Watcher]:
        rows = await self._db.query("SELECT * FROM watchers ORDER BY name ASC")
        return [self._to_watcher(r) for r in rows]

    async def record_check(
        self, watcher_id: str, state: dict[str, Any], hit: bool
    ) -> None:
        """Save what this check saw. Written whether or not it fired -
        state that only updates on a hit would make the same change
        report forever."""
        now = utc_now().isoformat()
        if hit:
            await self._db.execute(
                "UPDATE watchers SET state = ?, last_checked_ts = ?, "
                "last_hit_ts = ?, hit_count = hit_count + 1 WHERE id = ?",
                (json.dumps(state), now, now, watcher_id),
            )
        else:
            await self._db.execute(
                "UPDATE watchers SET state = ?, last_checked_ts = ? WHERE id = ?",
                (json.dumps(state), now, watcher_id),
            )

    async def set_enabled(self, name: str, enabled: bool) -> bool:
        changed = await self._db.execute_returning_changes(
            "UPDATE watchers SET enabled = ? WHERE name = ?",
            (1 if enabled else 0, name),
        )
        return changed == 1

    async def delete(self, name: str) -> bool:
        changed = await self._db.execute_returning_changes(
            "DELETE FROM watchers WHERE name = ?", (name,)
        )
        return changed == 1

    @staticmethod
    def _to_watcher(row: aiosqlite.Row) -> Watcher:
        return Watcher.model_validate({
            "id": row["id"],
            "name": row["name"],
            "kind": WatcherKind(row["kind"]),
            "config": json.loads(row["config"]),
            "state": json.loads(row["state"]),
            "priority": row["priority"],
            "note": row["note"],
            "enabled": bool(row["enabled"]),
            "hit_count": row["hit_count"],
            "last_checked_ts": row["last_checked_ts"],
            "last_hit_ts": row["last_hit_ts"],
            "created_ts": row["created_ts"],
        })
