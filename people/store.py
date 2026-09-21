"""people/store.py — connection + read side of the people database.

Everything here is read-only apart from connect(). Writes happen in
review.py, and only through approved proposals.
"""
from __future__ import annotations

import re
from pathlib import Path

import aiosqlite

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class PeopleStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db = await aiosqlite.connect(self.db_path)
        self.db.row_factory = aiosqlite.Row
        # foreign_keys is per-connection in SQLite, so it's set here, not in schema.sql
        await self.db.execute("PRAGMA foreign_keys = ON")
        await self.db.execute("PRAGMA journal_mode = WAL")
        await self.db.executescript(SCHEMA_PATH.read_text())
        await self.db.commit()

    async def close(self) -> None:
        if self.db:
            await self.db.close()

    # ── query helpers ─────────────────────────────────────────────────────
    async def rows(self, sql: str, *args) -> list[dict]:
        async with self.db.execute(sql, args) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def one(self, sql: str, *args) -> dict | None:
        rows = await self.rows(sql, *args)
        return rows[0] if rows else None

    async def aliases_of(self, person_id: int) -> list[str]:
        rows = await self.rows("SELECT alias FROM aliases WHERE person_id = ?", person_id)
        return [r["alias"] for r in rows]

    # ── lookup ────────────────────────────────────────────────────────────
    async def find_people(self, name: str) -> list[dict]:
        """Everyone matching a name or alias, case-insensitive. May be >1."""
        return await self.rows(
            """SELECT DISTINCT p.* FROM people p
               LEFT JOIN aliases a ON a.person_id = p.id
               WHERE p.name = ? COLLATE NOCASE OR a.alias = ?""",
            name, name,
        )

    async def mentioned_in(self, text: str) -> list[int]:
        """IDs of people whose name, first name, or alias appears in text."""
        labels = await self.rows(
            "SELECT id AS pid, name AS label FROM people "
            "UNION SELECT person_id, alias FROM aliases"
        )
        found: set[int] = set()
        for row in labels:
            candidates = {row["label"]}
            first = row["label"].split()[0]
            if len(first) > 2:
                candidates.add(first)
            for c in candidates:
                if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE):
                    found.add(row["pid"])
        return sorted(found)

    # ── profiles & roster ─────────────────────────────────────────────────
    async def profile(self, person_id: int, include_sensitive: bool = True) -> dict | None:
        person = await self.one("SELECT * FROM people WHERE id = ?", person_id)
        if not person:
            return None
        sens = "" if include_sensitive else "AND sensitivity = 'normal'"
        person["aliases"] = await self.aliases_of(person_id)
        person["facts"] = await self.rows(
            f"""SELECT id, category, text, sensitivity, learned_at FROM facts
                WHERE person_id = ? AND status = 'active' {sens}
                ORDER BY category, learned_at""", person_id)
        person["commitments"] = await self.rows(
            """SELECT id, direction, what, due_date, created_at FROM commitments
               WHERE person_id = ? AND status = 'open' ORDER BY created_at""", person_id)
        person["key_dates"] = await self.rows(
            "SELECT id, label, date, recurring FROM key_dates WHERE person_id = ?", person_id)
        return person

    async def roster(self, limit: int = 20) -> list[dict]:
        """Closest and most recently mentioned people first."""
        people = await self.rows(
            """SELECT id, name, relationship, summary FROM people
               WHERE status = 'active'
               ORDER BY closeness, last_mentioned DESC NULLS LAST, name
               LIMIT ?""", limit)
        for p in people:
            p["aliases"] = await self.aliases_of(p["id"])
        return people
