# =============================================================================
# src/jarvis/core/memory/profile.py - the always-present page about the owner
# =============================================================================
#
# One short markdown document describing who the owner is, what he is
# working on, and how he likes things done. Injected into EVERY
# conversation, which is what lets JARVIS answer in the context of the
# owner's world without being told each time.
#
# Two constraints shape it:
#
#   SHORT - it costs tokens on every single turn, so it is capped. A
#     profile that grows without limit is a tax on every conversation
#     forever. Facts that do not earn their place live in the vault and
#     arrive through retrieval instead.
#
#   STABLE - it sits early in the system prompt where the provider caches
#     it. Rewriting it constantly would throw that cache away, which is
#    why only the sleep cycle rewrites it, once a night at most.
#
# Versioned by append: every rewrite is a new row. The archivist will be
# writing this automatically, and an automated writer that can silently
# destroy the previous version is an automated writer you cannot trust.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import aiosqlite

from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.log import get_logger
from jarvis.core.db.database import Database

log = get_logger("core.memory.profile")

# Roughly 1-2k tokens. Enforced on write so a runaway generation cannot
# quietly triple the cost of every future turn.
MAX_PROFILE_CHARS = 6000


@dataclass(frozen=True)
class ProfileVersion:
    """One historical version of the profile document."""

    id: str
    content: str
    generated_by: str
    fact_count: int
    ts: datetime


class ProfileStore:
    """Read the live profile; append new versions."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def current(self) -> str:
        """The live profile text, or empty string if none exists yet.
        Empty is normal on a fresh system - the prompt assembler simply
        omits the section."""
        row = await self._db.query_one(
            "SELECT content FROM profile_doc ORDER BY id DESC LIMIT 1"
        )
        return str(row["content"]) if row else ""

    async def write(
        self,
        content: str,
        generated_by: str,
        fact_count: int = 0,
    ) -> ProfileVersion:
        """Append a new version. Over-long content is truncated rather
        than rejected: a slightly clipped profile is better than a failed
        sleep cycle, and the log line makes the clipping visible."""
        text = content.strip()
        if len(text) > MAX_PROFILE_CHARS:
            log.warning("profile truncated to cap", extra={
                "chars": len(text), "cap": MAX_PROFILE_CHARS,
            })
            text = text[:MAX_PROFILE_CHARS].rsplit("\n", 1)[0]

        version = ProfileVersion(
            id=new_ulid(),
            content=text,
            generated_by=generated_by,
            fact_count=fact_count,
            ts=utc_now(),
        )
        await self._db.execute(
            "INSERT INTO profile_doc (id, content, generated_by, fact_count, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                version.id, version.content, version.generated_by,
                version.fact_count, version.ts.isoformat(),
            ),
        )
        log.info("profile updated", extra={
            "profile_version": version.id,
            "generated_by": generated_by,
            "chars": len(text),
            "fact_count": fact_count,
        })
        return version

    async def history(self, limit: int = 10) -> list[ProfileVersion]:
        """Recent versions, newest first - for inspecting what an
        automated rewrite changed."""
        rows = await self._db.query(
            "SELECT * FROM profile_doc ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [self._to_version(r) for r in rows]

    @staticmethod
    def _to_version(row: aiosqlite.Row) -> ProfileVersion:
        return ProfileVersion(
            id=row["id"],
            content=row["content"],
            generated_by=row["generated_by"],
            fact_count=row["fact_count"],
            ts=datetime.fromisoformat(row["ts"]),
        )