# =============================================================================
# src/jarvis/core/db/repos/facts.py - the memory vault
# =============================================================================
#
# The only code touching the facts tables. Handles the two halves of a
# fact - its row and its vector - together.
#
# Vectors are stored as packed float32 bytes: compact, exact, and
# readable back with one struct call. Comparison happens in Python
# (see the retrieval module next batch), so nothing here depends on a
# database extension.
#
# FTS5 query safety matters here: user and model text can contain quotes,
# hyphens, and words like AND or OR that FTS5 reads as SYNTAX. Passing
# raw text into a MATCH query would raise on perfectly ordinary input, so
# every query is rebuilt from its alphanumeric tokens.
# =============================================================================

from __future__ import annotations

import json
import re
import struct

import aiosqlite

from jarvis.common.facts import Fact, FactStatus
from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.core.db.database import Database

log = get_logger("core.db.facts")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def pack_vector(vector: list[float]) -> bytes:
    """Floats -> compact bytes for storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    """Bytes -> floats for comparison."""
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _fts_query(text: str) -> str:
    """Rebuild arbitrary text as a safe FTS5 OR-query.

    Tokens are quoted so FTS5 treats them as literals, and joined with OR
    so a partial match still returns candidates - ranking sorts out
    quality, and recall matters more than precision when a second search
    path (vectors) is fusing in alongside.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return " OR ".join(f'"{t}"' for t in tokens if len(t) > 1)


class FactsRepo:
    """Store, search, and age the owner's remembered facts."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # -- writes ---------------------------------------------------------------

    async def store(self, fact: Fact, embedding: list[float] | None = None) -> None:
        """Persist a new fact, optionally with its vector already computed."""
        await self._db.execute(
            "INSERT INTO facts "
            "(id, text, category, importance, confidence, status, supersedes, "
            " source_event_ids, created_ts, last_accessed_ts, access_count, "
            " embedder_version, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.id, fact.text, fact.category.value, fact.importance,
                fact.confidence, fact.status.value, fact.supersedes,
                json.dumps(fact.source_event_ids),
                fact.created_ts.isoformat(), fact.last_accessed_ts.isoformat(),
                fact.access_count, fact.embedder_version,
                pack_vector(embedding) if embedding else None,
            ),
        )

    async def set_embedding(
        self, fact_id: str, embedding: list[float], version: str
    ) -> None:
        """Attach or refresh a fact's vector."""
        await self._db.execute(
            "UPDATE facts SET embedding = ?, embedder_version = ? WHERE id = ?",
            (pack_vector(embedding), version, fact_id),
        )

    async def supersede(self, old_id: str, new_id: str) -> None:
        """Retire an old belief in favour of a newer one. The old row
        stays - provenance outlives correctness."""
        await self._db.execute(
            "UPDATE facts SET status = ? WHERE id = ?",
            (FactStatus.SUPERSEDED.value, old_id),
        )
        await self._db.execute(
            "UPDATE facts SET supersedes = ? WHERE id = ?", (old_id, new_id)
        )
        log.info("fact superseded", extra={"old_fact": old_id, "new_fact": new_id})

    async def expire(self, fact_id: str) -> None:
        """Fade a fact out through disuse. Not a deletion."""
        await self._db.execute(
            "UPDATE facts SET status = ? WHERE id = ?",
            (FactStatus.EXPIRED.value, fact_id),
        )

    async def set_confidence(self, fact_id: str, confidence: float) -> None:
        await self._db.execute(
            "UPDATE facts SET confidence = ? WHERE id = ?", (confidence, fact_id)
        )

    async def touch(self, fact_ids: list[str]) -> None:
        """Record that facts were actually used. Access recency and count
        feed retrieval ranking and protect useful facts from decay."""
        if not fact_ids:
            return
        now = utc_now().isoformat()
        placeholders = ",".join("?" for _ in fact_ids)
        await self._db.execute(
            f"UPDATE facts SET last_accessed_ts = ?, "
            f"access_count = access_count + 1 WHERE id IN ({placeholders})",
            (now, *fact_ids),
        )

    # -- reads ----------------------------------------------------------------

    async def get(self, fact_id: str) -> Fact | None:
        row = await self._db.query_one("SELECT * FROM facts WHERE id = ?", (fact_id,))
        return self._to_fact(row) if row else None

    async def search_keyword(self, query: str, limit: int = 20) -> list[tuple[Fact, float]]:
        """Keyword candidates with a 0-1 relevance score.

        FTS5's bm25() returns a rank where MORE NEGATIVE is better, which
        is awkward to fuse with similarity scores. It is mapped into 0-1
        (higher better) so both search paths speak the same language.
        """
        match = _fts_query(query)
        if not match:
            return []
        rows = await self._db.query(
            "SELECT f.*, bm25(facts_fts) AS rank FROM facts_fts "
            "JOIN facts f ON f.id = facts_fts.fact_id "
            "WHERE facts_fts MATCH ? AND f.status = 'active' "
            "ORDER BY rank LIMIT ?",
            (match, limit),
        )
        results: list[tuple[Fact, float]] = []
        for row in rows:
            rank = float(row["rank"])          # negative; nearer zero is worse
            score = min(1.0, -rank / 10.0) if rank < 0 else 0.0
            results.append((self._to_fact(row), score))
        return results

    async def active_with_vectors(self) -> list[tuple[Fact, list[float]]]:
        """Every active fact that has an embedding, with its vector.

        Loaded in full for brute-force comparison. Fine at personal scale
        (thousands of facts = milliseconds); the day it is not, this one
        method is what gets replaced by an index.
        """
        rows = await self._db.query(
            "SELECT * FROM facts WHERE status = 'active' AND embedding IS NOT NULL"
        )
        return [(self._to_fact(r), unpack_vector(r["embedding"])) for r in rows]

    async def needing_embedding(self, current_version: str, limit: int = 200) -> list[Fact]:
        """Active facts with no vector, or one from an older model."""
        rows = await self._db.query(
            "SELECT * FROM facts WHERE status = 'active' "
            "AND (embedding IS NULL OR embedder_version IS NOT ?) LIMIT ?",
            (current_version, limit),
        )
        return [self._to_fact(r) for r in rows]

    async def all_active(self, limit: int = 500) -> list[Fact]:
        """Active facts, most important first - the sleep cycle's working
        set and the profile document's raw material."""
        rows = await self._db.query(
            "SELECT * FROM facts WHERE status = 'active' "
            "ORDER BY importance DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [self._to_fact(r) for r in rows]

    async def count_active(self) -> int:
        row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM facts WHERE status = 'active'"
        )
        return int(row["n"]) if row else 0

    # -- mapping --------------------------------------------------------------

    @staticmethod
    def _to_fact(row: aiosqlite.Row) -> Fact:
        return Fact.model_validate({
            "id": row["id"],
            "text": row["text"],
            "category": row["category"],
            "importance": row["importance"],
            "confidence": row["confidence"],
            "status": row["status"],
            "supersedes": row["supersedes"],
            "source_event_ids": json.loads(row["source_event_ids"]),
            "created_ts": row["created_ts"],
            "last_accessed_ts": row["last_accessed_ts"],
            "access_count": row["access_count"],
            "embedder_version": row["embedder_version"],
        })