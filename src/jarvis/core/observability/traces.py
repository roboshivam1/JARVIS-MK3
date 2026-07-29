# =============================================================================
# src/jarvis/core/observability/traces.py - writing and querying llm_calls
# =============================================================================
#
# Two halves:
#
#   make_db_trace_sink(db) - manufactures the async function that LLMLayer
#     calls after every model call. This replaces the logging-only sink at
#     wiring time; the layer itself never changes. One row per call, no
#     exceptions - failed calls included (error column set, zero tokens).
#
#   TracesRepo - the question side: today's spend, one trace's calls,
#     daily rollups. "What did you cost me this week?" is answered from
#     here, and the later budget guard reads the same numbers.
#
# Resilience rule: the sink must NEVER propagate its own failure into the
# model call it is recording. A successful call with a lost trace row is
# an accounting gap to fix; a successful call crashed by its own
# bookkeeping is self-harm. The sink logs write failures loudly and
# swallows them.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import aiosqlite

from jarvis.common.log import get_logger
from jarvis.core.db.database import Database
from jarvis.llm.layer import LLMCallRecord, TraceSink

log = get_logger("core.traces")


def make_db_trace_sink(db: Database) -> TraceSink:
    """Build the database-backed trace sink for LLMLayer."""

    async def sink(record: LLMCallRecord) -> None:
        try:
            await db.execute(
                "INSERT INTO llm_calls "
                "(id, ts, trace_id, actor, tier, model, latency_ms, "
                " tokens_in, tokens_out, cached_tokens, cost_usd, cost_inr, "
                " stop_reason, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.ts,
                    record.trace_id,
                    record.actor,
                    record.tier,
                    record.model,
                    record.latency_ms,
                    record.tokens_in,
                    record.tokens_out,
                    record.cached_tokens,
                    record.cost_usd,
                    record.cost_inr,
                    record.stop_reason,
                    record.error,
                ),
            )
        except Exception:
            # Never let bookkeeping crash the call it records.
            log.error(
                "failed to write llm call trace - accounting gap",
                exc_info=True,
                extra={"llm_call_id": record.id, "trace_id": record.trace_id},
            )

    return sink


@dataclass(frozen=True)
class DayCost:
    """One day's rolled-up spend."""

    day: str          # YYYY-MM-DD (UTC)
    calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cost_inr: float


class TracesRepo:
    """Questions asked of the money ledger."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def calls_for_trace(self, trace_id: str) -> list[aiosqlite.Row]:
        """Every model call in one causal chain, in time order. Returned as
        raw rows for now - the status page renders them directly; a typed
        model can arrive when a second consumer needs one."""
        return await self._db.query(
            "SELECT * FROM llm_calls WHERE trace_id = ? ORDER BY id ASC",
            (trace_id,),
        )

    async def cost_today_usd(self) -> float:
        """Total spend since UTC midnight - the number the budget guard
        will watch."""
        midnight = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        row = await self._db.query_one(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total "
            "FROM llm_calls WHERE ts >= ?",
            (midnight.isoformat(),),
        )
        assert row is not None
        return float(row["total"])

    async def daily_costs(self, days: int = 7) -> list[DayCost]:
        """Per-day rollup for the last N days, newest first. The direct
        answer to 'what did you cost me this week?'."""
        since = (
            datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            - timedelta(days=days - 1)
        )
        rows = await self._db.query(
            "SELECT substr(ts, 1, 10) AS day, "
            "       COUNT(*)          AS calls, "
            "       SUM(tokens_in)    AS tokens_in, "
            "       SUM(tokens_out)   AS tokens_out, "
            "       SUM(cost_usd)     AS cost_usd, "
            "       SUM(cost_inr)     AS cost_inr "
            "FROM llm_calls WHERE ts >= ? "
            "GROUP BY day ORDER BY day DESC",
            (since.isoformat(),),
        )
        return [
            DayCost(
                day=r["day"],
                calls=int(r["calls"]),
                tokens_in=int(r["tokens_in"]),
                tokens_out=int(r["tokens_out"]),
                cost_usd=round(float(r["cost_usd"]), 6),
                cost_inr=round(float(r["cost_inr"]), 4),
            )
            for r in rows
        ]