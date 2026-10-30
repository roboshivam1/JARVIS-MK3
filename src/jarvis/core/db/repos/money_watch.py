# =============================================================================
# src/jarvis/core/db/repos/money_watch.py - the watchdog's storage
# =============================================================================

from __future__ import annotations

from typing import Any

from jarvis.common.ids import new_ulid, utc_now
from jarvis.core.db.database import Database


class MoneyWatchRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._db.query(sql, params)]

    # -- payments needing a look --------------------------------------------------

    async def flag(self, kind: str, txn_id: str, reason: str) -> bool:
        """Open a watch item. False if this payment was already flagged so."""
        if await self._rows("SELECT 1 FROM money_watch WHERE txn_id = ? AND kind = ?",
                            (txn_id, kind)):
            return False
        await self._db.execute(
            "INSERT INTO money_watch (id, kind, txn_id, reason, status, created_ts) "
            "VALUES (?, ?, ?, ?, 'open', ?)",
            (new_ulid(), kind, txn_id, reason, utc_now().isoformat()))
        return True

    async def open_items(self) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT w.*, t.amount_paise, t.payee_raw, t.occurred_at, t.date "
            "FROM money_watch w JOIN money_transactions t ON t.id = w.txn_id "
            "WHERE w.status = 'open' ORDER BY w.created_ts")

    async def find(self, item: str) -> dict[str, Any] | None:
        """A watch item by id or the last characters of it."""
        rows = await self._rows(
            "SELECT * FROM money_watch WHERE id = ? OR id LIKE ?", (item, f"%{item.upper()}"))
        return rows[0] if len(rows) == 1 else None

    async def clear(self, item_id: str, note: str | None) -> None:
        await self._db.execute(
            "UPDATE money_watch SET status = 'cleared', note = ?, cleared_ts = ? WHERE id = ?",
            (note, utc_now().isoformat(), item_id))

    # -- subscriptions --------------------------------------------------------------

    async def subscriptions(self, include_dismissed: bool = False) -> list[dict[str, Any]]:
        where = "" if include_dismissed else "WHERE s.status = 'active'"
        return await self._rows(
            "SELECT s.*, COALESCE(r.label, (SELECT payee_raw FROM money_transactions t "
            " WHERE t.payee_key = s.payee_key ORDER BY t.occurred_at DESC LIMIT 1), "
            " s.payee_key) AS payee FROM money_subscriptions s "
            f"LEFT JOIN money_rules r ON r.payee_key = s.payee_key {where} "
            "ORDER BY s.next_due")

    async def subscription(self, key: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM money_subscriptions WHERE payee_key = ?", (key,))
        return rows[0] if rows else None

    async def save_subscription(self, key: str, *, cadence: str, typical: int, last: int,
                                last_date: str, next_due: str, source: str,
                                status: str = "active") -> None:
        await self._db.execute(
            "INSERT INTO money_subscriptions (payee_key, cadence, typical_paise, last_paise, "
            "last_date, next_due, status, source, updated_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (payee_key) DO UPDATE SET cadence = excluded.cadence, "
            "typical_paise = excluded.typical_paise, last_paise = excluded.last_paise, "
            "last_date = excluded.last_date, next_due = excluded.next_due, "
            "status = excluded.status, source = excluded.source, updated_ts = excluded.updated_ts",
            (key, cadence, typical, last, last_date, next_due, status, source,
             utc_now().isoformat()))

    async def set_subscription_status(self, key: str, status: str) -> int:
        return await self._db.execute_returning_changes(
            "UPDATE money_subscriptions SET status = ?, updated_ts = ? WHERE payee_key = ?",
            (status, utc_now().isoformat(), key))

    async def mark(self, key: str, field: str, value: str) -> None:
        if field not in ("warned_for", "missed_for"):
            raise ValueError(field)
        await self._db.execute(
            f"UPDATE money_subscriptions SET {field} = ? WHERE payee_key = ?", (value, key))

    async def payments_of(self, key: str, since: str) -> list[dict[str, Any]]:
        """Successful payments to one payee since a date, oldest first."""
        return await self._rows(
            "SELECT date, amount_paise FROM money_transactions WHERE payee_key = ? "
            "AND status = 'success' AND date >= ? ORDER BY occurred_at", (key, since))

    async def recurring_candidates(self, since: str) -> list[str]:
        """Payees paid at least twice since a date - all a cycle needs."""
        rows = await self._rows(
            "SELECT t.payee_key FROM money_transactions t "
            "LEFT JOIN money_categories c ON c.name = t.category "
            "WHERE t.status = 'success' AND t.date >= ? AND COALESCE(c.is_spend, 1) = 1 "
            "GROUP BY t.payee_key HAVING COUNT(*) >= 2", (since,))
        return [r["payee_key"] for r in rows]

    async def amounts_of_payee(self, key: str, before_id: str) -> list[int]:
        """Earlier successful amounts to one payee, for 'is this unusual'."""
        rows = await self._rows(
            "SELECT amount_paise FROM money_transactions WHERE payee_key = ? "
            "AND status = 'success' AND id != ?", (key, before_id))
        return [int(r["amount_paise"]) for r in rows]

    async def recent_amounts(self, since: str, exclude_id: str) -> list[int]:
        rows = await self._rows(
            "SELECT amount_paise FROM money_transactions WHERE status = 'success' "
            "AND date >= ? AND id != ?", (since, exclude_id))
        return [int(r["amount_paise"]) for r in rows]
