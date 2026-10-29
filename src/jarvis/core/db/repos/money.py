# =============================================================================
# src/jarvis/core/db/repos/money.py - the ledger's storage
# =============================================================================

from __future__ import annotations

import re
from typing import Any

from jarvis.common.ids import new_ulid, utc_now
from jarvis.core.db.database import Database

SCOPES = ("personal", "basement_supply")


class MoneyError(Exception):
    """Shown to the owner as-is."""


def payee_key(payee: str) -> str:
    """What rules match on: lower case, single spaces, no trailing dots."""
    return re.sub(r"\s+", " ", payee.strip().lower()).rstrip(".")


class MoneyRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._db.query(sql, params)]

    # -- categories and rules ---------------------------------------------------

    async def categories(self) -> list[dict[str, Any]]:
        rows = await self._rows("SELECT name, is_spend FROM money_categories ORDER BY sort, name")
        for r in rows:
            r["is_spend"] = bool(r["is_spend"])
        return rows

    async def check_category(self, name: str) -> str:
        clean = name.strip().lower()
        if not await self._rows("SELECT 1 FROM money_categories WHERE name = ?", (clean,)):
            names = ", ".join(c["name"] for c in await self.categories())
            raise MoneyError(f"No category {name!r}. Categories: {names}.")
        return clean

    async def add_category(self, name: str, is_spend: bool = True) -> str:
        clean = name.strip().lower()
        if not 2 <= len(clean) <= 40:
            raise MoneyError("A category name is 2 to 40 characters.")
        await self._db.execute(
            "INSERT OR IGNORE INTO money_categories (name, sort, is_spend) VALUES (?, 150, ?)",
            (clean, int(is_spend)))
        return clean

    async def rule(self, key: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM money_rules WHERE payee_key = ?", (key,))
        return rows[0] if rows else None

    async def rules(self) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT r.*, (SELECT COUNT(*) FROM money_transactions t "
            " WHERE t.payee_key = r.payee_key) AS uses "
            "FROM money_rules r ORDER BY r.payee_key")

    async def set_rule(self, key: str, category: str, *, scope: str = "personal",
                       label: str | None = None, project_id: str | None = None) -> int:
        """Store the rule and apply it to every payment to this payee that
        the owner has not categorised by hand. Returns how many changed."""
        if scope not in SCOPES:
            raise MoneyError(f"Scope is one of {', '.join(SCOPES)}.")
        await self._db.execute(
            "INSERT INTO money_rules (payee_key, label, category, scope, project_id, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (payee_key) DO UPDATE SET "
            "label = COALESCE(excluded.label, money_rules.label), category = excluded.category, "
            "scope = excluded.scope, project_id = excluded.project_id",
            (key, label, category, scope, project_id, utc_now().isoformat()))
        return await self._db.execute_returning_changes(
            "UPDATE money_transactions SET category = ?, scope = ?, project_id = ?, "
            "categorized_by = 'rule' WHERE payee_key = ? "
            "AND (categorized_by IS NULL OR categorized_by = 'rule')",
            (category, scope, project_id, key))

    async def delete_rule(self, key: str) -> None:
        await self._db.execute("DELETE FROM money_rules WHERE payee_key = ?", (key,))

    # -- transactions -------------------------------------------------------------

    async def insert(self, data: dict[str, Any]) -> str | None:
        """A new transaction. None when its bank reference is already known -
        the same payment described twice."""
        if data.get("ref") and await self._rows(
                "SELECT 1 FROM money_transactions WHERE ref = ?", (data["ref"],)):
            return None
        row_id = new_ulid()
        await self._db.execute(
            "INSERT INTO money_transactions (id, ref, occurred_at, date, amount_paise, "
            "payee_raw, payee_key, category, categorized_by, scope, project_id, channel, "
            "status, account_last4, source, note, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row_id, data.get("ref"), data["occurred_at"], data["date"], data["amount_paise"],
             data["payee_raw"], data["payee_key"], data.get("category"),
             data.get("categorized_by"), data.get("scope", "personal"), data.get("project_id"),
             data.get("channel"), data.get("status", "success"), data.get("account_last4"),
             data["source"], data.get("note"), utc_now().isoformat()))
        return row_id

    async def get(self, txn_id: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM money_transactions WHERE id = ?", (txn_id,))
        return rows[0] if rows else None

    async def find_short(self, short: str) -> dict[str, Any] | None:
        """A transaction by the last characters of its id, as PLUTUS shows them."""
        rows = await self._rows(
            "SELECT * FROM money_transactions WHERE id LIKE ? ORDER BY id DESC",
            (f"%{short.upper()}",))
        return rows[0] if len(rows) == 1 else None

    async def update(self, txn_id: str, **fields: Any) -> None:
        allowed = {"category", "scope", "project_id", "note", "categorized_by"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        if "scope" in sets and sets["scope"] not in SCOPES:
            raise MoneyError(f"Scope is one of {', '.join(SCOPES)}.")
        assignments = ", ".join(f"{k} = ?" for k in sets)
        await self._db.execute(
            f"UPDATE money_transactions SET {assignments} WHERE id = ?", (*sets.values(), txn_id))

    async def delete(self, txn_id: str) -> None:
        await self._db.execute("DELETE FROM money_transactions WHERE id = ?", (txn_id,))

    async def between(self, start: str, end: str) -> list[dict[str, Any]]:
        """Every transaction from start to end (dates, inclusive), newest first."""
        return await self._rows(
            "SELECT t.*, COALESCE(r.label, t.payee_raw) AS payee, p.name AS project, "
            "COALESCE(c.is_spend, 1) AS is_spend FROM money_transactions t "
            "LEFT JOIN money_rules r ON r.payee_key = t.payee_key "
            "LEFT JOIN projects p ON p.id = t.project_id "
            "LEFT JOIN money_categories c ON c.name = t.category "
            "WHERE t.date BETWEEN ? AND ? ORDER BY t.occurred_at DESC", (start, end))

    async def to_sort(self) -> list[dict[str, Any]]:
        """Payees with uncategorised successful payments, biggest first."""
        return await self._rows(
            "SELECT payee_key, MIN(payee_raw) AS payee, COUNT(*) AS n, "
            "SUM(amount_paise) AS paise, MAX(date) AS last FROM money_transactions "
            "WHERE category IS NULL AND status = 'success' GROUP BY payee_key "
            "ORDER BY paise DESC")

    # -- the mailbox reader's state ---------------------------------------------

    async def state(self, key: str) -> str | None:
        rows = await self._rows("SELECT value FROM money_state WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    async def set_state(self, key: str, value: str) -> None:
        await self._db.execute(
            "INSERT INTO money_state (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))
