"""people/review.py — the gate between the LLM and the real tables.

Debrief → proposals (staged) → approve / reject → real rows.
Approving a proposal about a new person auto-approves that person first.
Rejecting a new person auto-rejects everything that depended on them.
"""
from __future__ import annotations

import json

from .models import ExtractionResult, PAYLOAD_MODELS, RESULT_FIELDS
from .store import PeopleStore


class ReviewError(Exception):
    """A proposal can't be approved as-is — message is safe to show in the UI."""


class ReviewQueue:
    def __init__(self, store: PeopleStore):
        self.store = store

    @property
    def db(self):
        return self.store.db

    # ── staging ───────────────────────────────────────────────────────────
    async def stage(self, raw_text: str, result: ExtractionResult) -> int:
        cur = await self.db.execute("INSERT INTO debriefs (raw_text) VALUES (?)", (raw_text,))
        debrief_id = cur.lastrowid
        for kind, field in RESULT_FIELDS.items():
            for item in getattr(result, field):
                await self.db.execute(
                    "INSERT INTO proposals (debrief_id, kind, payload) VALUES (?, ?, ?)",
                    (debrief_id, kind, item.model_dump_json()))
        await self._close_if_done(debrief_id)
        await self.db.commit()
        return debrief_id

    async def pending(self, debrief_id: int | None = None) -> list[dict]:
        sql, args = "SELECT * FROM proposals WHERE status = 'pending'", []
        if debrief_id is not None:
            sql, args = sql + " AND debrief_id = ?", [debrief_id]
        rows = await self.store.rows(sql + " ORDER BY debrief_id, id", *args)
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    # ── decisions ─────────────────────────────────────────────────────────
    async def approve(self, proposal_id: int, edited: dict | None = None) -> int:
        """Commit a proposal. `edited` replaces the payload (frontend edits)."""
        try:
            row_id = await self._approve(proposal_id, edited)
            await self.db.commit()
            return row_id
        except Exception:
            await self.db.rollback()
            raise

    async def reject(self, proposal_id: int) -> list[int]:
        """Reject a proposal (and dependents, if it's a new person). Returns rejected ids."""
        prop = await self._pending_proposal(proposal_id)
        ids = [proposal_id]
        if prop["kind"] == "new_person":
            ref = f"new:{json.loads(prop['payload'])['tag']}"
            for row in await self.store.rows(
                "SELECT id, payload FROM proposals WHERE debrief_id = ? "
                "AND status = 'pending' AND kind != 'new_person'", prop["debrief_id"]):
                if json.loads(row["payload"]).get("person_ref") == ref:
                    ids.append(row["id"])
        await self.db.executemany(
            "UPDATE proposals SET status = 'rejected', decided_at = datetime('now') WHERE id = ?",
            [(i,) for i in ids])
        await self._close_if_done(prop["debrief_id"])
        await self.db.commit()
        return ids

    # ── internals ─────────────────────────────────────────────────────────
    async def _approve(self, proposal_id: int, edited: dict | None) -> int:
        prop = await self._pending_proposal(proposal_id)
        kind, debrief_id = prop["kind"], prop["debrief_id"]
        original = json.loads(prop["payload"])
        data = PAYLOAD_MODELS[kind].model_validate(edited if edited is not None else original)

        if kind == "new_person":
            if data.tag != original["tag"]:
                raise ReviewError("A new person's tag can't be edited — other proposals point at it.")
            row_id = await self._insert(
                "INSERT INTO people (name, relationship, closeness, how_met, summary, last_mentioned) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                data.name, data.relationship, data.closeness, data.how_met, data.summary)
        else:
            pid = await self._resolve(debrief_id, data.person_ref)
            if kind == "alias":
                await self.db.execute(
                    "INSERT OR IGNORE INTO aliases (person_id, alias) VALUES (?, ?)", (pid, data.alias))
                row_id = pid
            elif kind == "fact":
                row_id = await self._insert(
                    "INSERT INTO facts (person_id, category, text, sensitivity, debrief_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    pid, data.category, data.text, data.sensitivity, debrief_id)
                if data.supersedes_fact_id:
                    await self._supersede(data.supersedes_fact_id, pid, row_id)
            elif kind == "commitment":
                row_id = await self._insert(
                    "INSERT INTO commitments (person_id, direction, what, due_date, debrief_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    pid, data.direction, data.what, data.due_date, debrief_id)
            else:  # key_date
                row_id = await self._insert(
                    "INSERT INTO key_dates (person_id, label, date, recurring, debrief_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    pid, data.label, data.date, int(data.recurring), debrief_id)
            await self.db.execute(
                "UPDATE people SET last_mentioned = datetime('now'), updated_at = datetime('now') "
                "WHERE id = ?", (pid,))

        # Store the payload that was actually approved (edits included)
        await self.db.execute(
            "UPDATE proposals SET status = 'approved', payload = ?, result_id = ?, "
            "decided_at = datetime('now') WHERE id = ?",
            (data.model_dump_json(), row_id, proposal_id))
        await self._close_if_done(debrief_id)
        return row_id

    async def _resolve(self, debrief_id: int, ref: str) -> int:
        kind, _, value = ref.partition(":")
        if kind == "existing":
            if not await self.store.one("SELECT id FROM people WHERE id = ?", int(value)):
                raise ReviewError(f"Person {value} doesn't exist.")
            return int(value)
        for row in await self.store.rows(
            "SELECT * FROM proposals WHERE debrief_id = ? AND kind = 'new_person'", debrief_id):
            if json.loads(row["payload"])["tag"] == value:
                if row["status"] == "rejected":
                    raise ReviewError(f"New person {value} was rejected.")
                if row["status"] == "pending":
                    return await self._approve(row["id"], None)
                return row["result_id"]
        raise ReviewError(f"No new person tagged {value} in this debrief.")

    async def _supersede(self, old_id: int, person_id: int, new_id: int) -> None:
        old = await self.store.one("SELECT person_id, status FROM facts WHERE id = ?", old_id)
        if not old or old["person_id"] != person_id or old["status"] != "active":
            raise ReviewError(f"Fact {old_id} isn't an active fact about this person.")
        await self.db.execute(
            "UPDATE facts SET status = 'superseded', superseded_by = ? WHERE id = ?", (new_id, old_id))

    async def _pending_proposal(self, proposal_id: int) -> dict:
        prop = await self.store.one("SELECT * FROM proposals WHERE id = ?", proposal_id)
        if not prop:
            raise ReviewError(f"No proposal {proposal_id}.")
        if prop["status"] != "pending":
            raise ReviewError(f"Proposal {proposal_id} is already {prop['status']}.")
        return prop

    async def _insert(self, sql: str, *args) -> int:
        cur = await self.db.execute(sql, args)
        return cur.lastrowid

    async def _close_if_done(self, debrief_id: int) -> None:
        left = await self.store.one(
            "SELECT COUNT(*) AS n FROM proposals WHERE debrief_id = ? AND status = 'pending'", debrief_id)
        if left["n"] == 0:
            await self.db.execute("UPDATE debriefs SET status = 'reviewed' WHERE id = ?", (debrief_id,))
