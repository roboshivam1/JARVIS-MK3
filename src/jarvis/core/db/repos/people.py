# =============================================================================
# src/jarvis/core/db/repos/people.py - the people database
# =============================================================================
#
# The only code touching the person_* and people_* tables. Two halves:
#
#   reads    - lookup by name or alias, spotting who a message mentions,
#              full profiles, the roster
#   review   - staging a debrief's proposals, then approving or rejecting
#              them one at a time
#
# Nothing reaches the real tables except through approve(). The
# extraction model writes proposals and nothing else.
#
# DECIDE, THEN WRITE. approve() does every read and every check first,
# building a list of write statements, and only then runs the whole list
# in one transaction. IDs are ULIDs minted in Python, so no statement
# ever needs to read back what an earlier one inserted - which matters,
# because a transaction here can only execute, not query.
#
# Approving something about a new person approves that person too, in
# the same transaction. Rejecting a new person rejects everything that
# was about them. A debrief closes itself when nothing in it is pending.
# =============================================================================

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.log import get_logger
from jarvis.common.people import (
    PROPOSAL_MODELS,
    RESULT_FIELDS,
    AliasProposal,
    CommitmentProposal,
    ExtractionResult,
    FactProposal,
    GroupIn,
    KeyDateProposal,
    NewPersonProposal,
    OwnerFact,
    PersonUpdate,
    ProposalKind,
    ProposalStatus,
)
from jarvis.core.db.database import Database

log = get_logger("core.db.people")

Write = tuple[str, tuple[Any, ...]]

# Runs last in every decision's transaction, so it sees that decision.
_CLOSE_DEBRIEF = (
    "UPDATE people_debriefs SET status = 'reviewed' WHERE id = ? AND NOT EXISTS "
    "(SELECT 1 FROM people_proposals WHERE debrief_id = ? AND status = 'pending')"
)


class ReviewError(Exception):
    """A proposal cannot be decided as asked. The message is written to
    be shown to the owner as-is."""


class PeopleRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._db.query(sql, params)]

    # -- reads ----------------------------------------------------------------

    async def aliases_of(self, person_id: str) -> list[str]:
        rows = await self._rows(
            "SELECT alias FROM person_aliases WHERE person_id = ? ORDER BY alias",
            (person_id,))
        return [r["alias"] for r in rows]

    async def find(self, name: str) -> list[dict[str, Any]]:
        """Everyone matching a name or alias, case-insensitively. Can be
        more than one - two friends can share a name."""
        return await self._rows(
            "SELECT DISTINCT p.* FROM people p "
            "LEFT JOIN person_aliases a ON a.person_id = p.id "
            "WHERE p.name = ? COLLATE NOCASE OR a.alias = ?",
            (name, name))

    async def mentioned_in(self, text: str) -> list[str]:
        """IDs of people whose name, first name, or alias appears in text
        as a whole word. Cheap enough to run on every message at personal
        scale; a few hundred people is a few hundred regex checks."""
        labels = await self._rows(
            "SELECT id AS pid, name AS label FROM people WHERE status = 'active' "
            "UNION SELECT person_id, alias FROM person_aliases")
        found: list[str] = []
        for row in labels:
            candidates = {row["label"]}
            first = row["label"].split()[0]
            if len(first) > 2:
                candidates.add(first)
            if any(re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE)
                   for c in candidates) and row["pid"] not in found:
                found.append(row["pid"])
        return found

    async def profile(
        self, person_id: str, include_sensitive: bool = True
    ) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM people WHERE id = ?", (person_id,))
        if not rows:
            return None
        person = rows[0]
        sensitive = "" if include_sensitive else "AND sensitivity = 'normal'"
        person["aliases"] = await self.aliases_of(person_id)
        person["facts"] = await self._rows(
            f"SELECT id, category, text, sensitivity, created_ts, debrief_id "
            f"FROM person_facts "
            f"WHERE person_id = ? AND status = 'active' {sensitive} "
            f"ORDER BY category, created_ts", (person_id,))
        person["commitments"] = await self._rows(
            "SELECT id, direction, what, due_date, created_ts FROM person_commitments "
            "WHERE person_id = ? AND status = 'open' ORDER BY created_ts", (person_id,))
        person["dates"] = await self._rows(
            "SELECT id, label, date, recurring FROM person_dates "
            "WHERE person_id = ? ORDER BY date", (person_id,))
        return person

    async def roster(self, limit: int = 20) -> list[dict[str, Any]]:
        """Closest first, then most recently mentioned."""
        people = await self._rows(
            "SELECT id, name, relationship, summary FROM people "
            "WHERE status = 'active' "
            "ORDER BY closeness, last_mentioned_ts DESC NULLS LAST, name LIMIT ?",
            (limit,))
        for p in people:
            p["aliases"] = await self.aliases_of(p["id"])
            p["groups"] = [g["name"] for g in await self.groups_of(p["id"])]
        return people

    async def pending_debriefs(self) -> list[dict[str, Any]]:
        """Debriefs waiting for review, newest first, each with its
        pending proposals attached - the review page's whole payload."""
        debriefs = await self._rows(
            "SELECT id, raw_text, source, created_ts FROM people_debriefs "
            "WHERE status = 'pending' ORDER BY created_ts DESC")
        for d in debriefs:
            proposals = await self._rows(
                "SELECT id, kind, payload FROM people_proposals "
                "WHERE debrief_id = ? AND status = 'pending' ORDER BY id", (d["id"],))
            for p in proposals:
                p["payload"] = json.loads(p["payload"])
            d["proposals"] = proposals
            # tag -> name for every new person in this debrief that was
            # not rejected, so a pending fact about an already-approved
            # new person can still say whose it is.
            d["new_people"] = {}
            for row in await self._rows(
                "SELECT payload, status, result_id FROM people_proposals "
                "WHERE debrief_id = ? AND kind = 'new_person' AND status != 'rejected'",
                (d["id"],)):
                person = json.loads(row["payload"])
                d["new_people"][person["tag"]] = {
                    "name": person["name"], "approved": row["status"] == "approved"}
        return debriefs

    async def open_commitments(self) -> list[dict[str, Any]]:
        """Every open commitment, both directions, with whose it is."""
        return await self._rows(
            "SELECT c.direction, c.what, c.due_date, p.name FROM person_commitments c "
            "JOIN people p ON p.id = c.person_id WHERE c.status = 'open' "
            "ORDER BY c.due_date IS NULL, c.due_date, c.created_ts")

    async def all_dates(self) -> list[dict[str, Any]]:
        """Every stored date of every active person."""
        return await self._rows(
            "SELECT d.label, d.date, d.recurring, p.name FROM person_dates d "
            "JOIN people p ON p.id = d.person_id WHERE p.status = 'active'")

    # -- staging --------------------------------------------------------------

    async def stage(self, raw_text: str, result: ExtractionResult, source: str) -> str:
        """Record a debrief and its proposals. Returns the debrief id."""
        debrief_id = new_ulid()
        now = utc_now().isoformat()
        async with self._db.transaction() as tx:
            await tx.execute(
                "INSERT INTO people_debriefs (id, raw_text, source, status, created_ts) "
                "VALUES (?, ?, ?, 'pending', ?)", (debrief_id, raw_text, source, now))
            for kind, field in RESULT_FIELDS.items():
                items: list[BaseModel] = getattr(result, field)
                for item in items:
                    await tx.execute(
                        "INSERT INTO people_proposals (id, debrief_id, kind, payload) "
                        "VALUES (?, ?, ?, ?)",
                        (new_ulid(), debrief_id, kind.value, item.model_dump_json()))
            await tx.execute(_CLOSE_DEBRIEF, (debrief_id, debrief_id))
        log.info("debrief staged", extra={"debrief_id": debrief_id, "proposals": result.count})
        return debrief_id

    # -- decisions ------------------------------------------------------------

    async def approve(self, proposal_id: str, edited: dict[str, Any] | None = None) -> str:
        """Commit one proposal - with the owner's edits, if any. Returns
        the id of the row it created (for an alias, the person's id)."""
        writes: list[Write] = []
        result_id, debrief_id = await self._plan_approval(proposal_id, edited, writes)
        writes.append((_CLOSE_DEBRIEF, (debrief_id, debrief_id)))
        async with self._db.transaction() as tx:
            for sql, params in writes:
                await tx.execute(sql, params)
        return result_id

    async def reject(self, proposal_id: str) -> list[str]:
        """Reject a proposal, and if it is a new person, everything about
        them. Returns every proposal id that was rejected."""
        prop = await self._pending(proposal_id)
        rejected = [proposal_id]
        if prop["kind"] == ProposalKind.NEW_PERSON:
            ref = f"new:{json.loads(prop['payload'])['tag']}"
            for row in await self._rows(
                "SELECT id, payload FROM people_proposals WHERE debrief_id = ? "
                "AND status = 'pending' AND kind != 'new_person'", (prop["debrief_id"],)):
                if json.loads(row["payload"]).get("person_ref") == ref:
                    rejected.append(row["id"])
        now = utc_now().isoformat()
        async with self._db.transaction() as tx:
            for pid in rejected:
                await tx.execute(
                    "UPDATE people_proposals SET status = 'rejected', decided_ts = ? "
                    "WHERE id = ?", (now, pid))
            await tx.execute(_CLOSE_DEBRIEF, (prop["debrief_id"], prop["debrief_id"]))
        return rejected

    # -- planning an approval (reads and checks only; no writes) -------------

    async def _plan_approval(
        self, proposal_id: str, edited: dict[str, Any] | None, writes: list[Write]
    ) -> tuple[str, str]:
        prop = await self._pending(proposal_id)
        kind = ProposalKind(prop["kind"])
        debrief_id: str = prop["debrief_id"]
        original = json.loads(prop["payload"])
        data = PROPOSAL_MODELS[kind].model_validate(edited if edited is not None else original)
        now = utc_now().isoformat()
        row_id = new_ulid()

        if isinstance(data, NewPersonProposal):
            if data.tag != original["tag"]:
                raise ReviewError("A new person's tag cannot change - other proposals point at it.")
            writes.append((
                "INSERT INTO people (id, name, relationship, closeness, how_met, summary, "
                "status, last_mentioned_ts, created_ts, updated_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)",
                (row_id, data.name, data.relationship, data.closeness,
                 data.how_met, data.summary, now, now, now)))
        else:
            assert isinstance(data, (AliasProposal, FactProposal,
                                     CommitmentProposal, KeyDateProposal))
            person_id = await self._resolve(debrief_id, data.person_ref, writes)
            if isinstance(data, AliasProposal):
                row_id = person_id
                writes.append((
                    "INSERT OR IGNORE INTO person_aliases (person_id, alias) VALUES (?, ?)",
                    (person_id, data.alias)))
            elif isinstance(data, FactProposal):
                writes.append((
                    "INSERT INTO person_facts (id, person_id, category, text, sensitivity, "
                    "status, debrief_id, created_ts) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
                    (row_id, person_id, data.category.value, data.text,
                     data.sensitivity.value, debrief_id, now)))
                if data.supersedes_fact_id:
                    await self._check_supersedable(data.supersedes_fact_id, person_id)
                    writes.append((
                        "UPDATE person_facts SET status = 'superseded', superseded_by = ? "
                        "WHERE id = ?", (row_id, data.supersedes_fact_id)))
            elif isinstance(data, CommitmentProposal):
                writes.append((
                    "INSERT INTO person_commitments (id, person_id, direction, what, "
                    "due_date, status, debrief_id, created_ts) "
                    "VALUES (?, ?, ?, ?, ?, 'open', ?, ?)",
                    (row_id, person_id, data.direction.value, data.what,
                     data.due_date, debrief_id, now)))
            else:
                writes.append((
                    "INSERT INTO person_dates (id, person_id, label, date, recurring, "
                    "debrief_id, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row_id, person_id, data.label, data.date, int(data.recurring),
                     debrief_id, now)))
            writes.append((
                "UPDATE people SET last_mentioned_ts = ?, updated_ts = ? WHERE id = ?",
                (now, now, person_id)))

        # Record what was actually approved, edits included.
        writes.append((
            "UPDATE people_proposals SET status = 'approved', payload = ?, "
            "result_id = ?, decided_ts = ? WHERE id = ?",
            (data.model_dump_json(), row_id, now, proposal_id)))
        return row_id, debrief_id

    async def _resolve(self, debrief_id: str, ref: str, writes: list[Write]) -> str:
        """person_ref -> a person id, planning the new person's approval
        first if it is still pending."""
        kind, _, value = ref.partition(":")
        if kind == "existing":
            if not await self._rows("SELECT id FROM people WHERE id = ?", (value,)):
                raise ReviewError(f"There is no person {value}.")
            return value
        for row in await self._rows(
            "SELECT id, payload, status, result_id FROM people_proposals "
            "WHERE debrief_id = ? AND kind = 'new_person'", (debrief_id,)):
            if json.loads(row["payload"])["tag"] != value:
                continue
            if row["status"] == ProposalStatus.REJECTED:
                raise ReviewError(f"New person {value} was rejected.")
            if row["status"] == ProposalStatus.PENDING:
                person_id, _ = await self._plan_approval(row["id"], None, writes)
                return person_id
            result_id: str = row["result_id"]
            return result_id
        raise ReviewError(f"No new person tagged {value} in this debrief.")

    async def _check_supersedable(self, fact_id: str, person_id: str) -> None:
        rows = await self._rows(
            "SELECT person_id, status FROM person_facts WHERE id = ?", (fact_id,))
        if not rows or rows[0]["person_id"] != person_id or rows[0]["status"] != "active":
            raise ReviewError("The fact this replaces is not an active fact about this person.")

    async def _pending(self, proposal_id: str) -> dict[str, Any]:
        rows = await self._rows("SELECT * FROM people_proposals WHERE id = ?", (proposal_id,))
        if not rows:
            raise ReviewError(f"No proposal {proposal_id}.")
        if rows[0]["status"] != ProposalStatus.PENDING:
            raise ReviewError(f"That proposal is already {rows[0]['status']}.")
        return rows[0]
    # -- groups ---------------------------------------------------------------

    async def groups(self) -> list[dict[str, Any]]:
        """Every group, with its members' ids."""
        groups = await self._rows(
            "SELECT id, name, kind, description FROM people_groups ORDER BY kind, name")
        for g in groups:
            rows = await self._rows(
                "SELECT person_id FROM person_group_members WHERE group_id = ?", (g["id"],))
            g["members"] = [r["person_id"] for r in rows]
        return groups

    async def groups_of(self, person_id: str) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT g.id, g.name, g.kind FROM people_groups g "
            "JOIN person_group_members m ON m.group_id = g.id "
            "WHERE m.person_id = ? ORDER BY g.name", (person_id,))

    async def create_group(self, data: GroupIn) -> str:
        existing = await self._rows(
            "SELECT id FROM people_groups WHERE name = ?", (data.name,))
        if existing:
            raise ReviewError(f"A group called {data.name!r} already exists.")
        group_id = new_ulid()
        await self._db.execute(
            "INSERT INTO people_groups (id, name, kind, description, created_ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (group_id, data.name, data.kind.value, data.description, utc_now().isoformat()))
        return group_id

    async def update_group(self, group_id: str, data: GroupIn) -> None:
        await self._db.execute(
            "UPDATE people_groups SET name = ?, kind = ?, description = ? WHERE id = ?",
            (data.name, data.kind.value, data.description, group_id))

    async def delete_group(self, group_id: str) -> None:
        """Removes the group. The people in it are untouched."""
        await self._db.execute("DELETE FROM people_groups WHERE id = ?", (group_id,))

    async def set_membership(self, group_id: str, person_id: str, member: bool) -> None:
        if member:
            await self._db.execute(
                "INSERT OR IGNORE INTO person_group_members "
                "(group_id, person_id, created_ts) VALUES (?, ?, ?)",
                (group_id, person_id, utc_now().isoformat()))
        else:
            await self._db.execute(
                "DELETE FROM person_group_members WHERE group_id = ? AND person_id = ?",
                (group_id, person_id))

    async def group_by_name(self, name: str) -> dict[str, Any] | None:
        rows = await self._rows(
            "SELECT id, name, kind FROM people_groups WHERE name = ?", (name,))
        return rows[0] if rows else None

    async def members_of(self, group_id: str) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT p.id, p.name, p.summary FROM people p "
            "JOIN person_group_members m ON m.person_id = p.id "
            "WHERE m.group_id = ? ORDER BY p.name", (group_id,))

    # -- the directory and one person in full ---------------------------------

    async def directory(self) -> dict[str, Any]:
        """Everyone and every group: the People section's whole payload."""
        people = await self._rows(
            "SELECT id, name, relationship, summary, status, closeness, last_mentioned_ts "
            "FROM people ORDER BY name")
        for p in people:
            p["aliases"] = await self.aliases_of(p["id"])
            p["groups"] = [g["id"] for g in await self.groups_of(p["id"])]
        return {"people": people, "groups": await self.groups()}

    async def person_detail(self, person_id: str) -> dict[str, Any] | None:
        """A profile plus everything the page needs to show WHY: the
        facts that were superseded, and the debriefs they came from."""
        person = await self.profile(person_id, include_sensitive=True)
        if person is None:
            return None
        person["groups"] = await self.groups_of(person_id)
        person["history"] = await self._rows(
            "SELECT id, category, text, sensitivity, created_ts, superseded_by, debrief_id "
            "FROM person_facts WHERE person_id = ? AND status = 'superseded' "
            "ORDER BY created_ts DESC", (person_id,))
        person["settled"] = await self._rows(
            "SELECT direction, what, status, resolved_ts FROM person_commitments "
            "WHERE person_id = ? AND status != 'open' ORDER BY resolved_ts DESC LIMIT 20",
            (person_id,))
        # The debriefs behind the facts, so a fact can always be traced
        # back to the words that produced it.
        ids = {f["debrief_id"] for f in person["facts"] + person["history"] if f.get("debrief_id")}
        person["debriefs"] = {}
        for debrief_id in ids:
            rows = await self._rows(
                "SELECT id, raw_text, source, created_ts FROM people_debriefs WHERE id = ?",
                (debrief_id,))
            if rows:
                person["debriefs"][debrief_id] = rows[0]
        return person

    # -- the owner's own edits ------------------------------------------------
    #
    # These skip the review queue on purpose: review gates the extraction
    # model, not the owner. Facts are still never overwritten - editing
    # one supersedes it, so the person's history stays readable.

    async def update_person(self, person_id: str, data: PersonUpdate) -> None:
        changes = data.changes()
        if not changes:
            return
        columns = ", ".join(f"{name} = ?" for name in changes)
        await self._db.execute(
            f"UPDATE people SET {columns}, updated_ts = ? WHERE id = ?",
            (*changes.values(), utc_now().isoformat(), person_id))

    async def set_alias(self, person_id: str, alias: str, present: bool) -> None:
        if present:
            await self._db.execute(
                "INSERT OR IGNORE INTO person_aliases (person_id, alias) VALUES (?, ?)",
                (person_id, alias))
        else:
            await self._db.execute(
                "DELETE FROM person_aliases WHERE person_id = ? AND alias = ?",
                (person_id, alias))

    async def add_fact(self, person_id: str, data: OwnerFact) -> str:
        fact_id = new_ulid()
        await self._db.execute(
            "INSERT INTO person_facts (id, person_id, category, text, sensitivity, "
            "status, created_ts) VALUES (?, ?, ?, ?, ?, 'active', ?)",
            (fact_id, person_id, data.category.value, data.text,
             data.sensitivity.value, utc_now().isoformat()))
        return fact_id

    async def edit_fact(self, fact_id: str, data: OwnerFact) -> str:
        """Replaces a fact with a corrected version. The old text stays in
        the person's history, marked as superseded by the new one."""
        rows = await self._rows(
            "SELECT person_id, status FROM person_facts WHERE id = ?", (fact_id,))
        if not rows or rows[0]["status"] != "active":
            raise ReviewError("That fact is not an active fact.")
        new_id = new_ulid()
        now = utc_now().isoformat()
        async with self._db.transaction() as tx:
            await tx.execute(
                "INSERT INTO person_facts (id, person_id, category, text, sensitivity, "
                "status, created_ts) VALUES (?, ?, ?, ?, ?, 'active', ?)",
                (new_id, rows[0]["person_id"], data.category.value, data.text,
                 data.sensitivity.value, now))
            await tx.execute(
                "UPDATE person_facts SET status = 'superseded', superseded_by = ? WHERE id = ?",
                (new_id, fact_id))
        return new_id

    async def retire_fact(self, fact_id: str) -> None:
        """No longer true, and nothing replaces it. Superseded by nothing,
        which is how the history shows a fact that simply stopped holding."""
        await self._db.execute(
            "UPDATE person_facts SET status = 'superseded', superseded_by = NULL "
            "WHERE id = ? AND status = 'active'", (fact_id,))

    async def resolve_commitment(self, commitment_id: str, status: str) -> None:
        if status not in ("done", "dropped"):
            raise ReviewError("A commitment can only be marked done or dropped.")
        await self._db.execute(
            "UPDATE person_commitments SET status = ?, resolved_ts = ? WHERE id = ?",
            (status, utc_now().isoformat(), commitment_id))

    async def delete_date(self, date_id: str) -> None:
        await self._db.execute("DELETE FROM person_dates WHERE id = ?", (date_id,))

    async def delete_person(self, person_id: str) -> None:
        """Everything about them goes: facts, commitments, dates, aliases,
        memberships, all by cascade."""
        await self._db.execute("DELETE FROM people WHERE id = ?", (person_id,))

    async def merge_people(self, keep_id: str, merge_id: str) -> None:
        """Two rows for one person - the usual cause is the model creating
        someone it could not tell was already known. Everything moves to
        the kept row, the old name survives as an alias, and pending
        proposals still pointing at the absorbed row are re-pointed."""
        if keep_id == merge_id:
            raise ReviewError("Those are the same person.")
        rows = await self._rows(
            "SELECT id, name FROM people WHERE id IN (?, ?)", (keep_id, merge_id))
        if len(rows) != 2:
            raise ReviewError("One of those people no longer exists.")
        absorbed = next(r for r in rows if r["id"] == merge_id)
        now = utc_now().isoformat()
        async with self._db.transaction() as tx:
            for table in ("person_facts", "person_commitments", "person_dates"):
                await tx.execute(
                    f"UPDATE {table} SET person_id = ? WHERE person_id = ?", (keep_id, merge_id))
            await tx.execute(
                "INSERT OR IGNORE INTO person_aliases (person_id, alias) "
                "SELECT ?, alias FROM person_aliases WHERE person_id = ?", (keep_id, merge_id))
            await tx.execute(
                "INSERT OR IGNORE INTO person_aliases (person_id, alias) VALUES (?, ?)",
                (keep_id, absorbed["name"]))
            await tx.execute(
                "INSERT OR IGNORE INTO person_group_members (group_id, person_id, created_ts) "
                "SELECT group_id, ?, ? FROM person_group_members WHERE person_id = ?",
                (keep_id, now, merge_id))
            await tx.execute(
                "UPDATE people_proposals SET payload = replace(payload, ?, ?) "
                "WHERE status = 'pending'",
                (f"existing:{merge_id}", f"existing:{keep_id}"))
            await tx.execute("DELETE FROM people WHERE id = ?", (merge_id,))
            await tx.execute(
                "UPDATE people SET updated_ts = ? WHERE id = ?", (now, keep_id))
