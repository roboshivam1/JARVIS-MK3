# =============================================================================
# src/jarvis/core/db/repos/projects.py - the projects database
# =============================================================================
#
# Reads, the review queue, and the owner's own edits. The only writer of
# the project tables: the service proposes, this commits what the owner
# approves, in one transaction per decision.
# =============================================================================

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel

from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.log import get_logger
from jarvis.common.projects import (
    PROPOSAL_MODELS,
    RESULT_FIELDS,
    AliasProposal,
    DecisionProposal,
    ExtractionResult,
    FactProposal,
    LinkProposal,
    LogProposal,
    NewProjectProposal,
    PartProposal,
    PartUpdateProposal,
    PersonProposal,
    ProjectUpdate,
    ProposalKind,
    StatusProposal,
    TaskDoneProposal,
    TaskProposal,
)
from jarvis.core.db.database import Database

log = get_logger("core.db.projects")

Write = tuple[str, tuple[Any, ...]]

_CLOSE_DEBRIEF = (
    "UPDATE project_debriefs SET status = 'reviewed' WHERE id = ? AND NOT EXISTS "
    "(SELECT 1 FROM project_proposals WHERE debrief_id = ? AND status = 'pending')"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ProjectReviewError(Exception):
    """A proposal or edit cannot be done as asked. Shown to the owner as-is."""


class ProjectsRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def _rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._db.query(sql, params)]

    # -- finding projects -----------------------------------------------------

    async def aliases_of(self, project_id: str) -> list[str]:
        rows = await self._rows(
            "SELECT alias FROM project_aliases WHERE project_id = ? ORDER BY alias",
            (project_id,))
        return [r["alias"] for r in rows]

    async def roster(self, limit: int = 100) -> list[dict[str, Any]]:
        """Live projects first, most recently active first."""
        projects = await self._rows(
            "SELECT * FROM projects ORDER BY "
            "CASE status WHEN 'active' THEN 0 WHEN 'idea' THEN 1 WHEN 'paused' THEN 2 "
            "ELSE 3 END, COALESCE(last_activity_ts, created_ts) DESC LIMIT ?", (limit,))
        for p in projects:
            p["aliases"] = await self.aliases_of(p["id"])
        return projects

    async def find(self, name: str) -> list[dict[str, Any]]:
        return await self._rows(
            "SELECT DISTINCT p.* FROM projects p "
            "LEFT JOIN project_aliases a ON a.project_id = p.id "
            "WHERE p.name = ? COLLATE NOCASE OR a.alias = ? COLLATE NOCASE",
            (name.strip(), name.strip()))

    async def mentioned_in(self, text: str) -> list[str]:
        """Projects whose name or alias appears in text as whole words."""
        labels = await self._rows(
            "SELECT id AS pid, name AS label FROM projects "
            "UNION SELECT project_id, alias FROM project_aliases")
        found: list[str] = []
        for row in labels:
            label = row["label"].strip()
            if len(label) < 3:
                continue
            if (re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text, re.IGNORECASE)
                    and row["pid"] not in found):
                found.append(row["pid"])
        return found

    async def search(self, text: str, limit: int = 5) -> list[str]:
        """Project ids whose records match the words in text, best first."""
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2]
        if not tokens:
            return []
        rows = await self._rows(
            "SELECT project_id, MIN(rank) AS best FROM projects_fts "
            "WHERE projects_fts MATCH ? GROUP BY project_id ORDER BY best LIMIT ?",
            (" OR ".join(f'"{t}"' for t in tokens), limit))
        return [r["project_id"] for r in rows]

    async def profile(self, project_id: str) -> dict[str, Any] | None:
        rows = await self._rows("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not rows:
            return None
        p = rows[0]
        p["aliases"] = await self.aliases_of(project_id)
        p["decisions"] = await self._rows(
            "SELECT id, decision, why, alternatives, decided_on, status, created_ts "
            "FROM project_decisions WHERE project_id = ? AND status = 'active' "
            "ORDER BY COALESCE(decided_on, created_ts)", (project_id,))
        p["facts"] = await self._rows(
            "SELECT id, category, text, created_ts FROM project_facts "
            "WHERE project_id = ? AND status = 'active' ORDER BY category, created_ts",
            (project_id,))
        p["parts"] = await self._rows(
            "SELECT id, name, spec, quantity, status, source, cost_inr, notes "
            "FROM project_parts WHERE project_id = ? AND status != 'dropped' "
            "ORDER BY created_ts", (project_id,))
        p["tasks"] = await self._rows(
            "SELECT id, text, due_date, created_ts FROM project_tasks "
            "WHERE project_id = ? AND status = 'open' "
            "ORDER BY due_date IS NULL, due_date, created_ts", (project_id,))
        p["log"] = await self._rows(
            "SELECT id, happened_on, text, source FROM project_log WHERE project_id = ? "
            "ORDER BY happened_on DESC, id DESC", (project_id,))
        p["people"] = await self._rows(
            "SELECT pe.id, pe.name, pp.role FROM project_people pp "
            "JOIN people pe ON pe.id = pp.person_id WHERE pp.project_id = ? "
            "ORDER BY pe.name", (project_id,))
        p["links"] = await self._rows(
            "SELECT id, label, url FROM project_links WHERE project_id = ? "
            "ORDER BY created_ts", (project_id,))
        p["relations"] = await self._rows(
            "SELECT r.kind, r.to_id AS other_id, o.name AS other, 'out' AS direction "
            "FROM project_relations r JOIN projects o ON o.id = r.to_id WHERE r.from_id = ? "
            "UNION ALL SELECT r.kind, r.from_id, o.name, 'in' FROM project_relations r "
            "JOIN projects o ON o.id = r.from_id WHERE r.to_id = ?",
            (project_id, project_id))
        p["documents"] = await self._rows(
            "SELECT version, artifact_id, created_ts FROM project_documents "
            "WHERE project_id = ? ORDER BY version DESC", (project_id,))
        return p

    async def history(self, project_id: str) -> dict[str, list[dict[str, Any]]]:
        """Superseded and reversed records - what the project used to be."""
        return {
            "decisions": await self._rows(
                "SELECT id, decision, why, decided_on, status, superseded_by "
                "FROM project_decisions WHERE project_id = ? AND status != 'active' "
                "ORDER BY created_ts", (project_id,)),
            "facts": await self._rows(
                "SELECT id, category, text, status, superseded_by FROM project_facts "
                "WHERE project_id = ? AND status != 'active' ORDER BY created_ts",
                (project_id,)),
        }

    async def directory(self) -> list[dict[str, Any]]:
        """Every project with the counts the list view shows."""
        return await self._rows(
            "SELECT p.id, p.name, p.kind, p.status, p.pitch, p.last_activity_ts, "
            "(SELECT COUNT(*) FROM project_tasks t WHERE t.project_id = p.id "
            " AND t.status = 'open') AS open_tasks, "
            "(SELECT COUNT(*) FROM project_decisions d WHERE d.project_id = p.id "
            " AND d.status = 'active') AS decisions, "
            "(SELECT COUNT(*) FROM project_parts b WHERE b.project_id = p.id "
            " AND b.status IN ('needed', 'ordered')) AS parts_waiting "
            "FROM projects p ORDER BY "
            "CASE p.status WHEN 'active' THEN 0 WHEN 'idea' THEN 1 WHEN 'paused' THEN 2 "
            "ELSE 3 END, COALESCE(p.last_activity_ts, p.created_ts) DESC")

    async def stale(self, days: int, now_iso: str) -> list[dict[str, Any]]:
        """Active projects untouched for `days` or more."""
        rows = await self._rows(
            "SELECT id, name, COALESCE(last_activity_ts, created_ts) AS last "
            "FROM projects WHERE status = 'active' ORDER BY last")
        cutoff = date.fromisoformat(now_iso[:10]) - timedelta(days=days)
        return [r for r in rows if date.fromisoformat(r["last"][:10]) <= cutoff]

    async def tasks_due(self, until: str) -> list[dict[str, Any]]:
        """Open tasks due on or before `until` (YYYY-MM-DD), overdue included."""
        return await self._rows(
            "SELECT t.id, t.text, t.due_date, p.name FROM project_tasks t "
            "JOIN projects p ON p.id = t.project_id WHERE t.status = 'open' "
            "AND t.due_date IS NOT NULL AND t.due_date <= ? ORDER BY t.due_date",
            (until,))

    async def people_names(self, limit: int) -> list[dict[str, Any]]:
        """Active people, most recently mentioned first - for linking."""
        return await self._rows(
            "SELECT id, name FROM people WHERE status = 'active' ORDER BY "
            "COALESCE(last_mentioned_ts, created_ts) DESC LIMIT ?", (limit,))

    async def pending_count(self) -> int:
        rows = await self._rows(
            "SELECT COUNT(*) AS n FROM project_proposals WHERE status = 'pending'")
        return int(rows[0]["n"])

    # -- the review queue -----------------------------------------------------

    async def stage(self, raw_text: str, result: ExtractionResult, source: str) -> str:
        debrief_id = new_ulid()
        now = utc_now().isoformat()
        async with self._db.transaction() as tx:
            await tx.execute(
                "INSERT INTO project_debriefs (id, raw_text, source, status, created_ts) "
                "VALUES (?, ?, ?, 'pending', ?)", (debrief_id, raw_text, source, now))
            for kind, field in RESULT_FIELDS.items():
                items: list[BaseModel] = getattr(result, field)
                for item in items:
                    await tx.execute(
                        "INSERT INTO project_proposals (id, debrief_id, kind, payload) "
                        "VALUES (?, ?, ?, ?)",
                        (new_ulid(), debrief_id, kind.value, item.model_dump_json()))
            await tx.execute(_CLOSE_DEBRIEF, (debrief_id, debrief_id))
        log.info("project debrief staged", extra={
            "debrief_id": debrief_id, "proposals": result.count})
        return debrief_id

    async def pending_debriefs(self) -> list[dict[str, Any]]:
        debriefs = await self._rows(
            "SELECT id, raw_text, source, created_ts FROM project_debriefs "
            "WHERE status = 'pending' ORDER BY created_ts DESC")
        names = {p["id"]: p["name"] for p in await self._rows("SELECT id, name FROM projects")}
        for d in debriefs:
            proposals = await self._rows(
                "SELECT id, kind, payload FROM project_proposals "
                "WHERE debrief_id = ? AND status = 'pending' ORDER BY id", (d["id"],))
            new_projects: dict[str, str] = {}
            for row in await self._rows(
                    "SELECT payload FROM project_proposals WHERE debrief_id = ? "
                    "AND kind = 'new_project' AND status != 'rejected'", (d["id"],)):
                payload = json.loads(row["payload"])
                new_projects[payload["tag"]] = payload["name"]
            for p in proposals:
                p["payload"] = json.loads(p["payload"])
                ref = p["payload"].get("project_ref") or f"new:{p['payload'].get('tag', '')}"
                kind, _, value = ref.partition(":")
                p["project"] = (names.get(value) if kind == "existing"
                                else new_projects.get(value, ""))
            d["proposals"] = proposals
        return debriefs

    async def approve(self, proposal_id: str, edited: dict[str, Any] | None = None) -> str:
        writes: list[Write] = []
        result_id, debrief_id = await self._plan(proposal_id, edited, writes)
        writes.append((_CLOSE_DEBRIEF, (debrief_id, debrief_id)))
        async with self._db.transaction() as tx:
            for sql, params in writes:
                await tx.execute(sql, params)
        return result_id

    async def reject(self, proposal_id: str) -> list[str]:
        """Reject one proposal - and for a new project, everything about it."""
        prop = await self._pending(proposal_id)
        rejected = [proposal_id]
        if prop["kind"] == ProposalKind.NEW_PROJECT:
            ref = f"new:{json.loads(prop['payload'])['tag']}"
            for row in await self._rows(
                    "SELECT id, payload FROM project_proposals WHERE debrief_id = ? "
                    "AND status = 'pending' AND kind != 'new_project'", (prop["debrief_id"],)):
                if json.loads(row["payload"]).get("project_ref") == ref:
                    rejected.append(row["id"])
        now = utc_now().isoformat()
        async with self._db.transaction() as tx:
            for pid in rejected:
                await tx.execute(
                    "UPDATE project_proposals SET status = 'rejected', decided_ts = ? "
                    "WHERE id = ?", (now, pid))
            await tx.execute(_CLOSE_DEBRIEF, (prop["debrief_id"], prop["debrief_id"]))
        return rejected

    async def _pending(self, proposal_id: str) -> dict[str, Any]:
        rows = await self._rows("SELECT * FROM project_proposals WHERE id = ?", (proposal_id,))
        if not rows:
            raise ProjectReviewError(f"No proposal {proposal_id}.")
        if rows[0]["status"] != "pending":
            raise ProjectReviewError(f"That proposal is already {rows[0]['status']}.")
        return rows[0]

    async def _plan(self, proposal_id: str, edited: dict[str, Any] | None,
                    writes: list[Write]) -> tuple[str, str]:
        """Plan the writes one approval makes. Reads and checks only."""
        prop = await self._pending(proposal_id)
        kind = ProposalKind(prop["kind"])
        debrief_id: str = prop["debrief_id"]
        original = json.loads(prop["payload"])
        data = PROPOSAL_MODELS[kind].model_validate(edited if edited is not None else original)
        now = utc_now().isoformat()
        today = now[:10]
        row_id = new_ulid()
        project_id: str | None = None

        if isinstance(data, NewProjectProposal):
            if data.tag != original["tag"]:
                raise ProjectReviewError("A new project's tag cannot change - "
                                         "other proposals point at it.")
            if await self.find(data.name):
                raise ProjectReviewError(f"A project called {data.name!r} already exists.")
            writes.append((
                "INSERT INTO projects (id, name, kind, status, pitch, goal, started_on, "
                "last_activity_ts, created_ts, updated_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row_id, data.name, data.kind.value, data.status.value, data.pitch,
                 data.goal, data.started_on, now, now, now)))
        elif isinstance(data, PartUpdateProposal):
            part = await self._rows(
                "SELECT project_id, notes FROM project_parts WHERE id = ?", (data.part_id,))
            if not part:
                raise ProjectReviewError("That part is no longer in the database.")
            project_id, row_id = part[0]["project_id"], data.part_id
            notes = part[0]["notes"]
            if data.note:
                notes = f"{notes}; {data.note}" if notes else data.note
            writes.append((
                "UPDATE project_parts SET status = ?, notes = ?, updated_ts = ? WHERE id = ?",
                (data.status.value, notes, now, data.part_id)))
        elif isinstance(data, TaskDoneProposal):
            task = await self._rows(
                "SELECT project_id, status FROM project_tasks WHERE id = ?", (data.task_id,))
            if not task or task[0]["status"] != "open":
                raise ProjectReviewError("That task is not open any more.")
            project_id, row_id = task[0]["project_id"], data.task_id
            writes.append((
                "UPDATE project_tasks SET status = 'done', done_ts = ? WHERE id = ?",
                (now, data.task_id)))
        else:
            assert isinstance(data, (AliasProposal, StatusProposal, DecisionProposal,
                                     FactProposal, PartProposal, TaskProposal,
                                     LogProposal, PersonProposal, LinkProposal))
            project_id = await self._resolve(debrief_id, data.project_ref, writes)
            if isinstance(data, AliasProposal):
                row_id = project_id
                writes.append((
                    "INSERT OR IGNORE INTO project_aliases (project_id, alias) VALUES (?, ?)",
                    (project_id, data.alias)))
            elif isinstance(data, StatusProposal):
                row_id = project_id
                writes.append((
                    "UPDATE projects SET status = ? WHERE id = ?",
                    (data.status.value, project_id)))
            elif isinstance(data, DecisionProposal):
                writes.append((
                    "INSERT INTO project_decisions (id, project_id, decision, why, "
                    "alternatives, decided_on, status, debrief_id, created_ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                    (row_id, project_id, data.decision, data.why, data.alternatives,
                     data.decided_on or today, debrief_id, now)))
                if data.supersedes_decision_id:
                    await self._check_active("project_decisions",
                                             data.supersedes_decision_id, project_id)
                    writes.append((
                        "UPDATE project_decisions SET status = 'superseded', "
                        "superseded_by = ? WHERE id = ?",
                        (row_id, data.supersedes_decision_id)))
            elif isinstance(data, FactProposal):
                writes.append((
                    "INSERT INTO project_facts (id, project_id, category, text, status, "
                    "debrief_id, created_ts) VALUES (?, ?, ?, ?, 'active', ?, ?)",
                    (row_id, project_id, data.category.value, data.text, debrief_id, now)))
                if data.supersedes_fact_id:
                    await self._check_active("project_facts", data.supersedes_fact_id,
                                             project_id)
                    writes.append((
                        "UPDATE project_facts SET status = 'superseded', superseded_by = ? "
                        "WHERE id = ?", (row_id, data.supersedes_fact_id)))
            elif isinstance(data, PartProposal):
                writes.append((
                    "INSERT INTO project_parts (id, project_id, name, spec, quantity, "
                    "status, source, cost_inr, debrief_id, created_ts, updated_ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row_id, project_id, data.name, data.spec, data.quantity,
                     data.status.value, data.source, data.cost_inr, debrief_id, now, now)))
            elif isinstance(data, TaskProposal):
                writes.append((
                    "INSERT INTO project_tasks (id, project_id, text, due_date, status, "
                    "debrief_id, created_ts) VALUES (?, ?, ?, ?, 'open', ?, ?)",
                    (row_id, project_id, data.text, data.due_date, debrief_id, now)))
            elif isinstance(data, LogProposal):
                writes.append((
                    "INSERT INTO project_log (id, project_id, happened_on, text, source, "
                    "debrief_id, created_ts) VALUES (?, ?, ?, ?, 'debrief', ?, ?)",
                    (row_id, project_id, data.happened_on, data.text, debrief_id, now)))
            elif isinstance(data, PersonProposal):
                if not await self._rows("SELECT id FROM people WHERE id = ?", (data.person_id,)):
                    raise ProjectReviewError("That person is no longer in the people database.")
                row_id = data.person_id
                writes.append((
                    "INSERT INTO project_people (project_id, person_id, role) VALUES (?, ?, ?) "
                    "ON CONFLICT (project_id, person_id) DO UPDATE SET role = excluded.role",
                    (project_id, data.person_id, data.role)))
            else:
                writes.append((
                    "INSERT INTO project_links (id, project_id, label, url, created_ts) "
                    "VALUES (?, ?, ?, ?, ?)", (row_id, project_id, data.label, data.url, now)))

        if project_id is not None:
            writes.append((
                "UPDATE projects SET last_activity_ts = ?, updated_ts = ? WHERE id = ?",
                (now, now, project_id)))
        writes.append((
            "UPDATE project_proposals SET status = 'approved', payload = ?, result_id = ?, "
            "decided_ts = ? WHERE id = ?",
            (data.model_dump_json(), row_id, now, proposal_id)))
        return row_id, debrief_id

    async def _resolve(self, debrief_id: str, ref: str, writes: list[Write]) -> str:
        kind, _, value = ref.partition(":")
        if kind == "existing":
            if not await self._rows("SELECT id FROM projects WHERE id = ?", (value,)):
                raise ProjectReviewError(f"There is no project {value}.")
            return value
        for row in await self._rows(
                "SELECT id, payload, status, result_id FROM project_proposals "
                "WHERE debrief_id = ? AND kind = 'new_project'", (debrief_id,)):
            if json.loads(row["payload"])["tag"] != value:
                continue
            if row["status"] == "rejected":
                raise ProjectReviewError(f"New project {value} was rejected.")
            if row["status"] == "pending":
                project_id, _ = await self._plan(row["id"], None, writes)
                return project_id
            result_id: str = row["result_id"]
            return result_id
        raise ProjectReviewError(f"No new project tagged {value} in this debrief.")

    async def _check_active(self, table: str, row_id: str, project_id: str) -> None:
        rows = await self._rows(
            f"SELECT project_id, status FROM {table} WHERE id = ?", (row_id,))
        if not rows or rows[0]["project_id"] != project_id or rows[0]["status"] != "active":
            raise ProjectReviewError("What this replaces is no longer an active record "
                                     "of this project.")

    # -- the owner's own edits ------------------------------------------------

    async def update_project(self, project_id: str, data: ProjectUpdate) -> None:
        changes = data.changes()
        if not changes:
            return
        if "name" in changes:
            clash = [p for p in await self.find(changes["name"]) if p["id"] != project_id]
            if clash:
                raise ProjectReviewError(f"A project called {changes['name']!r} already exists.")
        changes["updated_ts"] = utc_now().isoformat()
        assignments = ", ".join(f"{key} = ?" for key in changes)
        await self._db.execute(
            f"UPDATE projects SET {assignments} WHERE id = ?",
            (*changes.values(), project_id))

    async def add_log(self, project_id: str, text: str, source: str = "owner",
                      happened_on: str | None = None) -> str:
        now = utc_now().isoformat()
        row_id = new_ulid()
        async with self._db.transaction() as tx:
            await tx.execute(
                "INSERT INTO project_log (id, project_id, happened_on, text, source, created_ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (row_id, project_id, happened_on or now[:10], text, source, now))
            await tx.execute(
                "UPDATE projects SET last_activity_ts = ? WHERE id = ?", (now, project_id))
        return row_id

    async def resolve_task(self, task_id: str, status: str) -> None:
        if status not in ("done", "dropped", "open"):
            raise ProjectReviewError(f"Not a task status: {status}")
        await self._db.execute(
            "UPDATE project_tasks SET status = ?, done_ts = ? WHERE id = ?",
            (status, None if status == "open" else utc_now().isoformat(), task_id))

    async def set_part_status(self, part_id: str, status: str) -> None:
        if status not in ("needed", "ordered", "have", "used", "dropped"):
            raise ProjectReviewError(f"Not a part status: {status}")
        await self._db.execute(
            "UPDATE project_parts SET status = ?, updated_ts = ? WHERE id = ?",
            (status, utc_now().isoformat(), part_id))

    async def retire(self, table: str, row_id: str) -> None:
        """Retire a fact or reverse a decision the owner says is wrong."""
        if table == "facts":
            await self._db.execute(
                "UPDATE project_facts SET status = 'retired' WHERE id = ?", (row_id,))
        elif table == "decisions":
            await self._db.execute(
                "UPDATE project_decisions SET status = 'reversed' WHERE id = ?", (row_id,))
        else:
            raise ProjectReviewError(f"Cannot retire from {table}.")

    async def delete_project(self, project_id: str) -> None:
        await self._db.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    async def relate(self, from_id: str, to_id: str, kind: str, present: bool = True) -> None:
        if from_id == to_id:
            raise ProjectReviewError("A project cannot be related to itself.")
        if present:
            await self._db.execute(
                "INSERT OR IGNORE INTO project_relations (from_id, to_id, kind) "
                "VALUES (?, ?, ?)", (from_id, to_id, kind))
        else:
            await self._db.execute(
                "DELETE FROM project_relations WHERE from_id = ? AND to_id = ? AND kind = ?",
                (from_id, to_id, kind))

    # -- the master document --------------------------------------------------

    async def record_document(self, project_id: str, artifact_id: str,
                              job_id: str | None) -> int:
        async with self._db.transaction() as tx:
            await tx.execute(
                "INSERT INTO project_documents (id, project_id, version, artifact_id, "
                "job_id, created_ts) VALUES (?, ?, (SELECT COALESCE(MAX(version), 0) + 1 "
                "FROM project_documents WHERE project_id = ?), ?, ?, ?)",
                (new_ulid(), project_id, project_id, artifact_id, job_id,
                 utc_now().isoformat()))
        rows = await self._rows(
            "SELECT MAX(version) AS v FROM project_documents WHERE project_id = ?",
            (project_id,))
        return int(rows[0]["v"])
