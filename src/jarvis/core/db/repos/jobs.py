# =============================================================================
# src/jarvis/core/db/repos/jobs.py - repository for the jobs table
# =============================================================================
#
# The only code that touches the jobs table. Two disciplines enforced at
# this boundary:
#
#   1. LEGAL TRANSITIONS ONLY: every status change consults the state
#      machine map first; an illegal edge raises before any SQL runs.
#   2. OPTIMISTIC CHECKS: every status UPDATE carries
#      "WHERE status = <expected>". If another coroutine moved the job
#      first, zero rows change and the caller gets False instead of a
#      silent overwrite. Check and write are one atomic act.
#
# Event pairing: create() writes the job AND its job.enqueued event in
# ONE transaction - a crash between them cannot produce a job with no
# trail or a trail with no job. Transition events for later states are
# emitted by the queue engine (next batch), which owns the semantics of
# each transition.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import utc_now
from jarvis.common.jobs import (
    Approval,
    Job,
    JobStatus,
    Lease,
    TERMINAL_STATUSES,
    is_legal_transition,
)
from jarvis.common.log import get_logger
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo

log = get_logger("core.db.jobs")


class IllegalTransition(RuntimeError):
    """Code attempted a status edge the state machine forbids. Always a
    programming error, never a runtime condition to retry."""


class JobsRepo:
    """Store, transition, and query jobs."""

    def __init__(self, db: Database, events: EventsRepo) -> None:
        self._db = db
        self._events = events

    # -- creation -------------------------------------------------------------

    async def create(self, job: Job) -> None:
        """Persist a new job and its enqueued event atomically. After this
        commits, the job is real: the orchestrator may promise it."""
        async with self._db.transaction() as tx:
            await tx.execute(
                "INSERT INTO jobs "
                "(id, type, status, priority, requires, payload, result, "
                " error, artifacts, session_id, parent_job_id, approval, "
                " checkpoint, attempts, max_attempts, lease, not_before, "
                " created_ts, updated_ts, finished_ts, trace_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "        ?, ?, ?, ?)",
                self._to_row(job),
            )
            await self._events.append(
                Event(
                    kind=EventKind.JOB_ENQUEUED,
                    source="core.jobs",
                    job_id=job.id,
                    session_id=job.session_id,
                    trace_id=job.trace_id,
                    payload={"type": job.type, "priority": job.priority,
                             "requires": job.requires},
                ),
                tx=tx,
            )

    # -- transitions ----------------------------------------------------------

    async def transition(
        self,
        job_id: str,
        expected: JobStatus,
        new: JobStatus,
        *,
        set_fields: dict[str, Any] | None = None,
    ) -> bool:
        """Move a job along one legal edge, atomically.

        Returns True if THIS call performed the move; False if the job was
        not in `expected` status anymore (someone else moved it first) -
        the caller decides what a lost race means for it.

        set_fields lets a transition carry its baggage in the same atomic
        write: a lease when leasing, an error when failing, a result when
        succeeding. Keys are column names; values are Python values
        (JSON-encoded here where the column stores JSON).
        """
        if not is_legal_transition(expected, new):
            raise IllegalTransition(f"{expected} -> {new} is not a legal edge")

        assignments = ["status = ?", "updated_ts = ?"]
        params: list[Any] = [new.value, utc_now().isoformat()]

        # Terminal states stamp their finish time unless the caller set one.
        fields = dict(set_fields or {})
        if new in TERMINAL_STATUSES and "finished_ts" not in fields:
            fields["finished_ts"] = utc_now().isoformat()

        for column, value in fields.items():
            assignments.append(f"{column} = ?")
            params.append(self._encode_field(column, value))

        params.extend([job_id, expected.value])
        sql = (
            f"UPDATE jobs SET {', '.join(assignments)} "
            f"WHERE id = ? AND status = ?"    # <- the optimistic check
        )

        cursor = await self._db.query(
            "SELECT changes() AS n", ()
        )  # placeholder; real change count read below
        await self._db.execute(sql, params)
        row = await self._db.query_one("SELECT changes() AS n")
        moved = bool(row and int(row["n"]) == 1)
        if not moved:
            log.debug("transition lost race or job absent", extra={
                "job_id": job_id, "expected": expected.value, "new": new.value,
            })
        return moved

    async def set_checkpoint(self, job_id: str, checkpoint: dict[str, Any]) -> None:
        """Persist resume state for a running job. Not a status change, so
        no optimistic check - last checkpoint wins, which is correct: a
        newer checkpoint always supersedes an older one."""
        await self._db.execute(
            "UPDATE jobs SET checkpoint = ?, updated_ts = ? WHERE id = ?",
            (
                json.dumps(checkpoint, ensure_ascii=False, default=str),
                utc_now().isoformat(),
                job_id,
            ),
        )

    # -- queries --------------------------------------------------------------

    async def get(self, job_id: str) -> Job | None:
        row = await self._db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return self._to_job(row) if row else None

    async def next_queued(
        self,
        capabilities: set[str],
        now: datetime | None = None,
    ) -> Job | None:
        """The dispatch question: the oldest, most urgent queued job this
        capability set can serve, honouring not_before. Returns it WITHOUT
        leasing it - leasing is the queue engine's transition to make."""
        now = now or utc_now()
        rows = await self._db.query(
            "SELECT * FROM jobs WHERE status = 'queued' "
            "AND (not_before IS NULL OR not_before <= ?) "
            "ORDER BY priority ASC, id ASC LIMIT 50",
            (now.isoformat(),),
        )
        for row in rows:
            job = self._to_job(row)
            if set(job.requires).issubset(capabilities):
                return job
        return None

    async def live_jobs(self) -> list[Job]:
        """Everything queued, leased, running, or gated - the recovery
        scan and the /status count."""
        rows = await self._db.query(
            "SELECT * FROM jobs WHERE status IN "
            "('queued', 'leased', 'running', 'awaiting_approval') "
            "ORDER BY id ASC"
        )
        return [self._to_job(r) for r in rows]

    async def for_session(self, session_id: str, limit: int = 20) -> list[Job]:
        rows = await self._db.query(
            "SELECT * FROM jobs WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        return [self._to_job(r) for r in rows]

    # -- mapping --------------------------------------------------------------

    @staticmethod
    def _encode_field(column: str, value: Any) -> Any:
        """Python value -> storable column value for set_fields baggage."""
        if isinstance(value, (Lease, Approval)):
            return value.model_dump_json()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        return value

    @staticmethod
    def _to_row(job: Job) -> tuple[Any, ...]:
        return (
            job.id, job.type, job.status.value, job.priority,
            json.dumps(job.requires),
            json.dumps(job.payload, ensure_ascii=False, default=str),
            json.dumps(job.result, ensure_ascii=False, default=str)
                if job.result is not None else None,
            job.error,
            json.dumps(job.artifacts),
            job.session_id, job.parent_job_id,
            job.approval.model_dump_json() if job.approval else None,
            json.dumps(job.checkpoint, ensure_ascii=False, default=str)
                if job.checkpoint is not None else None,
            job.attempts, job.max_attempts,
            job.lease.model_dump_json() if job.lease else None,
            job.not_before.isoformat() if job.not_before else None,
            job.created_ts.isoformat(), job.updated_ts.isoformat(),
            job.finished_ts.isoformat() if job.finished_ts else None,
            job.trace_id,
        )

    @staticmethod
    def _to_job(row: aiosqlite.Row) -> Job:
        """Row -> validated Job. The model validator re-checks coherence
        (a leased row must carry a lease, etc.) on every read - corrupted
        state dies here, at the boundary."""
        return Job.model_validate({
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "priority": row["priority"],
            "requires": json.loads(row["requires"]),
            "payload": json.loads(row["payload"]),
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": row["error"],
            "artifacts": json.loads(row["artifacts"]),
            "session_id": row["session_id"],
            "parent_job_id": row["parent_job_id"],
            "approval": json.loads(row["approval"]) if row["approval"] else None,
            "checkpoint": json.loads(row["checkpoint"]) if row["checkpoint"] else None,
            "attempts": row["attempts"],
            "max_attempts": row["max_attempts"],
            "lease": json.loads(row["lease"]) if row["lease"] else None,
            "not_before": row["not_before"],
            "created_ts": row["created_ts"],
            "updated_ts": row["updated_ts"],
            "finished_ts": row["finished_ts"],
            "trace_id": row["trace_id"],
        })