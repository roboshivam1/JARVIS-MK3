# =============================================================================
# src/jarvis/core/db/repos/approvals.py - the approval ledger
# =============================================================================
#
# The only code touching the approvals table.
#
# resolve() uses an optimistic check on status: a decision only lands if
# the request is still pending. That is what makes the race between "the
# owner taps approve" and "the request expires" resolve exactly once,
# rather than an expiry sweep silently overwriting a decision the owner
# already made.
# =============================================================================

from __future__ import annotations

from datetime import datetime

import aiosqlite

from jarvis.common.approvals import ApprovalRequest, ApprovalStatus
from jarvis.common.capabilities import Gate
from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.core.db.database import Database

log = get_logger("core.db.approvals")


class ApprovalsRepo:
    """Raise, answer, and expire approval requests."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(self, request: ApprovalRequest) -> None:
        await self._db.execute(
            "INSERT INTO approvals "
            "(id, job_id, gate, actor, tool, summary, detail, risk_note, "
            " status, decision_note, requested_ts, expires_ts, resolved_ts, "
            " trace_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request.id, request.job_id, request.gate.value, request.actor,
                request.tool, request.summary, request.detail, request.risk_note,
                request.status.value, request.decision_note,
                request.requested_ts.isoformat(), request.expires_ts.isoformat(),
                request.resolved_ts.isoformat() if request.resolved_ts else None,
                request.trace_id,
            ),
        )
        log.info("approval requested", extra={
            "approval_id": request.id, "job_id": request.job_id,
            "gate": request.gate.value, "tool": request.tool,
        })

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        row = await self._db.query_one(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        )
        return self._to_request(row) if row else None

    async def pending_for_job(self, job_id: str) -> ApprovalRequest | None:
        row = await self._db.query_one(
            "SELECT * FROM approvals WHERE job_id = ? AND status = 'pending' "
            "ORDER BY id DESC LIMIT 1",
            (job_id,),
        )
        return self._to_request(row) if row else None

    async def all_pending(self) -> list[ApprovalRequest]:
        rows = await self._db.query(
            "SELECT * FROM approvals WHERE status = 'pending' ORDER BY id ASC"
        )
        return [self._to_request(r) for r in rows]

    async def resolve(
        self,
        approval_id: str,
        status: ApprovalStatus,
        note: str | None = None,
    ) -> bool:
        """Record a decision. Returns False if the request was already
        settled - the owner's tap and the expiry sweep can race, and
        exactly one of them must win."""
        changed = await self._db.execute_returning_changes(
            "UPDATE approvals SET status = ?, decision_note = ?, "
            "resolved_ts = ? WHERE id = ? AND status = 'pending'",
            (status.value, note, utc_now().isoformat(), approval_id),
        )
        return changed == 1

    async def expired(self, now: datetime | None = None) -> list[ApprovalRequest]:
        """Pending requests past their deadline."""
        moment = (now or utc_now()).isoformat()
        rows = await self._db.query(
            "SELECT * FROM approvals WHERE status = 'pending' AND expires_ts <= ?",
            (moment,),
        )
        return [self._to_request(r) for r in rows]

    async def history_for_gate(
        self, actor: str, gate: Gate, limit: int = 50
    ) -> list[ApprovalRequest]:
        """Past decisions for one actor and gate - the evidence behind a
        decision to grant more autonomy."""
        rows = await self._db.query(
            "SELECT * FROM approvals WHERE actor = ? AND gate = ? "
            "AND status IN ('approved', 'rejected') ORDER BY id DESC LIMIT ?",
            (actor, gate.value, limit),
        )
        return [self._to_request(r) for r in rows]

    @staticmethod
    def _to_request(row: aiosqlite.Row) -> ApprovalRequest:
        return ApprovalRequest.model_validate({
            "id": row["id"],
            "job_id": row["job_id"],
            "gate": row["gate"],
            "actor": row["actor"],
            "tool": row["tool"],
            "summary": row["summary"],
            "detail": row["detail"],
            "risk_note": row["risk_note"],
            "status": row["status"],
            "decision_note": row["decision_note"],
            "requested_ts": row["requested_ts"],
            "expires_ts": row["expires_ts"],
            "resolved_ts": row["resolved_ts"],
            "trace_id": row["trace_id"],
        })