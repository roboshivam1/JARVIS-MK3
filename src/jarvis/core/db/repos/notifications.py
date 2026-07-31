# =============================================================================
# src/jarvis/core/db/repos/notifications.py - the notification outbox
# =============================================================================
#
# The only code touching the notifications table.
#
# create() tolerates the unique-index rejection deliberately: a duplicate
# means another scan already queued this job's notification, which is the
# constraint doing its job, not an error to escalate.
# =============================================================================

from __future__ import annotations

import aiosqlite

from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.common.notifications import Notification, NotificationStatus
from jarvis.core.db.database import Database

log = get_logger("core.db.notifications")


class NotificationsRepo:
    """Queue, list, and settle unprompted messages."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(self, notification: Notification) -> bool:
        """Queue a notification. Returns False if one already exists for
        this job - the unique index refusing a duplicate, which is the
        rule working rather than a failure."""
        try:
            await self._db.execute(
                "INSERT INTO notifications "
                "(id, ts, priority, status, client_kind, text, session_id, "
                " job_id, artifact_id, delivered_ts, suppress_reason, trace_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    notification.id, notification.ts.isoformat(),
                    notification.priority, notification.status.value,
                    notification.client_kind, notification.text,
                    notification.session_id, notification.job_id,
                    notification.artifact_id,
                    notification.delivered_ts.isoformat()
                        if notification.delivered_ts else None,
                    notification.suppress_reason, notification.trace_id,
                ),
            )
            return True
        except aiosqlite.IntegrityError:
            return False

    async def pending(self, limit: int = 20) -> list[Notification]:
        """Undelivered notifications, oldest first."""
        rows = await self._db.query(
            "SELECT * FROM notifications WHERE status = ? ORDER BY id ASC LIMIT ?",
            (NotificationStatus.PENDING.value, limit),
        )
        return [self._to_notification(r) for r in rows]

    async def mark_delivered(self, notification_id: str) -> None:
        await self._db.execute(
            "UPDATE notifications SET status = ?, delivered_ts = ? WHERE id = ?",
            (
                NotificationStatus.DELIVERED.value,
                utc_now().isoformat(),
                notification_id,
            ),
        )

    async def mark_suppressed(self, notification_id: str, reason: str) -> None:
        await self._db.execute(
            "UPDATE notifications SET status = ?, suppress_reason = ? WHERE id = ?",
            (NotificationStatus.SUPPRESSED.value, reason, notification_id),
        )

    async def job_ids_awaiting_notification(self, limit: int = 20) -> list[str]:
        """Finished jobs that belong to a conversation and have no
        notification yet. Cancelled jobs are excluded: the owner
        cancelled them and does not need telling."""
        rows = await self._db.query(
            "SELECT j.id AS job_id FROM jobs j "
            "WHERE j.session_id IS NOT NULL "
            "AND j.status IN ('succeeded', 'failed') "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM notifications n WHERE n.job_id = j.id"
            ") ORDER BY j.id ASC LIMIT ?",
            (limit,),
        )
        return [str(r["job_id"]) for r in rows]

    @staticmethod
    def _to_notification(row: aiosqlite.Row) -> Notification:
        return Notification.model_validate({
            "id": row["id"],
            "ts": row["ts"],
            "priority": row["priority"],
            "status": row["status"],
            "client_kind": row["client_kind"],
            "text": row["text"],
            "session_id": row["session_id"],
            "job_id": row["job_id"],
            "artifact_id": row["artifact_id"],
            "delivered_ts": row["delivered_ts"],
            "suppress_reason": row["suppress_reason"],
            "trace_id": row["trace_id"],
        })