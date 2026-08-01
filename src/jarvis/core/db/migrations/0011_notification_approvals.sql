-- ============================================================================
-- src/jarvis/core/db/migrations/0011_notification_approvals.sql
--
-- Migration 0011: notifications can carry an approval request.
--
-- Approvals reuse the notification outbox rather than getting a delivery
-- path of their own: the outbox already handles durability, retry on a
-- failed send, and recording what was said as part of the conversation.
-- A parallel path would have been a second thing to keep correct.
--
-- The unique index enforces one notification per approval, the same way
-- job_id does for job results - a crash mid-scan cannot produce two
-- copies of the same question.
-- ============================================================================

ALTER TABLE notifications ADD COLUMN approval_id TEXT;

CREATE UNIQUE INDEX idx_notifications_approval ON notifications (approval_id)
    WHERE approval_id IS NOT NULL;