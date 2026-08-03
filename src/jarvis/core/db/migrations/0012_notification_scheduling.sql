-- ============================================================================
-- src/jarvis/core/db/migrations/0012_notification_scheduling.sql
--
-- Migration 0012: notifications can be held rather than sent.
--
-- not_before is how "deferred, never dropped" is implemented: a message
-- held during quiet hours keeps its pending status and simply becomes
-- invisible to the delivery loop until the moment arrives. Nothing is
-- discarded, so the owner can trust that silence means nothing
-- happened rather than something was withheld.
--
-- digest_of records which notifications a digest replaced, so the
-- individual entries can be traced back from the summary that was
-- actually delivered.
-- ============================================================================

ALTER TABLE notifications ADD COLUMN not_before TEXT;
ALTER TABLE notifications ADD COLUMN digest_of TEXT;

-- The delivery loop's question becomes "what is pending AND due", so
-- the partial index covers both.
DROP INDEX IF EXISTS idx_notifications_pending;
CREATE INDEX idx_notifications_pending
    ON notifications (not_before, id) WHERE status = 'pending';
