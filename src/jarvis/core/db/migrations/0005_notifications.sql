-- ============================================================================
-- src/jarvis/core/db/migrations/0005_notifications.sql
--
-- Migration 0005: the notification outbox.
--
-- Notifications are created as pending rows and delivered separately, so
-- a send that fails is a retryable row rather than a lost message.
--
-- The unique index on job_id is load-bearing: it makes "one notification
-- per job" a rule the DATABASE enforces, so a crash mid-scan cannot
-- produce duplicates and the scanner needs no bookkeeping of its own.
-- ============================================================================

CREATE TABLE notifications (
    id              TEXT PRIMARY KEY,   -- ULID
    ts              TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 5,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending|delivered|suppressed
    client_kind     TEXT NOT NULL,      -- telegram, voice.mac, ...
    text            TEXT NOT NULL,
    session_id      TEXT,
    job_id          TEXT,
    artifact_id     TEXT,
    delivered_ts    TEXT,
    suppress_reason TEXT,
    trace_id        TEXT NOT NULL
) WITHOUT ROWID;

-- The delivery loop's question: what still needs sending?
CREATE INDEX idx_notifications_pending ON notifications (id) WHERE status = 'pending';

-- One notification per job, enforced by the database itself.
CREATE UNIQUE INDEX idx_notifications_job ON notifications (job_id)
    WHERE job_id IS NOT NULL;