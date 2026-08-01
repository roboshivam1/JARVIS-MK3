-- ============================================================================
-- src/jarvis/core/db/migrations/0010_approvals.sql
--
-- Migration 0010: approval requests - the record of what JARVIS asked
-- permission for, and what the owner said.
--
-- The job's own `approval` column holds the CURRENT pending gate (that
-- is what the model validator enforces). This table is the HISTORY:
-- every request ever raised, with its outcome. History is what makes
-- graduated autonomy earnable - in three months, "should this gate be
-- promoted?" is answered from these rows rather than from a hunch.
--
-- Deliberately not append-only: a pending row is updated once, when the
-- owner answers. The decision itself also lands in the event log, which
-- is the immutable record.
-- ============================================================================

CREATE TABLE approvals (
    id           TEXT PRIMARY KEY,     -- ULID
    job_id       TEXT NOT NULL,        -- the paused job
    gate         TEXT NOT NULL,        -- outbound | credential | ...
    actor        TEXT NOT NULL,        -- which agent asked
    tool         TEXT NOT NULL,        -- what it wanted to do
    summary      TEXT NOT NULL,        -- one line for the owner
    detail       TEXT NOT NULL,        -- the EXACT action, in full
    risk_note    TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected|expired
    decision_note TEXT,                -- the owner's optional reason
    requested_ts TEXT NOT NULL,
    expires_ts   TEXT NOT NULL,        -- unanswered past this = expired
    resolved_ts  TEXT,
    trace_id     TEXT NOT NULL
) WITHOUT ROWID;

-- The two questions asked constantly: what is waiting, and has this job
-- got something pending?
CREATE INDEX idx_approvals_pending ON approvals (expires_ts) WHERE status = 'pending';
CREATE INDEX idx_approvals_job ON approvals (job_id);