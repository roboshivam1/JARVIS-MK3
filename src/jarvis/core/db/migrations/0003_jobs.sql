-- ============================================================================
-- src/jarvis/core/db/migrations/0003_jobs.sql
--
-- Migration 0003: the jobs table - the durable work queue.
--
-- Jobs are LIVING STATE (statuses change, leases move, checkpoints
-- update), so no append-only triggers; history lives in the job.* events
-- that accompany every transition. Structured sub-objects (lease,
-- approval, checkpoint) and lists (requires, artifacts) are stored as
-- JSON text, validated through the Job model at the repo boundary.
-- ============================================================================

CREATE TABLE jobs (
    id            TEXT PRIMARY KEY,     -- ULID; time-ordered = age-ordered
    type          TEXT NOT NULL,        -- registered job type, e.g. research.brief
    status        TEXT NOT NULL DEFAULT 'queued',
    priority      INTEGER NOT NULL DEFAULT 5,   -- 0 urgent .. 9 idle
    requires      TEXT NOT NULL DEFAULT '[]',   -- JSON list of capability tags
    payload       TEXT NOT NULL DEFAULT '{}',   -- JSON, job-type input model
    result        TEXT,                 -- JSON, job-type output model
    error         TEXT,
    artifacts     TEXT NOT NULL DEFAULT '[]',   -- JSON list of artifact ids
    session_id    TEXT,                 -- conversation to attach results to
    parent_job_id TEXT,                 -- job chaining
    approval      TEXT,                 -- JSON Approval or NULL
    checkpoint    TEXT,                 -- JSON resume state or NULL
    attempts      INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 3,
    lease         TEXT,                 -- JSON Lease or NULL
    not_before    TEXT,                 -- UTC ISO-8601; retry backoff / scheduling
    created_ts    TEXT NOT NULL,
    updated_ts    TEXT NOT NULL,
    finished_ts   TEXT,
    trace_id      TEXT NOT NULL
) WITHOUT ROWID;

-- The dispatcher's question, asked constantly: "oldest, most urgent
-- queued job". Partial index: only queued rows are indexed, so the index
-- stays tiny no matter how much terminal history accumulates.
CREATE INDEX idx_jobs_dispatch ON jobs (priority, id) WHERE status = 'queued';

-- Live-work views: everything currently leased or running (recovery
-- scans this at boot; /status counts it).
CREATE INDEX idx_jobs_status ON jobs (status) WHERE status IN
    ('queued', 'leased', 'running', 'awaiting_approval');

-- "jobs of this conversation" for session attachment.
CREATE INDEX idx_jobs_session ON jobs (session_id) WHERE session_id IS NOT NULL;