-- ============================================================================
-- src/jarvis/core/db/migrations/0001_init.sql
--
-- Migration 0001: bookkeeping table + the events table (the append-only
-- spine of the system).
--
-- Migration discipline: files are numbered, applied in order, and NEVER
-- edited after being applied anywhere. Schema changes are new files.
-- ============================================================================

-- Which migrations have been applied to this database file.
CREATE TABLE schema_migrations (
    version     INTEGER PRIMARY KEY,   -- the file's number prefix (1, 2, ...)
    name        TEXT    NOT NULL,      -- the file name, for humans
    applied_ts  TEXT    NOT NULL       -- UTC ISO-8601
);

-- The event log. One row per Event model instance. Append-only: enforced
-- by triggers below, not just by convention.
--
-- WITHOUT ROWID: the ULID id becomes the physical storage key. ULIDs are
-- time-ordered, so inserts always land at the right edge of the B-tree
-- (true append behaviour) and ORDER BY id is chronological order with no
-- extra index.
CREATE TABLE events (
    id          TEXT PRIMARY KEY,      -- ULID, 26 chars
    ts          TEXT NOT NULL,         -- UTC ISO-8601
    kind        TEXT NOT NULL,         -- closed taxonomy, validated in code
    source      TEXT NOT NULL,         -- e.g. core.app, worker.macbook
    session_id  TEXT,                  -- ULID or NULL
    job_id      TEXT,                  -- ULID or NULL
    trace_id    TEXT NOT NULL,         -- ULID; causal correlation
    payload     TEXT NOT NULL          -- JSON object as text
) WITHOUT ROWID;

-- Query patterns we know exist today:
--   1. Reconstruct one causal chain: WHERE trace_id = ?
--   2. Recent events of a kind:      WHERE kind = ? ORDER BY id DESC
-- The (kind, id) compound index serves pattern 2 as a pure index walk.
-- Indexes for session_id / job_id are deferred until the code that
-- queries by them exists - an unused index is a pure write tax.
CREATE INDEX idx_events_trace ON events (trace_id);
CREATE INDEX idx_events_kind  ON events (kind, id);

-- The append-only rule, enforced by the database itself. Any UPDATE or
-- DELETE attempt fails loudly, no matter what future code tries.
CREATE TRIGGER events_no_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only: UPDATE forbidden');
END;

CREATE TRIGGER events_no_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only: DELETE forbidden');
END;