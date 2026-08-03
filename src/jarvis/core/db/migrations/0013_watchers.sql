-- ============================================================================
-- src/jarvis/core/db/migrations/0013_watchers.sql
--
-- Migration 0013: watchers - recurring checks that remember.
--
-- The state column is what separates a watcher from a schedule: it
-- holds what the last check saw, so the next one can ask whether
-- anything changed rather than reporting unconditionally.
--
-- Living state, mutated after every check. The HISTORY of what was
-- noticed lives in the event log, as always.
-- ============================================================================

CREATE TABLE watchers (
    id              TEXT PRIMARY KEY,   -- ULID
    name            TEXT NOT NULL,      -- unique human label
    kind            TEXT NOT NULL,      -- web_page | job_health | spend | idle
    config          TEXT NOT NULL DEFAULT '{}',   -- JSON, what to check
    state           TEXT NOT NULL DEFAULT '{}',   -- JSON, what was last seen
    priority        INTEGER NOT NULL DEFAULT 5,
    note            TEXT NOT NULL DEFAULT '',
    enabled         INTEGER NOT NULL DEFAULT 1,
    hit_count       INTEGER NOT NULL DEFAULT 0,
    last_checked_ts TEXT,
    last_hit_ts     TEXT,
    created_ts      TEXT NOT NULL
) WITHOUT ROWID;

CREATE UNIQUE INDEX idx_watchers_name ON watchers (name);
CREATE INDEX idx_watchers_enabled ON watchers (id) WHERE enabled = 1;
