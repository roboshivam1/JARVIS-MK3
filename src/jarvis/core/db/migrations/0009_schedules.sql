-- ============================================================================
-- src/jarvis/core/db/migrations/0009_schedules.sql
--
-- Migration 0009: schedules - standing instructions bound to a time.
--
-- Living state: next_fire_ts advances, counters increment, schedules get
-- disabled. The firing HISTORY lives in the event log, which is what
-- makes this table safe to mutate.
--
-- The unique index on name is deliberate: default schedules are seeded
-- at every boot, and the constraint makes that seeding idempotent
-- without any "have I done this already" bookkeeping in code.
-- ============================================================================

CREATE TABLE schedules (
    id            TEXT PRIMARY KEY,     -- ULID
    name          TEXT NOT NULL,        -- human label; unique, see below
    kind          TEXT NOT NULL,        -- cron | interval | once
    cron_expr     TEXT,                 -- when kind = cron
    interval_s    INTEGER,              -- when kind = interval
    job_type      TEXT NOT NULL,        -- what to enqueue on firing
    job_payload   TEXT NOT NULL DEFAULT '{}',
    priority      INTEGER NOT NULL DEFAULT 5,
    enabled       INTEGER NOT NULL DEFAULT 1,
    next_fire_ts  TEXT NOT NULL,        -- UTC ISO-8601; the hot-path column
    last_fired_ts TEXT,
    fire_count    INTEGER NOT NULL DEFAULT 0,
    created_ts    TEXT NOT NULL
) WITHOUT ROWID;

-- The engine's only frequent question: what is due? Partial index keeps
-- it to enabled rows.
CREATE INDEX idx_schedules_due ON schedules (next_fire_ts) WHERE enabled = 1;

-- Makes boot-time seeding of default schedules idempotent.
CREATE UNIQUE INDEX idx_schedules_name ON schedules (name);