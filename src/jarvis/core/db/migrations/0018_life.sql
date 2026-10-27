-- ============================================================================
-- src/jarvis/core/db/migrations/0018_life.sql
--
-- Migration 0018: the life log.
--
-- What the owner's days actually look like, as data: when he woke and
-- slept, what he did (gym, study, project work, learning), whether he ate,
-- and a daily checklist of what he meant to do - fixed targets that
-- recur (gym, Monday to Saturday) and one-off targets set for a day
-- ("finish one Coursera module tomorrow").
--
-- Unlike people and projects there is NO review queue. This is his own
-- record of his own day; a report goes straight in, and every row keeps
-- the id of the report that wrote it so "undo that" removes exactly
-- what one message created.
--
-- Dates are the owner's LOCAL calendar date (YYYY-MM-DD): "today" means
-- his day in Jaipur, not a UTC one. Times of day are stored as local HH:MM.
-- ============================================================================

-- One message's worth of logging, kept so it can be undone as a unit.
CREATE TABLE life_reports (
    id         TEXT PRIMARY KEY,                  -- ULID
    raw_text   TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'conversation', -- conversation | web | shortcut
    summary    TEXT NOT NULL DEFAULT '',          -- what was recorded, in words
    undone     INTEGER NOT NULL DEFAULT 0,
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE life_days (
    date       TEXT PRIMARY KEY,                  -- YYYY-MM-DD, local
    wake_time  TEXT,                              -- HH:MM, local
    sleep_time TEXT,                              -- HH:MM, local; may be after midnight
    note       TEXT,                              -- how the day went, his words
    wake_report  TEXT,                            -- which report set each field,
    sleep_report TEXT,                            -- so undo can clear exactly that
    note_report  TEXT,
    updated_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE life_activities (
    id         TEXT PRIMARY KEY,                  -- ULID
    date       TEXT NOT NULL,                     -- local date it happened on
    kind       TEXT NOT NULL
               CHECK (kind IN ('gym', 'study', 'project', 'learning', 'other')),
    subject    TEXT,                              -- "OS", "legs", "Coursera agentic AI"
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    started_at TEXT,                              -- HH:MM, local
    ended_at   TEXT,                              -- HH:MM; NULL while running
    minutes    INTEGER,                           -- known or computed duration
    note       TEXT,
    report_id  TEXT REFERENCES life_reports(id),
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX idx_life_activities_date ON life_activities (date, kind);

CREATE TABLE life_meals (
    id         TEXT PRIMARY KEY,                  -- ULID
    date       TEXT NOT NULL,
    slot       TEXT NOT NULL
               CHECK (slot IN ('breakfast', 'lunch', 'dinner', 'snack')),
    status     TEXT NOT NULL DEFAULT 'eaten' CHECK (status IN ('eaten', 'skipped')),
    what       TEXT,
    place      TEXT,                              -- home, mess, out
    report_id  TEXT REFERENCES life_reports(id),
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX idx_life_meals_date ON life_meals (date);

-- Fixed targets: things that belong on the checklist on given weekdays.
-- An activity of the same kind on that day ticks the item by itself.
CREATE TABLE life_habits (
    id         TEXT PRIMARY KEY,                  -- ULID
    name       TEXT NOT NULL,                     -- "Gym"
    kind       TEXT CHECK (kind IN ('gym', 'study', 'project', 'learning', 'other')),
    weekdays   TEXT NOT NULL DEFAULT 'mon,tue,wed,thu,fri,sat,sun',
    active     INTEGER NOT NULL DEFAULT 1,
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE UNIQUE INDEX idx_life_habits_name ON life_habits (name COLLATE NOCASE);

-- The day's checklist: fixed items copied in from habits, one-off items
-- added for that day. Past days' open items become 'missed'.
CREATE TABLE life_checklist (
    id         TEXT PRIMARY KEY,                  -- ULID
    date       TEXT NOT NULL,
    text       TEXT NOT NULL,
    habit_id   TEXT REFERENCES life_habits(id) ON DELETE SET NULL,
    status     TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done', 'missed')),
    done_ts    TEXT,
    report_id  TEXT REFERENCES life_reports(id),  -- the report that added it
    done_report TEXT,                             -- the report that ticked it
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX idx_life_checklist_date ON life_checklist (date, status);
CREATE UNIQUE INDEX idx_life_checklist_habit ON life_checklist (date, habit_id)
    WHERE habit_id IS NOT NULL;

-- Small owner settings: the wake-by target, for now.
CREATE TABLE life_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

INSERT INTO life_habits (id, name, kind, weekdays, active, created_ts)
VALUES ('01JGYM00000000000000000000', 'Gym', 'gym', 'mon,tue,wed,thu,fri,sat', 1,
        strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now'));
