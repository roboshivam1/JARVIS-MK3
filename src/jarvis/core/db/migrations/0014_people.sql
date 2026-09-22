-- ============================================================================
-- src/jarvis/core/db/migrations/0014_people.sql
--
-- Migration 0014: the people database - who the owner knows, what is true
-- about them, and what is owed in either direction.
--
-- Separate from the facts vault on purpose. A vault fact is one
-- free-standing sentence; a person is an ENTITY that facts, commitments,
-- and dates hang off. Tables are prefixed person_/people_ because `facts`
-- already belongs to the vault.
--
-- Two halves:
--   people, person_aliases, person_facts, person_commitments, person_dates
--       the record, written ONLY when the owner approves a proposal
--   people_debriefs, people_proposals
--       the staging area, the only place the extraction model writes
--
-- Person facts follow the vault's rule: never deleted, only superseded,
-- so the history of what was believed survives.
-- ============================================================================

CREATE TABLE people (
    id                TEXT PRIMARY KEY,   -- ULID
    name              TEXT NOT NULL,
    relationship      TEXT,               -- friend, family, mentor, colleague, client...
    closeness         INTEGER NOT NULL DEFAULT 2
                      CHECK (closeness BETWEEN 1 AND 3),   -- 1 close .. 3 acquaintance
    how_met           TEXT,
    summary           TEXT,               -- one line: who they are to the owner
    status            TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('active', 'dormant', 'lost_touch')),
    last_mentioned_ts TEXT,
    created_ts        TEXT NOT NULL,
    updated_ts        TEXT NOT NULL
) WITHOUT ROWID;

-- Every way the owner refers to someone. NOCASE: "aarav" finds "Aarav".
CREATE TABLE person_aliases (
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    alias     TEXT NOT NULL COLLATE NOCASE,
    PRIMARY KEY (person_id, alias)
) WITHOUT ROWID;

CREATE TABLE people_debriefs (
    id         TEXT PRIMARY KEY,          -- ULID
    raw_text   TEXT NOT NULL,             -- exactly what the owner said
    source     TEXT NOT NULL,             -- web | conversation
    status     TEXT NOT NULL DEFAULT 'pending'
               CHECK (status IN ('pending', 'reviewed')),
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE person_facts (
    id            TEXT PRIMARY KEY,       -- ULID
    person_id     TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    category      TEXT NOT NULL,
    text          TEXT NOT NULL,
    sensitivity   TEXT NOT NULL DEFAULT 'normal'
                  CHECK (sensitivity IN ('normal', 'sensitive')),
    status        TEXT NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'superseded')),
    superseded_by TEXT REFERENCES person_facts(id),
    debrief_id    TEXT REFERENCES people_debriefs(id),
    created_ts    TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE person_commitments (
    id          TEXT PRIMARY KEY,         -- ULID
    person_id   TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    direction   TEXT NOT NULL CHECK (direction IN ('i_owe', 'they_owe')),
    what        TEXT NOT NULL,
    due_date    TEXT,                     -- YYYY-MM-DD or NULL
    status      TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'done', 'dropped')),
    debrief_id  TEXT REFERENCES people_debriefs(id),
    created_ts  TEXT NOT NULL,
    resolved_ts TEXT
) WITHOUT ROWID;

CREATE TABLE person_dates (
    id         TEXT PRIMARY KEY,          -- ULID
    person_id  TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    label      TEXT NOT NULL,             -- "birthday", "end-sem exams"
    date       TEXT NOT NULL,             -- MM-DD if recurring, else YYYY-MM-DD
    recurring  INTEGER NOT NULL DEFAULT 1,
    debrief_id TEXT REFERENCES people_debriefs(id),
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE people_proposals (
    id         TEXT PRIMARY KEY,          -- ULID
    debrief_id TEXT NOT NULL REFERENCES people_debriefs(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL
               CHECK (kind IN ('new_person', 'alias', 'fact', 'commitment', 'key_date')),
    payload    TEXT NOT NULL,             -- JSON, validated by common/people.py
    status     TEXT NOT NULL DEFAULT 'pending'
               CHECK (status IN ('pending', 'approved', 'rejected')),
    result_id  TEXT,                      -- the row created on approval
    decided_ts TEXT
) WITHOUT ROWID;

CREATE INDEX idx_person_facts_person       ON person_facts (person_id, status);
CREATE INDEX idx_person_commitments_person ON person_commitments (person_id, status);
CREATE INDEX idx_person_dates_person       ON person_dates (person_id);
CREATE INDEX idx_people_proposals_debrief  ON people_proposals (debrief_id, status);
CREATE INDEX idx_people_debriefs_status    ON people_debriefs (status, created_ts);
