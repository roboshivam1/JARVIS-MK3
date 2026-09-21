-- people/schema.sql — People database (v1)
-- Idempotent: safe to run on every boot.

CREATE TABLE IF NOT EXISTS people (
    id             INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL,
    relationship   TEXT,                       -- friend, family, mentor, colleague, client...
    closeness      INTEGER NOT NULL DEFAULT 2  -- 1 close, 2 regular, 3 acquaintance
                   CHECK (closeness BETWEEN 1 AND 3),
    how_met        TEXT,
    summary        TEXT,                       -- one line: who they are to Shivam
    status         TEXT    NOT NULL DEFAULT 'active'
                   CHECK (status IN ('active', 'dormant', 'lost_touch')),
    last_mentioned TEXT,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Every way Shivam refers to someone. NOCASE makes "aarav" match "Aarav".
CREATE TABLE IF NOT EXISTS aliases (
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    alias     TEXT    NOT NULL COLLATE NOCASE,
    PRIMARY KEY (person_id, alias)
);

CREATE TABLE IF NOT EXISTS debriefs (
    id         INTEGER PRIMARY KEY,
    raw_text   TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Facts are never overwritten. A changed fact supersedes the old one,
-- so history survives and there's only ever one *active* version.
CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY,
    person_id     INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    category      TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    sensitivity   TEXT    NOT NULL DEFAULT 'normal'
                  CHECK (sensitivity IN ('normal', 'sensitive')),
    status        TEXT    NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'superseded')),
    superseded_by INTEGER REFERENCES facts(id),
    debrief_id    INTEGER REFERENCES debriefs(id),
    learned_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS commitments (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    direction   TEXT    NOT NULL CHECK (direction IN ('i_owe', 'they_owe')),
    what        TEXT    NOT NULL,
    due_date    TEXT,                          -- YYYY-MM-DD or NULL
    status      TEXT    NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'done', 'dropped')),
    debrief_id  INTEGER REFERENCES debriefs(id),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS key_dates (
    id         INTEGER PRIMARY KEY,
    person_id  INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    label      TEXT    NOT NULL,               -- "birthday", "end-sem exams"
    date       TEXT    NOT NULL,               -- MM-DD if recurring, else YYYY-MM-DD
    recurring  INTEGER NOT NULL DEFAULT 1,
    debrief_id INTEGER REFERENCES debriefs(id)
);

-- The staging area. The LLM only ever writes here — nothing reaches the
-- tables above except through ReviewQueue.approve().
CREATE TABLE IF NOT EXISTS proposals (
    id         INTEGER PRIMARY KEY,
    debrief_id INTEGER NOT NULL REFERENCES debriefs(id) ON DELETE CASCADE,
    kind       TEXT    NOT NULL
               CHECK (kind IN ('new_person', 'alias', 'fact', 'commitment', 'key_date')),
    payload    TEXT    NOT NULL,               -- JSON, validated by models.py
    status     TEXT    NOT NULL DEFAULT 'pending'
               CHECK (status IN ('pending', 'approved', 'rejected')),
    result_id  INTEGER,                        -- row created on approval
    decided_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_person       ON facts(person_id, status);
CREATE INDEX IF NOT EXISTS idx_commitments_person ON commitments(person_id, status);
CREATE INDEX IF NOT EXISTS idx_proposals_debrief  ON proposals(debrief_id, status);
