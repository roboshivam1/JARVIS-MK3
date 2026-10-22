-- ============================================================================
-- src/jarvis/core/db/migrations/0017_projects.sql
--
-- Migration 0017: the projects database.
--
-- Everything the owner builds, as records JARVIS can reason over: what a
-- project is, what was decided and why, what it is made of, who is in
-- it, what happened when, and what is still to do. The people database's
-- shape, applied to work: debriefs go in, the model PROPOSES rows, the
-- owner approves them on /app/projects. Nothing the model extracts is
-- stored until he does.
--
-- DECISIONS ARE FIRST-CLASS. "Why did we use the ESP32 and not the Pi"
-- is the question a project log answers worst and this answers best:
-- each decision keeps its reason and the alternatives turned down, and
-- a reversed decision is kept (status) rather than deleted.
--
-- FACTS AND DECISIONS SUPERSEDE, they are not edited in place, so a
-- project's history stays readable - the same rule as person_facts.
--
-- The master document (a patent-style write-up of the whole project) is
-- an artifact; project_documents records each version of it.
-- ============================================================================

CREATE TABLE projects (
    id               TEXT PRIMARY KEY,             -- ULID
    name             TEXT NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'other'
                     CHECK (kind IN ('hardware', 'software', 'mixed', 'research',
                                     'creative', 'other')),
    status           TEXT NOT NULL DEFAULT 'active'
                     CHECK (status IN ('idea', 'active', 'paused', 'shipped',
                                       'abandoned')),
    pitch            TEXT,                          -- one line: what it is
    goal             TEXT,                          -- what "done" means
    description      TEXT,                          -- a paragraph, longer
    repo_url         TEXT,
    folder           TEXT,                          -- where it lives on disk
    started_on       TEXT,                          -- YYYY-MM-DD
    last_activity_ts TEXT,
    created_ts       TEXT NOT NULL,
    updated_ts       TEXT NOT NULL
) WITHOUT ROWID;

CREATE UNIQUE INDEX idx_projects_name ON projects (name COLLATE NOCASE);

CREATE TABLE project_aliases (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    alias      TEXT NOT NULL,
    PRIMARY KEY (project_id, alias)
) WITHOUT ROWID;

CREATE INDEX idx_project_aliases_alias ON project_aliases (alias COLLATE NOCASE);

-- Review queue first: the record tables point back at the debrief each
-- row came from.
CREATE TABLE project_debriefs (
    id         TEXT PRIMARY KEY,                    -- ULID
    raw_text   TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'web',         -- web | conversation | telegram
    status     TEXT NOT NULL DEFAULT 'pending'
               CHECK (status IN ('pending', 'reviewed')),
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE project_proposals (
    id         TEXT PRIMARY KEY,                    -- ULID
    debrief_id TEXT NOT NULL REFERENCES project_debriefs(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,                       -- new_project | alias | decision | ...
    payload    TEXT NOT NULL,                       -- JSON, the proposal as approved
    status     TEXT NOT NULL DEFAULT 'pending'
               CHECK (status IN ('pending', 'approved', 'rejected')),
    result_id  TEXT,                                -- the row approval created
    decided_ts TEXT
) WITHOUT ROWID;

CREATE INDEX idx_project_proposals_debrief ON project_proposals (debrief_id, status);

CREATE TABLE project_decisions (
    id            TEXT PRIMARY KEY,                 -- ULID
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    decision      TEXT NOT NULL,                    -- "Use an ESP32-S3"
    why           TEXT,                             -- the reason, in his words
    alternatives  TEXT,                             -- what was turned down
    decided_on    TEXT,                             -- YYYY-MM-DD
    status        TEXT NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'reversed', 'superseded')),
    superseded_by TEXT,
    debrief_id    TEXT REFERENCES project_debriefs(id),
    created_ts    TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX idx_project_decisions ON project_decisions (project_id, status);

CREATE TABLE project_facts (
    id            TEXT PRIMARY KEY,                 -- ULID
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    category      TEXT NOT NULL DEFAULT 'other'
                  CHECK (category IN ('spec', 'constraint', 'status', 'lesson',
                                      'risk', 'idea', 'other')),
    text          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'superseded', 'retired')),
    superseded_by TEXT,
    debrief_id    TEXT REFERENCES project_debriefs(id),
    created_ts    TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX idx_project_facts ON project_facts (project_id, status);

-- The bill of materials. Hardware mostly; a paid API or a domain is a
-- part too.
CREATE TABLE project_parts (
    id         TEXT PRIMARY KEY,                    -- ULID
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,                       -- "ESP32-S3 DevKitC"
    spec       TEXT,                                -- "N16R8, USB-C"
    quantity   INTEGER NOT NULL DEFAULT 1,
    status     TEXT NOT NULL DEFAULT 'needed'
               CHECK (status IN ('needed', 'ordered', 'have', 'used', 'dropped')),
    source     TEXT,                                -- where from: Robu, Amazon...
    cost_inr   REAL,                                -- for the whole quantity
    notes      TEXT,
    debrief_id TEXT REFERENCES project_debriefs(id),
    created_ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX idx_project_parts ON project_parts (project_id, status);

CREATE TABLE project_links (
    id         TEXT PRIMARY KEY,                    -- ULID
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    label      TEXT NOT NULL,                       -- "datasheet", "CAD", "demo video"
    url        TEXT NOT NULL,
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

-- Who is in it. People live in the people database; this only says
-- what part they play.
CREATE TABLE project_people (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    person_id  TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    role       TEXT,                                -- "mechanical", "mentor", "client"
    PRIMARY KEY (project_id, person_id)
) WITHOUT ROWID;

CREATE TABLE project_relations (
    from_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    to_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind    TEXT NOT NULL
            CHECK (kind IN ('part_of', 'depends_on', 'spun_off_from', 'related')),
    PRIMARY KEY (from_id, to_id, kind)
) WITHOUT ROWID;

-- What happened, dated. The timeline the master document and "where
-- was I on this?" are written from.
CREATE TABLE project_log (
    id         TEXT PRIMARY KEY,                    -- ULID, so it sorts by time
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    happened_on TEXT NOT NULL,                      -- YYYY-MM-DD
    text       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'debrief',     -- debrief | owner | job
    debrief_id TEXT REFERENCES project_debriefs(id),
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX idx_project_log ON project_log (project_id, happened_on);

CREATE TABLE project_tasks (
    id         TEXT PRIMARY KEY,                    -- ULID
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    due_date   TEXT,                                -- YYYY-MM-DD
    status     TEXT NOT NULL DEFAULT 'open'
               CHECK (status IN ('open', 'done', 'dropped')),
    debrief_id TEXT REFERENCES project_debriefs(id),
    created_ts TEXT NOT NULL,
    done_ts    TEXT
) WITHOUT ROWID;

CREATE INDEX idx_project_tasks ON project_tasks (project_id, status);

-- Each version of the master document. The file is an artifact.
CREATE TABLE project_documents (
    id          TEXT PRIMARY KEY,                   -- ULID
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version     INTEGER NOT NULL,
    artifact_id TEXT NOT NULL,
    job_id      TEXT,
    created_ts  TEXT NOT NULL,
    UNIQUE (project_id, version)
) WITHOUT ROWID;

-- Full-text search across everything written about a project. Kept in
-- step by triggers, so no code path can forget it.
CREATE VIRTUAL TABLE projects_fts USING fts5(
    project_id UNINDEXED,
    row_id UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);

CREATE TRIGGER projects_fts_project AFTER INSERT ON projects BEGIN
    INSERT INTO projects_fts (project_id, row_id, text) VALUES
        (new.id, new.id, new.name || ' ' || COALESCE(new.pitch, '') || ' '
                         || COALESCE(new.goal, '') || ' ' || COALESCE(new.description, ''));
END;
CREATE TRIGGER projects_fts_project_upd AFTER UPDATE ON projects BEGIN
    UPDATE projects_fts SET text = new.name || ' ' || COALESCE(new.pitch, '') || ' '
        || COALESCE(new.goal, '') || ' ' || COALESCE(new.description, '')
        WHERE row_id = old.id;
END;
CREATE TRIGGER projects_fts_fact AFTER INSERT ON project_facts BEGIN
    INSERT INTO projects_fts (project_id, row_id, text) VALUES (new.project_id, new.id, new.text);
END;
CREATE TRIGGER projects_fts_decision AFTER INSERT ON project_decisions BEGIN
    INSERT INTO projects_fts (project_id, row_id, text) VALUES
        (new.project_id, new.id, new.decision || ' ' || COALESCE(new.why, '') || ' '
                                 || COALESCE(new.alternatives, ''));
END;
CREATE TRIGGER projects_fts_log AFTER INSERT ON project_log BEGIN
    INSERT INTO projects_fts (project_id, row_id, text) VALUES (new.project_id, new.id, new.text);
END;
CREATE TRIGGER projects_fts_part AFTER INSERT ON project_parts BEGIN
    INSERT INTO projects_fts (project_id, row_id, text) VALUES
        (new.project_id, new.id, new.name || ' ' || COALESCE(new.spec, ''));
END;
CREATE TRIGGER projects_fts_gone AFTER DELETE ON projects BEGIN
    DELETE FROM projects_fts WHERE project_id = old.id;
END;
