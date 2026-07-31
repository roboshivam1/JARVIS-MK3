-- ============================================================================
-- src/jarvis/core/db/migrations/0006_facts.sql
--
-- Migration 0006: semantic memory - the facts table, its keyword search
-- index, and the triggers that keep them in sync.
--
-- Two search paths share one table:
--   keyword - the FTS5 virtual table below (built into SQLite, no
--             extension needed)
--   vector  - the embedding BLOB column, compared in Python
--
-- The vector column is a plain BLOB of packed float32 rather than a
-- sqlite-vec virtual table, by owner-approved amendment: no loadable C
-- extension to fail on a fresh VPS, and brute-force comparison is
-- milliseconds at personal scale. Revisit if fact count ever reaches
-- six figures.
-- ============================================================================

CREATE TABLE facts (
    id               TEXT PRIMARY KEY,   -- ULID
    text             TEXT NOT NULL,      -- one self-contained sentence
    category         TEXT NOT NULL DEFAULT 'other',
    importance       REAL NOT NULL DEFAULT 0.5,
    confidence       REAL NOT NULL DEFAULT 0.8,
    status           TEXT NOT NULL DEFAULT 'active',  -- active|superseded|expired
    supersedes       TEXT,               -- fact id this one replaced
    source_event_ids TEXT NOT NULL DEFAULT '[]',      -- JSON list, provenance
    created_ts       TEXT NOT NULL,
    last_accessed_ts TEXT NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    embedder_version TEXT,               -- NULL until embedded
    embedding        BLOB                -- packed float32 vector, NULL until embedded
) WITHOUT ROWID;

-- Retrieval only ever considers active facts; the partial index keeps
-- that scan tight no matter how much superseded history accumulates.
CREATE INDEX idx_facts_active ON facts (importance DESC) WHERE status = 'active';

-- The sleep cycle's question: which facts need embedding or re-embedding?
CREATE INDEX idx_facts_embedder ON facts (embedder_version) WHERE status = 'active';

-- Keyword search. Standalone FTS5 table (not external-content, which
-- needs an integer rowid our WITHOUT ROWID table does not have).
CREATE VIRTUAL TABLE facts_fts USING fts5(
    fact_id UNINDEXED,
    text
);

-- The index maintains itself. Same discipline as the append-only event
-- log: make the database enforce the rule so no future code can forget.
CREATE TRIGGER facts_fts_insert AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts (fact_id, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER facts_fts_update AFTER UPDATE OF text ON facts BEGIN
    UPDATE facts_fts SET text = new.text WHERE fact_id = old.id;
END;

CREATE TRIGGER facts_fts_delete AFTER DELETE ON facts BEGIN
    DELETE FROM facts_fts WHERE fact_id = old.id;
END;