-- ============================================================================
-- src/jarvis/core/db/migrations/0007_facts_fts_stemming.sql
--
-- Migration 0007: rebuild the keyword index with stemming.
--
-- Default FTS5 matches tokens exactly, so a search for "studying" misses
-- a fact that says "studies". The porter tokenizer reduces both to the
-- same root ("studi"), which is the difference between the keyword path
-- helping and the keyword path sitting out.
--
-- A virtual table's tokenizer cannot be altered, so the table is dropped
-- and rebuilt from the facts table - which is safe precisely because the
-- index is derived data, never a source of truth.
-- ============================================================================

DROP TRIGGER facts_fts_insert;
DROP TRIGGER facts_fts_update;
DROP TRIGGER facts_fts_delete;
DROP TABLE facts_fts;

CREATE VIRTUAL TABLE facts_fts USING fts5(
    fact_id UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);

-- Repopulate from the source of truth.
INSERT INTO facts_fts (fact_id, text) SELECT id, text FROM facts;

CREATE TRIGGER facts_fts_insert AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts (fact_id, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER facts_fts_update AFTER UPDATE OF text ON facts BEGIN
    UPDATE facts_fts SET text = new.text WHERE fact_id = old.id;
END;

CREATE TRIGGER facts_fts_delete AFTER DELETE ON facts BEGIN
    DELETE FROM facts_fts WHERE fact_id = old.id;
END;