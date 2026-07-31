-- ============================================================================
-- src/jarvis/core/db/migrations/0004_artifacts.sql
--
-- Migration 0004: the artifacts catalogue.
--
-- Rows here; bytes on disk under data/artifacts/<id>/<name>. The two are
-- written together by the repository: file first, then row, so a crash
-- can leave an orphan FILE (harmless, sweepable) but never a row that
-- promises a file which does not exist.
-- ============================================================================

CREATE TABLE artifacts (
    id           TEXT PRIMARY KEY,    -- ULID
    name         TEXT NOT NULL,       -- human filename
    mime         TEXT NOT NULL,
    size         INTEGER NOT NULL,
    sha256       TEXT NOT NULL,       -- hex digest, for verification
    storage_path TEXT NOT NULL,       -- relative to the artifact root
    created_by   TEXT NOT NULL,       -- job id / turn id / client.upload
    ts           TEXT NOT NULL
) WITHOUT ROWID;

-- "what did this job produce" - the delivery path's question.
CREATE INDEX idx_artifacts_creator ON artifacts (created_by);