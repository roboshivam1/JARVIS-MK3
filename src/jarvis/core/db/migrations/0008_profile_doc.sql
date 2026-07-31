-- ============================================================================
-- src/jarvis/core/db/migrations/0008_profile_doc.sql
--
-- Migration 0008: the profile document - a short curated page about the
-- owner, injected into every conversation.
--
-- Versioned by append: each rewrite is a NEW row, and the newest (by
-- ULID, which sorts by time) is current. Nothing is ever overwritten, so
-- a bad automated rewrite can be inspected and reverted rather than
-- silently losing what it replaced.
-- ============================================================================

CREATE TABLE profile_doc (
    id           TEXT PRIMARY KEY,   -- ULID; newest row is the live profile
    content      TEXT NOT NULL,      -- markdown, kept short by design
    generated_by TEXT NOT NULL,      -- archivist | owner | seed
    fact_count   INTEGER NOT NULL DEFAULT 0,   -- facts it was distilled from
    ts           TEXT NOT NULL
) WITHOUT ROWID;