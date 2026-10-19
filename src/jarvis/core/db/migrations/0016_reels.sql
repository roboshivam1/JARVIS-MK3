-- ============================================================================
-- src/jarvis/core/db/migrations/0016_reels.sql
--
-- Migration 0016: the reel episode log.
--
-- One row per reel JARVIS has been asked for, whatever became of it. The
-- log does three jobs: it is the state the reel service drives (what has
-- been sent for review, what is rendering), it is what the brief writer
-- reads so episodes can call back to earlier ones and avoid repeats, and
-- it is the owner's record of what went out.
--
-- The script, stills and voice live in artifacts; this table points at
-- them. The brief is stored whole as JSON because it is a value, never
-- queried by its parts.
-- ============================================================================

CREATE TABLE reel_episodes (
    id                 TEXT PRIMARY KEY,              -- ULID
    kit_id             TEXT NOT NULL UNIQUE,          -- the id the kit sees
    job_id             TEXT,
    type               TEXT NOT NULL,
    title              TEXT NOT NULL DEFAULT '',
    request            TEXT NOT NULL,                 -- what the owner asked for
    brief              TEXT NOT NULL,                 -- JSON ReelBrief
    lines              TEXT NOT NULL DEFAULT '[]',    -- JSON list, latest draft
    status             TEXT NOT NULL DEFAULT 'briefed'
                       CHECK (status IN ('briefed', 'drafting', 'awaiting_review',
                                         'rendering', 'delivered', 'posted',
                                         'cancelled', 'failed')),
    round              INTEGER NOT NULL DEFAULT 0,    -- latest draft round
    review_sent_round  INTEGER NOT NULL DEFAULT 0,    -- last round shown to the owner
    approval_id        TEXT,
    session_id         TEXT,                          -- Claude Code session on the Mac
    sheet_artifact_id  TEXT,
    voice_artifact_id  TEXT,
    mp4_artifact_id    TEXT,
    notes              TEXT,                          -- latest notes from the owner
    error              TEXT,
    verification       TEXT,                          -- the kit's own mux check
    created_ts         TEXT NOT NULL,
    updated_ts         TEXT NOT NULL,
    approved_ts        TEXT,
    delivered_ts       TEXT,
    posted_ts          TEXT
) WITHOUT ROWID;

CREATE INDEX idx_reel_episodes_status ON reel_episodes (status);
CREATE INDEX idx_reel_episodes_created ON reel_episodes (created_ts);
CREATE INDEX idx_reel_episodes_job ON reel_episodes (job_id);
