-- ============================================================================
-- src/jarvis/core/db/migrations/0002_sessions_turns_llm_calls.sql
--
-- Migration 0002: conversation storage (sessions, turns) and the money
-- ledger (llm_calls).
--
-- Note the contrast with events: sessions and turns are LIVING STATE and
-- may be updated (summary refresh, last-active time, archiving), so there
-- are no append-only triggers here. llm_calls is effectively append-only
-- in practice, but enforcement is not needed - nothing has any reason to
-- update it, and a trigger would block legitimate future corrections to
-- mispriced rows.
-- ============================================================================

CREATE TABLE sessions (
    id              TEXT PRIMARY KEY,      -- ULID
    client_kind     TEXT NOT NULL,         -- telegram, web, voice.mac, ...
    title           TEXT,                  -- auto-generated later
    status          TEXT NOT NULL DEFAULT 'active',   -- active | archived
    rolling_summary TEXT NOT NULL DEFAULT '',         -- empty until memory phase
    created_ts      TEXT NOT NULL,         -- UTC ISO-8601
    last_active_ts  TEXT NOT NULL
) WITHOUT ROWID;

-- The common lookup: "the active session for this client kind" (Telegram
-- reuses one long-lived session).
CREATE INDEX idx_sessions_client_status ON sessions (client_kind, status);

CREATE TABLE turns (
    id           TEXT PRIMARY KEY,         -- ULID; time-ordered
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    role         TEXT NOT NULL,            -- user | assistant
    content      TEXT NOT NULL,
    attachments  TEXT NOT NULL DEFAULT '[]',   -- JSON list of artifact ids
    job_refs     TEXT NOT NULL DEFAULT '[]',   -- JSON list of job ids
    llm_call_ids TEXT NOT NULL DEFAULT '[]',   -- JSON list of llm_calls ids
    ts           TEXT NOT NULL
) WITHOUT ROWID;

-- THE turn query: last N turns of one session, in time order. With this
-- compound index it is a single index sweep, no sort step.
CREATE INDEX idx_turns_session ON turns (session_id, id);

-- One row per model call, no exceptions - the accounting truth that
-- answers "what did you cost me this week?" and later feeds the budget
-- guard.
CREATE TABLE llm_calls (
    id            TEXT PRIMARY KEY,        -- ULID
    ts            TEXT NOT NULL,           -- UTC ISO-8601
    trace_id      TEXT NOT NULL,           -- causal chain this call served
    actor         TEXT NOT NULL,           -- core.orchestrator, subagent.researcher, ...
    tier          TEXT NOT NULL,           -- reasoner | utility | ...
    model         TEXT NOT NULL,           -- concrete model that ran
    latency_ms    INTEGER NOT NULL,
    tokens_in     INTEGER NOT NULL,
    tokens_out    INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,
    cost_inr      REAL NOT NULL,
    stop_reason   TEXT NOT NULL,           -- end_turn | tool_use | error | ...
    error         TEXT                     -- NULL on success
) WITHOUT ROWID;

-- "everything this root cause spent" and "what did today cost".
CREATE INDEX idx_llm_calls_trace ON llm_calls (trace_id);
CREATE INDEX idx_llm_calls_ts    ON llm_calls (ts);