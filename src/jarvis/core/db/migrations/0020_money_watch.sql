-- ============================================================================
-- src/jarvis/core/db/migrations/0020_money_watch.sql
--
-- Migration 0020: PLUTUS's watchdog.
--
-- Two things it keeps an eye on:
--
--   money_watch          payments that need the owner's attention: one
--                        that FAILED (was the money really not taken?) or
--                        one that is UNUSUAL against his own history.
--                        Open until he clears it.
--
--   money_subscriptions  payees he pays on a cycle. Detected from the
--                        ledger (or marked by hand), each with the next
--                        date a payment is expected - so a renewal can be
--                        flagged before it lands, a price rise noticed,
--                        and a payment that never came pointed out.
-- ============================================================================

CREATE TABLE money_watch (
    id         TEXT PRIMARY KEY,                 -- ULID
    kind       TEXT NOT NULL CHECK (kind IN ('failed', 'unusual')),
    txn_id     TEXT NOT NULL REFERENCES money_transactions(id) ON DELETE CASCADE,
    reason     TEXT NOT NULL,                    -- what made it worth a look
    status     TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'cleared')),
    note       TEXT,                             -- how he resolved it
    created_ts TEXT NOT NULL,
    cleared_ts TEXT
) WITHOUT ROWID;

CREATE UNIQUE INDEX idx_money_watch_txn ON money_watch (txn_id, kind);
CREATE INDEX idx_money_watch_open ON money_watch (status);

CREATE TABLE money_subscriptions (
    payee_key     TEXT PRIMARY KEY,
    cadence       TEXT NOT NULL CHECK (cadence IN ('monthly', 'yearly')),
    typical_paise INTEGER NOT NULL,              -- what it usually costs
    last_paise    INTEGER NOT NULL,
    last_date     TEXT NOT NULL,                 -- YYYY-MM-DD of the latest payment
    next_due      TEXT NOT NULL,                 -- when the next one is expected
    status        TEXT NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'dismissed')),
    source        TEXT NOT NULL DEFAULT 'detected' CHECK (source IN ('detected', 'owner')),
    warned_for    TEXT,                          -- next_due a renewal warning went out for
    missed_for    TEXT,                          -- next_due a "never came" note went out for
    updated_ts    TEXT NOT NULL
) WITHOUT ROWID;
