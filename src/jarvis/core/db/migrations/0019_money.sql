-- ============================================================================
-- src/jarvis/core/db/migrations/0019_money.sql
--
-- Migration 0019: the money ledger, kept by PLUTUS.
--
-- Spending only, for now. Every row is a payment that left the owner's
-- account (or cash he mentioned). Credits are not tracked: his account
-- also receives family transfers that are none of the analysis's
-- business.
--
-- WHAT IS NOT STORED: full account numbers (last four digits only), the
-- bank's message bodies, OTPs, card numbers, PAN. A transaction is the
-- handful of fields needed to reason about spending, nothing more.
--
-- Amounts are integer PAISE: money in floating point drifts.
-- ============================================================================

CREATE TABLE money_categories (
    name       TEXT PRIMARY KEY,
    sort       INTEGER NOT NULL DEFAULT 100,
    is_spend   INTEGER NOT NULL DEFAULT 1        -- 0: moves money, isn't spending
) WITHOUT ROWID;

INSERT INTO money_categories (name, sort, is_spend) VALUES
    ('food', 10, 1),
    ('groceries', 20, 1),
    ('transport', 30, 1),
    ('shopping', 40, 1),
    ('project parts', 50, 1),
    ('subscriptions', 60, 1),
    ('education', 70, 1),
    ('gym', 80, 1),
    ('entertainment', 90, 1),
    ('personal care', 100, 1),
    ('gifts', 110, 1),
    ('other', 190, 1),
    ('transfers', 200, 0);                       -- to himself or family: not spending

CREATE TABLE money_transactions (
    id             TEXT PRIMARY KEY,               -- ULID
    ref            TEXT,                           -- bank's RRN / transaction id
    occurred_at    TEXT NOT NULL,                  -- local 'YYYY-MM-DD HH:MM:SS'
    date           TEXT NOT NULL,                  -- local YYYY-MM-DD
    amount_paise   INTEGER NOT NULL CHECK (amount_paise > 0),
    payee_raw      TEXT NOT NULL,                  -- as the bank wrote it
    payee_key      TEXT NOT NULL,                  -- normalised, what rules match on
    category       TEXT REFERENCES money_categories(name) ON UPDATE CASCADE,
    categorized_by TEXT CHECK (categorized_by IN ('rule', 'owner')),
    scope          TEXT NOT NULL DEFAULT 'personal'
                   CHECK (scope IN ('personal', 'basement_supply')),
    project_id     TEXT REFERENCES projects(id) ON DELETE SET NULL,
    channel        TEXT,                           -- UPI, cash...
    status         TEXT NOT NULL DEFAULT 'success'
                   CHECK (status IN ('success', 'failed', 'pending')),
    account_last4  TEXT,
    source         TEXT NOT NULL,                  -- email | chat | web
    note           TEXT,
    created_ts     TEXT NOT NULL
) WITHOUT ROWID;

-- One payment, one row, however many alerts describe it.
CREATE UNIQUE INDEX idx_money_ref ON money_transactions (ref) WHERE ref IS NOT NULL;
CREATE INDEX idx_money_date ON money_transactions (date);
CREATE INDEX idx_money_payee ON money_transactions (payee_key);

-- "Campus P is food": set once, applied to every payment to that payee.
CREATE TABLE money_rules (
    payee_key  TEXT PRIMARY KEY,
    label      TEXT,                               -- a nicer name: "Campus canteen"
    category   TEXT NOT NULL REFERENCES money_categories(name) ON UPDATE CASCADE,
    scope      TEXT NOT NULL DEFAULT 'personal'
               CHECK (scope IN ('personal', 'basement_supply')),
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    created_ts TEXT NOT NULL
) WITHOUT ROWID;

-- The mailbox reader's bookkeeping: where it got to, and how it went.
CREATE TABLE money_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
