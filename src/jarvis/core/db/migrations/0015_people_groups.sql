-- ============================================================================
-- src/jarvis/core/db/migrations/0015_people_groups.sql
--
-- Migration 0015: groups, the bins the owner sorts people into - family,
-- school, college, the colony, a project team.
--
-- Membership is many-to-many on purpose. One person is LNMIIT AND the
-- Robofest team AND the robotics club at the same time; fixed single
-- bins break on the first real person.
--
-- Groups earn their place three ways: two people with the same name stop
-- colliding once one is "colony" and the other "LNMIIT", the directory
-- becomes browsable, and "the Robofest team" becomes something JARVIS
-- can resolve to actual people.
-- ============================================================================

CREATE TABLE people_groups (
    id          TEXT PRIMARY KEY,           -- ULID
    name        TEXT NOT NULL COLLATE NOCASE UNIQUE,
    kind        TEXT NOT NULL DEFAULT 'other'
                CHECK (kind IN ('family', 'school', 'college', 'locality',
                                'project', 'work', 'other')),
    description TEXT,
    created_ts  TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE person_group_members (
    group_id   TEXT NOT NULL REFERENCES people_groups(id) ON DELETE CASCADE,
    person_id  TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    created_ts TEXT NOT NULL,
    PRIMARY KEY (group_id, person_id)
) WITHOUT ROWID;

CREATE INDEX idx_person_group_members_person ON person_group_members (person_id);
