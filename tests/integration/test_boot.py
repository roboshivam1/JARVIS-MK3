# =============================================================================
# tests/integration/test_boot.py - phase 0 acceptance: boot, persist, restart
# =============================================================================
#
# These tests run a REAL CoreApp against a REAL SQLite file in a pytest
# temp directory. They encode the phase 0 acceptance criteria:
#
#   - the Core boots and writes core.started to the database
#   - it shuts down cleanly
#   - a second boot against the same database loses nothing and
#     re-applies nothing (restart survival)
#   - the event log physically refuses UPDATE and DELETE
# =============================================================================

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.core.db import database as db_module

from jarvis.common.events import EventKind
from jarvis.common.settings import CoreSettings
from jarvis.core.app import CoreApp
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo


def _settings(tmp_path: Path) -> CoreSettings:
    # Point the Core at a throwaway data dir; nothing touches ./data.
    # _env_file=None: tests build their entire world explicitly and must
    # never read the developer's real .env - a test that sees your real
    # bot token is a test that fails differently on every machine.
    return CoreSettings(_env_file=None, data_dir=tmp_path / "data")

async def test_boot_writes_core_started(tmp_path: Path) -> None:
    app = CoreApp(_settings(tmp_path))
    await app.boot()
    try:
        assert app.events is not None
        started = await app.events.recent(kind=EventKind.CORE_STARTED)
        assert len(started) == 1
        assert started[0].source == "core.app"
        # A fresh database applies EVERY migration that exists - counted
        # from the migrations folder itself, so this test does not go
        # stale each time a phase adds a schema change. (The other half
        # of the invariant - a rebooted db applies zero - is pinned by
        # test_restart_survival.)
        known_migrations = len(list(db_module._MIGRATIONS_DIR.glob("*.sql")))
        assert known_migrations >= 1
        assert started[0].payload["migrations_applied"] == known_migrations
    finally:
        await app.shutdown()


async def test_restart_survival(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    # First life.
    app1 = CoreApp(settings)
    await app1.boot()
    await app1.shutdown()

    # Second life, same database file.
    app2 = CoreApp(settings)
    await app2.boot()
    try:
        assert app2.events is not None
        started = await app2.events.recent(kind=EventKind.CORE_STARTED)
        # Both boots are on record - history survived the restart.
        assert len(started) == 2
        # And the second boot applied zero migrations - idempotent boot.
        newest = started[0]  # recent() returns newest first
        assert newest.payload["migrations_applied"] == 0
    finally:
        await app2.shutdown()


async def test_shutdown_is_idempotent(tmp_path: Path) -> None:
    app = CoreApp(_settings(tmp_path))
    await app.boot()
    await app.shutdown()
    await app.shutdown()  # second call must be harmless


async def test_event_log_refuses_mutation(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = CoreApp(settings)
    await app.boot()
    await app.shutdown()

    # Reopen the raw database and try to falsify history.
    db = await Database.connect(settings.db_path)
    try:
        with pytest.raises(Exception, match="append-only"):
            await db.execute("UPDATE events SET source = 'tampered'")
        with pytest.raises(Exception, match="append-only"):
            await db.execute("DELETE FROM events")
        # And confirm the row is untouched.
        repo = EventsRepo(db)
        events = await repo.recent(kind=EventKind.CORE_STARTED)
        assert events and events[0].source == "core.app"
    finally:
        await db.close()