# =============================================================================
# tests/conftest.py - shared test fixtures
# =============================================================================
#
# Every test gets a REAL SQLite database in its own temp folder, fully
# migrated. Real storage, real SQL, real transitions - only the expensive
# unreliable edges (models, embedders, networks) are faked.
#
# _env_file=None on settings is load-bearing: without it, pydantic
# reads the developer's actual .env and tests start depending on whose
# machine they run on.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest

from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo


@pytest.fixture
def settings(tmp_path: Path) -> CoreSettings:
    """Settings pointed entirely at throwaway paths."""
    return CoreSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        gateway_token="test-token",  # type: ignore[arg-type]
    )


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """A migrated database, closed after the test."""
    database = await Database.connect(tmp_path / "test.db")
    await database.migrate()
    yield database
    await database.close()


@pytest.fixture
def events(db: Database) -> EventsRepo:
    return EventsRepo(db)


@pytest.fixture
def jobs(db: Database, events: EventsRepo) -> JobsRepo:
    return JobsRepo(db, events)