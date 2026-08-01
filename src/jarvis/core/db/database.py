# =============================================================================
# src/jarvis/core/db/database.py - the Core's single gateway to SQLite
# =============================================================================
#
# One Database object per Core process, holding ONE aiosqlite connection.
#
# Why one connection, on purpose:
#   - SQLite permits exactly one writer at a time. Multiple writing
#     connections produce "database is locked" errors under load.
#   - aiosqlite serialises all operations on a connection through its own
#     worker thread, so a single shared connection gives us a correct,
#     ordered, single-writer discipline with zero locking code.
#   - Cost (reads queue behind writes) is negligible at single-user scale.
#     If measurement ever says otherwise, the fix is one extra READ-ONLY
#     connection - added then, not speculatively.
#
# Nothing outside core/db/ opens a connection or writes SQL. Repositories
# in core/db/repos/ receive this object and use execute()/query().
#
# Migrations: numbered .sql files applied in order inside transactions,
# recorded in schema_migrations. Gaps, or a DB newer than the code, abort
# boot - mismatched code and data must never be guessed through.
# =============================================================================

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

from jarvis.common.log import get_logger

log = get_logger("core.db")

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_FILE_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


class MigrationError(RuntimeError):
    """Migration state is inconsistent with the code. Boot must not proceed."""


class Database:
    """Owns the SQLite connection, pragmas, migrations, and execution
    primitives. Create via `await Database.connect(db_path)`."""

    def __init__(self, conn: aiosqlite.Connection, db_path: Path) -> None:
        # Private on purpose: use connect(), which applies the pragmas.
        # A connection without our pragmas is not safe to use.
        self._conn = conn
        self.db_path = db_path

    # -- lifecycle ------------------------------------------------------------

    @classmethod
    async def connect(cls, db_path: Path) -> "Database":
        """Open (creating if absent) the database and apply session pragmas.
        Does NOT run migrations - boot calls migrate() explicitly so the
        step is visible in the boot sequence and in logs."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(db_path)

        # Rows come back index- and name-addressable; repos read by name,
        # which survives column reordering in later migrations.
        conn.row_factory = aiosqlite.Row

        # WAL: readers never block the writer and vice versa. The journal
        # mode persists in the file, but setting it every boot keeps a
        # fresh or copied DB self-configuring.
        await conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL is safe under WAL: worst case on power loss is losing the
        # last unflushed moments, never corruption. FULL doubles fsyncs
        # for a durability margin this system does not need.
        await conn.execute("PRAGMA synchronous=NORMAL")
        # Off by default in SQLite for legacy reasons; we want real FKs.
        await conn.execute("PRAGMA foreign_keys=ON")
        # If some external tool briefly holds the file, wait up to 5s
        # instead of failing instantly.
        await conn.execute("PRAGMA busy_timeout=5000")

        log.info("database connected", extra={"db_path": str(db_path)})
        return cls(conn, db_path)

    async def close(self) -> None:
        """Flush and close. After this the object must not be used."""
        await self._conn.commit()
        await self._conn.close()
        log.info("database closed", extra={"db_path": str(self.db_path)})

    # -- execution primitives (what repositories build on) --------------------

    async def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        """One write statement, committed immediately. For multi-statement
        atomicity use transaction()."""
        await self._conn.execute(sql, tuple(params))
        await self._conn.commit()

    async def query(self, sql: str, params: Iterable[Any] = ()) -> list[aiosqlite.Row]:
        """Read rows. Named access: row["column"]."""
        cursor = await self._conn.execute(sql, tuple(params))
        try:
            return list(await cursor.fetchall())
        finally:
            await cursor.close()

    async def execute_returning_changes(
        self, sql: str, params: Iterable[Any] = ()
    ) -> int:
        """Run a write and report how many rows it touched, atomically.

        SQLite's changes() reports the LAST statement's row count on this
        connection - and every component shares one connection. Asking in
        a separate call leaves a gap in which another coroutine's write
        can land, so the answer would sometimes describe someone else's
        UPDATE. Reading cursor.rowcount from the same cursor closes the
        gap: no other statement can intervene.
        """
        cursor = await self._conn.execute(sql, tuple(params))
        try:
            changed = cursor.rowcount
        finally:
            await cursor.close()
        await self._conn.commit()
        return int(changed)

    async def query_one(self, sql: str, params: Iterable[Any] = ()) -> aiosqlite.Row | None:
        """Read at most one row, None if no match."""
        cursor = await self._conn.execute(sql, tuple(params))
        try:
            return await cursor.fetchone()
        finally:
            await cursor.close()

    def transaction(self) -> "_Transaction":
        """Group several statements atomically:

            async with db.transaction() as tx:
                await tx.execute(...)
                await tx.execute(...)
        All-or-nothing: an exception inside the block rolls everything back.
        """
        return _Transaction(self._conn)

    # -- migrations -----------------------------------------------------------

    async def migrate(self) -> int:
        """Bring the schema up to date. Returns how many migrations ran.

        Each file runs inside one transaction (SQLite DDL is transactional,
        so a failed migration rolls back cleanly and the version row is
        only recorded alongside the changes it describes).
        """
        files = self._discover_migration_files()
        applied = await self._applied_versions()

        newest_known = max(files) if files else 0
        ahead = [v for v in applied if v > newest_known]
        if ahead:
            raise MigrationError(
                f"database has migrations {ahead} unknown to this code "
                f"(newest known: {newest_known:04d}) - refusing to run "
                f"older code against a newer schema"
            )

        ran = 0
        for version in sorted(files):
            if version in applied:
                continue
            path = files[version]
            sql = path.read_text(encoding="utf-8")
            log.info("applying migration", extra={"version": version, "file": path.name})

            await self._conn.execute("BEGIN")
            try:
                await self._conn.executescript(sql)
                await self._conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_ts) "
                    "VALUES (?, ?, ?)",
                    (
                        version,
                        path.name,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                await self._conn.commit()
            except Exception:
                await self._conn.rollback()
                raise
            ran += 1

        log.info("migrations up to date", extra={"applied_now": ran, "at_version": newest_known})
        return ran

    def _discover_migration_files(self) -> dict[int, Path]:
        """Map version number -> file path. Rejects duplicates and gaps:
        a numbering mistake must fail at boot, not lurk."""
        found: dict[int, Path] = {}
        for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            m = _MIGRATION_FILE_RE.match(path.name)
            if not m:
                raise MigrationError(
                    f"migration file name {path.name!r} does not match "
                    f"NNNN_lowercase_name.sql"
                )
            version = int(m.group(1))
            if version in found:
                raise MigrationError(
                    f"duplicate migration number {version:04d}: "
                    f"{found[version].name} and {path.name}"
                )
            found[version] = path

        expected = list(range(1, len(found) + 1))
        if sorted(found) != expected:
            raise MigrationError(
                f"migration numbering has gaps: found {sorted(found)}, "
                f"expected {expected}"
            )
        return found

    async def _applied_versions(self) -> set[int]:
        """Versions recorded in schema_migrations; empty set for a fresh DB
        (where the table itself does not exist yet)."""
        row = await self.query_one(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='schema_migrations'"
        )
        if row is None:
            return set()
        rows = await self.query("SELECT version FROM schema_migrations")
        return {r["version"] for r in rows}


class _Transaction:
    """Async context manager for atomic multi-statement writes. Commit on
    clean exit, rollback on exception. Not re-entrant - SQLite has no
    nested transactions, and pretending otherwise breeds subtle bugs."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def __aenter__(self) -> "_Transaction":
        await self._conn.execute("BEGIN")
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            await self._conn.commit()
        else:
            await self._conn.rollback()

    async def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        await self._conn.execute(sql, tuple(params))