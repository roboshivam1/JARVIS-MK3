# =============================================================================
# src/jarvis/core/db/repos/artifacts.py - storing and fetching files
# =============================================================================
#
# The only code that writes artifact bytes or artifact rows. Handles both
# halves together so they cannot drift apart.
#
# Write order is deliberate: FILE first, then ROW. A crash between them
# leaves an orphan file - invisible, harmless, and sweepable later. The
# reverse order would leave a row promising a file that does not exist,
# which is a broken promise the rest of the system would trip over.
#
# Paths are built from the artifact's own ULID plus its validated name,
# so two jobs producing "brief.md" can never collide, and a crafted name
# cannot escape the artifact root.
# =============================================================================

from __future__ import annotations

import hashlib
from pathlib import Path

import aiosqlite

from jarvis.common.artifacts import Artifact
from jarvis.common.ids import new_ulid
from jarvis.common.log import get_logger
from jarvis.core.db.database import Database

log = get_logger("core.db.artifacts")


class ArtifactIntegrityError(RuntimeError):
    """Stored bytes do not match the recorded checksum."""


class ArtifactsRepo:
    """Store files with a catalogue row; fetch them back verified."""

    def __init__(self, db: Database, root: Path) -> None:
        self._db = db
        self._root = root

    # -- writes ---------------------------------------------------------------

    async def write(
        self,
        name: str,
        mime: str,
        content: bytes,
        created_by: str,
    ) -> Artifact:
        """Store bytes as a new artifact and return its catalogue entry."""
        artifact_id = new_ulid()
        digest = hashlib.sha256(content).hexdigest()
        relative = f"{artifact_id}/{name}"

        # Validate the name BEFORE touching the filesystem: constructing
        # the model runs the no-path-separators check.
        artifact = Artifact(
            id=artifact_id,
            name=name,
            mime=mime,
            size=len(content),
            sha256=digest,
            storage_path=relative,
            created_by=created_by,
        )

        target = self._root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

        await self._db.execute(
            "INSERT INTO artifacts "
            "(id, name, mime, size, sha256, storage_path, created_by, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact.id, artifact.name, artifact.mime, artifact.size,
                artifact.sha256, artifact.storage_path, artifact.created_by,
                artifact.ts.isoformat(),
            ),
        )
        log.info("artifact stored", extra={
            "artifact_id": artifact.id, "name": name,
            "size": artifact.size, "created_by": created_by,
        })
        return artifact

    # -- reads ----------------------------------------------------------------

    async def get(self, artifact_id: str) -> Artifact | None:
        row = await self._db.query_one(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        )
        return self._to_artifact(row) if row else None

    async def for_creator(self, created_by: str) -> list[Artifact]:
        """Everything one job (or turn) produced, oldest first."""
        rows = await self._db.query(
            "SELECT * FROM artifacts WHERE created_by = ? ORDER BY id ASC",
            (created_by,),
        )
        return [self._to_artifact(r) for r in rows]

    def path_for(self, artifact: Artifact) -> Path:
        """Absolute path to the stored file - for delivery paths that
        stream from disk rather than loading into memory."""
        return self._root / artifact.storage_path

    def read_bytes(self, artifact: Artifact, verify: bool = True) -> bytes:
        """Load an artifact's content, checking the checksum by default.
        Verification is the point of storing one - skip it only for large
        files where the cost matters and the source is already trusted."""
        data = self.path_for(artifact).read_bytes()
        if verify:
            actual = hashlib.sha256(data).hexdigest()
            if actual != artifact.sha256:
                raise ArtifactIntegrityError(
                    f"artifact {artifact.id} checksum mismatch: "
                    f"stored {artifact.sha256}, found {actual}"
                )
        return data

    # -- mapping --------------------------------------------------------------

    @staticmethod
    def _to_artifact(row: aiosqlite.Row) -> Artifact:
        return Artifact.model_validate({
            "id": row["id"],
            "name": row["name"],
            "mime": row["mime"],
            "size": row["size"],
            "sha256": row["sha256"],
            "storage_path": row["storage_path"],
            "created_by": row["created_by"],
            "ts": row["ts"],
        })