# =============================================================================
# src/jarvis/common/artifacts.py - the Artifact model
# =============================================================================
#
# An artifact is a FILE the system produced or received: a document, an
# image, a dataset. The bytes live on disk; this model is the catalogue
# card that describes them.
#
# Two halves, deliberately:
#   - the row (this model): small, queryable, backed up with the database
#   - the file: readable with ordinary tools, streamable, tarred by backups
#
# sha256 is not decoration. It lets the Core VERIFY an artifact that
# arrived over a network from a worker, and it detects the quieter
# failure of a file corrupted on disk long after it was written.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now


class Artifact(BaseModel):
    """One stored file, described."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    name: str                       # human filename, e.g. "battery-brief.md"
    mime: str                       # e.g. "text/markdown", "image/png"
    size: int = Field(ge=0)         # bytes
    sha256: str                     # hex digest of the content
    storage_path: str               # relative to the artifact root
    created_by: str                 # job id, turn id, or "client.upload"
    ts: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def _id_is_ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("name")
    @classmethod
    def _name_is_safe(cls, v: str) -> str:
        # A name is a LABEL, never a path. Rejecting separators here stops
        # a crafted filename (from a worker, or a model-chosen title) from
        # writing outside the artifact root.
        name = v.strip()
        if not name:
            raise ValueError("artifact name must be non-empty")
        if "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError(f"artifact name must not contain path separators: {v!r}")
        return name

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("artifact ts must be timezone-aware")
        return v.astimezone(timezone.utc)