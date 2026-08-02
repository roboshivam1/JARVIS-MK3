# =============================================================================
# src/jarvis/core/db/artifact_links.py - a browsable view of the artifact store
# =============================================================================
#
# Artifacts live at data/artifacts/<ulid>/<name> - a catalogue keyed by
# id, which is right for the database and useless for a human opening
# Finder. This module maintains a parallel tree of SYMLINKS organised by
# date, so files can be found the way people actually look for them:
# "that chart from Tuesday".
#
#   data/artifacts/by-date/2026-08-02/14-32_cadr-scaling.png
#       -> ../../01KZ13618AXKDF6AX3WXWNR53X/cadr-scaling.png
#
# SYMLINKS, NOT COPIES. The bytes exist once. A copy would double the
# storage and, worse, could drift from the original; a symlink either
# points at the real file or is visibly broken. Deleting an artifact
# leaves an obviously dead link rather than a stale duplicate that looks
# fine.
#
# THE FARM IS DISPOSABLE. Nothing in the database references it, and
# nothing breaks if it is deleted - it repopulates as new artifacts
# arrive, and rebuild_all() regenerates it wholesale. That is deliberate:
# a convenience layer must never become something the system depends on.
# =============================================================================

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from jarvis.common.artifacts import Artifact
from jarvis.common.log import get_logger

log = get_logger("core.artifacts.links")

BY_DATE = "by-date"


def _link_path(root: Path, artifact: Artifact) -> Path:
    """Where this artifact's symlink belongs.

    Named with the time of day so a day's files sort chronologically and
    two artifacts sharing a filename do not collide.
    """
    local = artifact.ts.astimezone()          # the owner's local day
    day = local.strftime("%Y-%m-%d")
    stamp = local.strftime("%H-%M")
    return root / BY_DATE / day / f"{stamp}_{artifact.name}"


def link_artifact(root: Path, artifact: Artifact) -> Path | None:
    """Add one artifact to the browsable tree. Returns the link path.

    Failure is logged and swallowed: this is a convenience, and a
    filesystem that will not take a symlink must not fail the job that
    produced the file.
    """
    target = root / artifact.storage_path
    link = _link_path(root, artifact)

    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        # Relative target, so the whole data directory stays portable -
        # copy it to the VPS or a backup drive and the links still work.
        link.symlink_to(os.path.relpath(target, link.parent))
        return link
    except OSError:
        log.warning("could not link artifact", exc_info=True, extra={
            "artifact_id": artifact.id, "artifact_name": artifact.name,
        })
        return None


def rebuild_all(root: Path, artifacts: list[Artifact]) -> int:
    """Regenerate the whole tree from the catalogue. Returns links made.

    For after a restore, or when the farm has been deleted. Safe to run
    any time: it only ever creates links to files the catalogue already
    knows about.
    """
    made = 0
    for artifact in artifacts:
        if link_artifact(root, artifact) is not None:
            made += 1
    log.info("artifact links rebuilt", extra={"count": made})
    return made


def prune_broken(root: Path) -> int:
    """Remove links whose target is gone. Returns how many were removed."""
    by_date = root / BY_DATE
    if not by_date.exists():
        return 0

    removed = 0
    for link in by_date.rglob("*"):
        if link.is_symlink() and not link.resolve().exists():
            link.unlink()
            removed += 1
    return removed
