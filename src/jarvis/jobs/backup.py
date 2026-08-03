# =============================================================================
# src/jarvis/jobs/backup.py - the nightly backup
# =============================================================================
#
# Everything JARVIS knows lives in one SQLite file and one directory of
# artifacts. This job copies both, nightly, and keeps a week of them.
#
# WHY sqlite3 .backup RATHER THAN cp: the database is being written to
# while this runs. Copying the file raw can capture it mid-transaction,
# producing a backup that looks fine and will not open. SQLite's online
# backup API takes a CONSISTENT snapshot of a live database, which is
# the entire reason it exists.
#
# WHAT IS DELIBERATELY NOT BACKED UP: .env. An automated, off-site copy
# of the owner's API keys is a worse failure mode than losing the keys
# and regenerating them. Secrets belong in a password manager, not in
# whatever bucket this job pushes to.
#
# OFF-SITE IS ONE SHELL COMMAND, configured by the owner. Local backups
# protect against "I deleted something stupid"; they do nothing about a
# dead disk. Rather than build integrations for four storage providers,
# the job runs a command the owner supplies with the backup path
# substituted in - rclone, scp, aws s3 cp, a git push, whatever he
# actually ends up using.
# =============================================================================

from __future__ import annotations

import asyncio
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec

log = get_logger("jobs.backup")

_OFFSITE_TIMEOUT_S = 600


class BackupIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="scheduled")


class BackupOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_bytes: int
    artifact_bytes: int
    backup_path: str
    offsite: str          # "sent" | "not configured" | "failed: ..."
    pruned: int


def register_backup_jobs(
    registry: JobTypeRegistry, settings: CoreSettings
) -> None:
    """Register the nightly backup. Core-side: it copies the Core's own
    files, so it could not sensibly run anywhere else."""

    async def handle(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, BackupIn)

        stamp = utc_now().strftime("%Y-%m-%dT%H-%M")
        target = settings.backups_dir / stamp
        target.mkdir(parents=True, exist_ok=True)

        # -- the database, consistently ---------------------------------------
        await ctx.progress("snapshotting the database")
        db_copy = target / "core.db"
        result = await asyncio.create_subprocess_exec(
            "sqlite3", str(settings.db_path), f".backup '{db_copy}'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await result.communicate()
        if result.returncode != 0:
            raise RuntimeError(
                f"database backup failed: {stderr.decode(errors='replace')[:500]}"
            )
        db_bytes = db_copy.stat().st_size

        # -- the artifacts ----------------------------------------------------
        await ctx.progress("archiving artifacts")
        tarball = target / "artifacts.tar.gz"
        artifact_bytes = 0
        if settings.artifacts_dir.exists():
            # Blocking work, so it runs off the event loop - a few
            # hundred megabytes of tar would otherwise stall every other
            # task in the Core for the duration.
            def _make_tarball() -> None:
                with tarfile.open(tarball, "w:gz") as archive:
                    # Skip the symlink farm: those point at files
                    # already inside the archive, and archiving both
                    # doubles the size for nothing.
                    archive.add(
                        settings.artifacts_dir,
                        arcname="artifacts",
                        filter=lambda info: (
                            None if "/by-date/" in info.name else info
                        ),
                    )

            await asyncio.to_thread(_make_tarball)
            artifact_bytes = tarball.stat().st_size

        # -- off-site ---------------------------------------------------------
        offsite = "not configured"
        command = settings.backup_command.strip()
        if command:
            await ctx.progress("sending off-site")
            offsite = await _send_offsite(command, target)

        # -- retention --------------------------------------------------------
        pruned = _prune_old(settings.backups_dir, settings.backup_retention_days)

        log.info("backup complete", extra={
            "path": str(target), "db_bytes": db_bytes,
            "artifact_bytes": artifact_bytes, "offsite": offsite,
            "pruned": pruned,
        })

        return BackupOut(
            db_bytes=db_bytes,
            artifact_bytes=artifact_bytes,
            backup_path=str(target),
            offsite=offsite,
            pruned=pruned,
        )

    registry.register(JobTypeSpec(
        type="system.backup",
        input_model=BackupIn,
        output_model=BackupOut,
        execution="idempotent",
        requires=[],
        default_priority=8,      # idle work, never ahead of the owner
        timeout_s=1800,
        handler=handle,
    ))


async def _send_offsite(command: str, backup_dir: Path) -> str:
    """Run the owner's off-site command with {path} substituted.

    An arbitrary shell command from config is a footgun in general -
    but this comes from .env, which is the same trust level as the API
    keys sitting beside it. The model never touches it.
    """
    filled = command.replace("{path}", str(backup_dir))
    try:
        process = await asyncio.create_subprocess_shell(
            filled,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=_OFFSITE_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        return f"failed: timed out after {_OFFSITE_TIMEOUT_S}s"
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"

    if process.returncode != 0:
        detail = stderr.decode(errors="replace").strip()[:300]
        # A failed off-site push does NOT fail the job: the local backup
        # succeeded and is worth keeping. The failure is reported so the
        # owner knows his off-site copy is stale.
        log.error("off-site backup failed", extra={"detail": detail})
        return f"failed: {detail}"
    return "sent"


def _prune_old(backups_dir: Path, keep_days: int) -> int:
    """Delete backups older than the retention window."""
    if not backups_dir.exists():
        return 0

    cutoff = datetime.now() - timedelta(days=keep_days)
    pruned = 0
    for entry in backups_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            made = datetime.strptime(entry.name, "%Y-%m-%dT%H-%M")
        except ValueError:
            continue        # not one of ours; leave it alone
        if made < cutoff:
            shutil.rmtree(entry, ignore_errors=True)
            pruned += 1
    return pruned
