# =============================================================================
# src/jarvis/worker/workspace.py - where DAEDALUS keeps its work
# =============================================================================
#
# A PERSISTENT directory of real project folders, replacing the throwaway
# temp directory each run used to get.
#
# WHY THAT MATTERS MORE THAN IT SOUNDS: with an ephemeral sandbox, a
# project cannot exist across steps, so the agent could only ever express
# itself as one enormous script. That is why prose ended up inside .py
# files and why "build me a project" produced run-1.py through
# run-10.py instead of a directory. The missing capability was not a
# better prompt; it was somewhere for work to live.
#
# OUTSIDE data/, deliberately: ~/Development/jarvismk3-sandbox is
# somewhere the owner opens in an editor without digging through the
# system's internals. The cost is that backups do not cover it - which
# is right, because these are git repositories and git is the backup.
#
# PATH SAFETY IS THE SECURITY MODEL. Every path passes through
# resolve_safe(), which refuses anything escaping the workspace. A model
# writing "../../.ssh/config" gets an error, not a compromised machine.
# Symlinks are resolved before the check, because a link inside the
# workspace pointing outside it is the sneaky version of the same attack.
# =============================================================================

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from jarvis.common.log import get_logger

log = get_logger("worker.workspace")

PROJECTS = "projects"
SCRATCH = "scratch"

# Scratch older than this is swept. Projects are never swept - they are
# the owner's, and deleting his work automatically is not a decision any
# system should make on his behalf.
SCRATCH_RETENTION_DAYS = 7

_DEFAULT_GITIGNORE = """\
__pycache__/
*.py[cod]
.venv/
venv/
.env
.DS_Store
node_modules/
dist/
build/
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
"""


class PathEscape(ValueError):
    """A path that would leave the workspace. Always refused."""


@dataclass
class Workspace:
    """The root of DAEDALUS's persistent working area."""

    root: Path

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        (self.root / PROJECTS).mkdir(parents=True, exist_ok=True)
        (self.root / SCRATCH).mkdir(parents=True, exist_ok=True)

    # -- path safety ----------------------------------------------------------

    def resolve_safe(self, relative: str) -> Path:
        """Resolve a path inside the workspace, or refuse it.

        The check happens AFTER resolution, so "projects/x/../../../etc"
        and a symlink pointing outside are both caught - the first by
        normalising the path, the second by following the link before
        comparing.
        """
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PathEscape(
                f"{relative!r} resolves outside the workspace. "
                f"Everything must stay within {self.root.name}/."
            ) from None
        return candidate

    # -- projects -------------------------------------------------------------

    def project_path(self, name: str) -> Path:
        """Where a named project lives. The name is a NAME, not a path -
        slashes in it are how you get a project called '../etc'."""
        safe = name.strip().replace("/", "-").replace("\\", "-")
        if not safe or safe.startswith("."):
            raise PathEscape(f"invalid project name: {name!r}")
        return self.resolve_safe(f"{PROJECTS}/{safe}")

    def create_project(
        self,
        name: str,
        description: str = "",
        gitignore: bool = True,
    ) -> Path:
        """Scaffold a project: folder, README, .gitignore.

        SCAFFOLDING IS A FIRST-CLASS STEP rather than something the model
        is asked to remember. A project that starts with structure keeps
        it; one that starts as a bare directory becomes a pile of loose
        files, which is exactly what happened before.

        git init is deliberately NOT here - it belongs with the other git
        operations, so there is one place that knows about repositories.
        """
        path = self.project_path(name)
        if path.exists():
            return path

        path.mkdir(parents=True)
        (path / "src").mkdir(exist_ok=True)

        readme = f"# {name}\n"
        if description:
            readme += f"\n{description}\n"
        readme += (
            f"\n---\n\nCreated by JARVIS on "
            f"{datetime.now().strftime('%d %B %Y')}.\n"
        )
        (path / "README.md").write_text(readme, encoding="utf-8")

        if gitignore:
            (path / ".gitignore").write_text(_DEFAULT_GITIGNORE, encoding="utf-8")

        log.info("project created", extra={"project": name, "path": str(path)})
        return path

    def list_projects(self) -> list[dict[str, object]]:
        """Every project, with enough detail to choose between them."""
        projects_dir = self.root / PROJECTS
        entries: list[dict[str, object]] = []
        for path in sorted(projects_dir.iterdir()):
            if not path.is_dir():
                continue
            files = sum(1 for _ in path.rglob("*") if _.is_file())
            entries.append({
                "name": path.name,
                "files": files,
                "git": (path / ".git").exists(),
                "modified": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).strftime("%d %b %H:%M"),
            })
        return entries

    # -- scratch --------------------------------------------------------------

    def scratch_path(self) -> Path:
        """A directory for one-off computation. Not a project: no README,
        no git, and swept after a week."""
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = self.resolve_safe(f"{SCRATCH}/{stamp}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sweep_scratch(self) -> int:
        """Delete old scratch directories. Projects are never touched."""
        cutoff = datetime.now() - timedelta(days=SCRATCH_RETENTION_DAYS)
        removed = 0
        for path in (self.root / SCRATCH).iterdir():
            if not path.is_dir():
                continue
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    # -- reading --------------------------------------------------------------

    def tree(self, relative: str = PROJECTS, max_entries: int = 200) -> str:
        """A readable directory tree.

        The single most useful orientation tool: an agent that can see
        the shape of a project stops guessing at filenames.
        """
        root = self.resolve_safe(relative)
        if not root.exists():
            return f"{relative} does not exist."

        lines: list[str] = [f"{root.name}/"]
        count = 0

        def walk(directory: Path, prefix: str) -> None:
            nonlocal count
            if count >= max_entries:
                return
            entries = sorted(
                (p for p in directory.iterdir() if not p.name.startswith(".")),
                key=lambda p: (p.is_file(), p.name),
            )
            for index, entry in enumerate(entries):
                if count >= max_entries:
                    lines.append(f"{prefix}... (truncated)")
                    return
                last = index == len(entries) - 1
                connector = "`-- " if last else "|-- "
                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    count += 1
                    walk(entry, prefix + ("    " if last else "|   "))
                else:
                    size = entry.stat().st_size
                    lines.append(f"{prefix}{connector}{entry.name}  ({size}b)")
                    count += 1

        walk(root, "")
        return "\n".join(lines)
