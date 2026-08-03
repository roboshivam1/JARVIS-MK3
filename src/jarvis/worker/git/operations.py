# =============================================================================
# src/jarvis/worker/git/operations.py - git and GitHub, narrowly
# =============================================================================
#
# Runs OUTSIDE the sandbox, necessarily: git needs network access and
# writes to real directories, which are the two things the jail exists
# to deny. So the protections here are different in kind - an allowlist
# rather than a kernel policy, and approval gates on anything that
# leaves the machine.
#
# WHAT IS FREE AND WHAT IS GATED:
#   clone, read, commit locally - free. Reversible, private, cheap.
#   push, create repo           - gated. These leave the machine and
#                                 cannot be taken back.
#
# The token never reaches the model. It is injected into remote URLs at
# the moment of use and scrubbed from anything returned, so a subagent
# cannot leak what it never sees.
# =============================================================================

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from jarvis.common.log import get_logger
from jarvis.worker.git.config import GitConfig

log = get_logger("worker.git")

_TIMEOUT_S = 120
_GITHUB_API = "https://api.github.com"


@dataclass
class GitResult:
    """What one git command produced."""

    stdout: str
    stderr: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def summary(self) -> str:
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.strip()[:4000])
        if self.stderr.strip():
            parts.append(f"stderr: {self.stderr.strip()[:2000]}")
        if not parts:
            parts.append("(no output)")
        if not self.ok:
            parts.append(f"exit code: {self.exit_code}")
        return "\n\n".join(parts)


def scrub(text: str, token: str) -> str:
    """Remove the token from anything on its way back to the model.

    Git prints remote URLs in error messages, and an authenticated URL
    contains the token. A subagent that never sees the secret cannot
    leak it - and the model has no legitimate use for it.
    """
    if not token:
        return text
    cleaned = text.replace(token, "***")
    # Also catch the https://token@github.com form in one pass.
    return re.sub(r"https://[^@\s]+@github\.com", "https://github.com", cleaned)


class GitOperations:
    """Git and GitHub, scoped to configured repositories."""

    def __init__(self, config: GitConfig, token: str, workspace: Path) -> None:
        self._config = config
        self._token = token
        self._workspace = workspace
        self._workspace.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> GitConfig:
        return self._config

    def repo_path(self, full_name: str) -> Path:
        """Where a repo lives locally. Flattened so owner/name cannot
        escape the workspace via a crafted name."""
        safe = full_name.replace("/", "__")
        return self._workspace / safe

    # -- running git ----------------------------------------------------------

    async def _git(self, args: list[str], cwd: Path | None = None) -> GitResult:
        """Run one git command with the identity from config."""
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "HOME": str(self._workspace),
            "GIT_AUTHOR_NAME": self._config.author_name,
            "GIT_AUTHOR_EMAIL": self._config.author_email,
            "GIT_COMMITTER_NAME": self._config.author_name,
            "GIT_COMMITTER_EMAIL": self._config.author_email,
            # Never prompt: an agent cannot answer, and a hung git
            # command waiting for a password is a job that times out
            # with no useful explanation.
            "GIT_TERMINAL_PROMPT": "0",
        }

        process = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(cwd) if cwd else str(self._workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return GitResult("", f"git timed out after {_TIMEOUT_S}s", -1)

        return GitResult(
            stdout=scrub(stdout.decode("utf-8", errors="replace"), self._token),
            stderr=scrub(stderr.decode("utf-8", errors="replace"), self._token),
            exit_code=process.returncode if process.returncode is not None else -1,
        )

    def _authed_url(self, full_name: str) -> str:
        """A clone URL carrying the token. Built at the moment of use and
        never stored, logged, or returned."""
        return f"https://{self._token}@github.com/{full_name}.git"

    # -- operations -----------------------------------------------------------

    async def clone(self, full_name: str) -> GitResult:
        """Clone or update a repo into the workspace."""
        if not self._config.may_read(full_name):
            return GitResult("", f"{full_name} is not in the allowlist", 1)

        path = self.repo_path(full_name)
        if (path / ".git").exists():
            return await self._git(["pull", "--ff-only"], cwd=path)

        result = await self._git([
            "clone", self._authed_url(full_name), str(path),
        ])
        if result.ok:
            log.info("repo cloned", extra={"repo": full_name})
        return result

    async def status(self, full_name: str) -> GitResult:
        path = self.repo_path(full_name)
        if not (path / ".git").exists():
            return GitResult("", f"{full_name} is not cloned yet", 1)
        return await self._git(["status", "--short", "--branch"], cwd=path)

    async def commit(self, full_name: str, message: str) -> GitResult:
        """Stage everything and commit. Local only - nothing leaves the
        machine until a push, which is gated."""
        if not self._config.may_write(full_name):
            return GitResult("", f"{full_name} is read-only", 1)

        path = self.repo_path(full_name)
        if not (path / ".git").exists():
            return GitResult("", f"{full_name} is not cloned yet", 1)

        staged = await self._git(["add", "-A"], cwd=path)
        if not staged.ok:
            return staged

        full_message = message
        if self._config.coauthor_trailer:
            # A trailer, not a signature: GitHub reads this and credits
            # both parties, which is an honest record of how the commit
            # was produced.
            full_message += (
                f"\n\nCo-Authored-By: {self._config.author_name} "
                f"<{self._config.author_email}>"
            )

        return await self._git(["commit", "-m", full_message], cwd=path)

    async def push(self, full_name: str) -> GitResult:
        """Send commits to GitHub. GATED - the caller must have obtained
        approval before this runs."""
        if not self._config.may_write(full_name):
            return GitResult("", f"{full_name} is read-only", 1)

        path = self.repo_path(full_name)
        if not (path / ".git").exists():
            return GitResult("", f"{full_name} is not cloned yet", 1)

        # Set the remote fresh each time so the token is current and is
        # never persisted in the repo's config between operations.
        await self._git(
            ["remote", "set-url", "origin", self._authed_url(full_name)],
            cwd=path,
        )
        result = await self._git(["push"], cwd=path)
        # Strip the token back out of the stored remote.
        await self._git([
            "remote", "set-url", "origin",
            f"https://github.com/{full_name}.git",
        ], cwd=path)

        if result.ok:
            log.info("pushed to remote", extra={"repo": full_name})
        return result

    async def create_repo(
        self, name: str, description: str = "", private: bool = True
    ) -> GitResult:
        """Create a new GitHub repository. GATED.

        Private by default: a repo made by an agent should not be public
        until the owner has looked at it.
        """
        if not self._config.allow_create:
            return GitResult("", "repository creation is disabled in git.json", 1)

        payload = json.dumps({
            "name": name,
            "description": description,
            "private": private,
            "auto_init": True,      # so it can be cloned immediately
        })

        process = await asyncio.create_subprocess_exec(
            "curl", "-sS", "-X", "POST",
            "-H", f"Authorization: Bearer {self._token}",
            "-H", "Accept: application/vnd.github+json",
            "-d", payload,
            f"{_GITHUB_API}/user/repos",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        body = scrub(stdout.decode("utf-8", errors="replace"), self._token)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return GitResult("", f"unexpected response: {body[:500]}", 1)

        if "full_name" not in data:
            message = data.get("message", "unknown error")
            return GitResult("", f"github refused: {message}", 1)

        full_name = str(data["full_name"])
        # It made this repo, so it may commit to it. Runtime only - a
        # restart forgets the grant unless the owner adds it to git.json.
        self._config.allow_repo(full_name, write=True)
        log.info("repo created", extra={"repo": full_name, "private": private})
        return GitResult(f"Created {full_name} ({'private' if private else 'public'})", "", 0)
