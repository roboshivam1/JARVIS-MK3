# =============================================================================
# src/jarvis/worker/tools/github.py - git and GitHub through the gh CLI
# =============================================================================
#
# JARVIS NEVER HANDLES A CREDENTIAL. The owner ran `gh auth login` once,
# in a terminal; the CLI holds the token in the system keychain and this
# module shells out to it. There is nothing here to leak, nothing in
# .env, nothing injected into a URL and scrubbed back out of error
# messages - which is what the previous token-based version spent most
# of its code doing.
#
# THE HONEST TRADEOFF: gh can reach every repository the owner can. The
# boundary is therefore no longer a config allowlist but his judgement
# at the approval gate. Given the gates work, that is the better trade -
# and more truthful than a JSON file pretending to constrain a
# credential that could do anything.
#
# WHAT IS FREE AND WHAT IS GATED:
#   init, status, add, commit, clone - free. Local, reversible, private.
#   push, create repo                - gated. These leave the machine.
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from jarvis.common.log import get_logger
from jarvis.worker.workspace import PathEscape, Workspace

log = get_logger("worker.tools.github")

_TIMEOUT_S = 120


@dataclass
class CommandResult:
    """What one git or gh invocation produced."""

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
            parts.append(self.stderr.strip()[:2000])
        if not parts:
            parts.append("(no output)")
        if not self.ok:
            parts.append(f"exit code: {self.exit_code}")
        return "\n\n".join(parts)


class GitTools:
    """Git and GitHub, scoped to the workspace."""

    def __init__(
        self,
        workspace: Workspace,
        author_name: str = "JARVIS",
        author_email: str = "jarvis@localhost",
        coauthor_trailer: bool = True,
    ) -> None:
        self._ws = workspace
        self._author_name = author_name
        self._author_email = author_email
        self._coauthor = coauthor_trailer

    async def _run(
        self, command: list[str], project: str
    ) -> CommandResult:
        """Run a command inside a project directory."""
        try:
            cwd = self._ws.project_path(project)
        except PathEscape as exc:
            return CommandResult("", str(exc), 1)

        if not cwd.exists():
            return CommandResult("", f"{project} does not exist.", 1)

        import os
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": self._author_name,
            "GIT_AUTHOR_EMAIL": self._author_email,
            "GIT_COMMITTER_NAME": self._author_name,
            "GIT_COMMITTER_EMAIL": self._author_email,
            # Never prompt. An agent cannot answer a password prompt, and
            # a git command hanging on one becomes a job that times out
            # with no useful explanation.
            "GIT_TERMINAL_PROMPT": "0",
        }

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            return CommandResult("", f"timed out after {_TIMEOUT_S}s", -1)
        except FileNotFoundError:
            return CommandResult(
                "",
                f"{command[0]} is not installed. For gh: "
                f"brew install gh && gh auth login",
                127,
            )

        return CommandResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=process.returncode if process.returncode is not None else -1,
        )

    # -- local operations, free -----------------------------------------------

    async def init(self, project: str) -> CommandResult:
        """Make a project a git repository."""
        result = await self._run(["git", "init", "-b", "main"], project)
        if result.ok:
            log.info("git repo initialised", extra={"project": project})
        return result

    async def status(self, project: str) -> CommandResult:
        return await self._run(["git", "status", "--short", "--branch"], project)

    async def log(self, project: str, count: int = 10) -> CommandResult:
        return await self._run(
            ["git", "log", f"-{count}", "--oneline"], project
        )

    async def diff(self, project: str) -> CommandResult:
        """Uncommitted changes. Worth reading before committing - it is
        how an agent notices it changed something it did not mean to."""
        return await self._run(["git", "diff"], project)

    async def commit(self, project: str, message: str) -> CommandResult:
        """Stage everything and commit locally."""
        staged = await self._run(["git", "add", "-A"], project)
        if not staged.ok:
            return staged

        full_message = message
        if self._coauthor:
            # A trailer, not a signature. GitHub reads this and credits
            # both parties, which is an honest record of how the commit
            # was produced.
            full_message += (
                f"\n\nCo-Authored-By: {self._author_name} "
                f"<{self._author_email}>"
            )

        result = await self._run(
            ["git", "commit", "-m", full_message], project
        )
        if not result.ok and "nothing to commit" in (
            result.stdout + result.stderr
        ).lower():
            return CommandResult(
                "Nothing to commit - the working tree is clean.", "", 0
            )
        return result

    async def clone(self, repo: str, project: str) -> CommandResult:
        """Clone a repository into the workspace. gh handles auth, so
        private repos work without any credential here."""
        try:
            target = self._ws.project_path(project)
        except PathEscape as exc:
            return CommandResult("", str(exc), 1)

        if target.exists() and any(target.iterdir()):
            return CommandResult("", f"{project} already exists.", 1)

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            process = await asyncio.create_subprocess_exec(
                "gh", "repo", "clone", repo, str(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            return CommandResult("", "clone timed out", -1)
        except FileNotFoundError:
            return CommandResult("", "gh is not installed", 127)

        return CommandResult(
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            process.returncode or 0,
        )

    # -- outward operations, gated --------------------------------------------

    async def push(self, project: str) -> CommandResult:
        """Push to the remote. The CALLER must have obtained approval."""
        result = await self._run(["git", "push", "-u", "origin", "HEAD"], project)
        if result.ok:
            log.info("pushed to remote", extra={"project": project})
        return result

    async def create_repo(
        self, project: str, private: bool = True, description: str = ""
    ) -> CommandResult:
        """Create a GitHub repo from a local project and link it as
        origin. The CALLER must have obtained approval.

        Private by default: a repository created by an agent should not
        be public until the owner has looked at it.

        Does not push - creation and pushing stay separate so the
        approval gate on pushing is the one place that decision is made.
        """
        command = [
            "gh", "repo", "create", project,
            "--private" if private else "--public",
            "--source=.", "--remote=origin",
        ]
        if description:
            command.extend(["--description", description])

        result = await self._run(command, project)
        if result.ok:
            log.info("github repo created", extra={
                "project": project, "private": private,
            })
        return result

    async def auth_status(self) -> CommandResult:
        """Whether gh is installed and authenticated. Worth checking
        before promising anything GitHub-shaped."""
        try:
            process = await asyncio.create_subprocess_exec(
                "gh", "auth", "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
        except FileNotFoundError:
            return CommandResult(
                "", "gh is not installed. brew install gh && gh auth login", 127
            )
        return CommandResult(
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            process.returncode or 0,
        )
