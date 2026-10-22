# =============================================================================
# src/jarvis/worker/claude_jobs.py - the Claude Code shift, Mac side
# =============================================================================
#
# DAEDALUS, ATHENA and CALLIOPE as `claude -p` runs on the owner's
# subscription. Same briefs, same kind of result as the API versions -
# a summary for chat and a file - from a different engine.
#
# WHERE EACH RUNS
#   code.cc      in the DAEDALUS workspace (~/Desktop/jarvismk3-sandbox),
#                so projects persist between tasks exactly as before.
#   research.cc  in a throwaway folder; the brief comes back as brief.md.
#   write.cc     in a throwaway folder, with any source files in sources/;
#                the piece comes back as document.md.
#
# WHAT CLAUDE CODE MAY DO
#   Its own sandbox wraps every shell command: writes only inside the
#   run folder (and temp), network only to the Python package index,
#   and no retrying outside the sandbox. failIfUnavailable makes a Mac
#   without the sandbox refuse the run rather than run unguarded.
#   File tools (Read/Write/Edit) follow permission rules instead: edits
#   are accepted inside the run folder; reads of keys and .env files are
#   denied. git push and gh are denied outright - publishing stays a
#   decision for the owner, as with the API DAEDALUS.
#
# WHEN THE PLAN IS OUT OF USAGE
#   The job does not retry for hours. It finishes with fallback=true and
#   says so; the Core starts the API version in its place.
# =============================================================================

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from jarvis.common.log import get_logger
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec
from jarvis.jobs.claude_code import (
    CLAUDE_CODE,
    CODE_CC,
    RESEARCH_CC,
    WRITE_CC,
    ClaudeCodeIn,
    ClaudeCodeOut,
)
from jarvis.subagents.researcher import ATHENA_PERSONA_V1
from jarvis.subagents.writer import CALLIOPE_PERSONA_V1
from jarvis.worker.reels import (
    ClaudeResult,
    UsageLimited,
    _interrupt,
    _parse_result,
    _raise_for,
    _subscription_env,
)

log = get_logger("worker.claude_jobs")

_MAX_DELIVERY_BYTES = 20 * 1024 * 1024
_MAX_DELIVERIES = 5

# Every shell command Claude Code runs goes through this sandbox.
SANDBOX: dict[str, Any] = {
    "sandbox": {
        "enabled": True,
        "allowUnsandboxedCommands": False,
        "failIfUnavailable": True,
        "network": {
            "allowedDomains": ["pypi.org", "files.pythonhosted.org"],
            "strictAllowlist": True,
        },
    },
}

_DENIED = ",".join([
    "Bash(git push *)", "Bash(git push)", "Bash(gh *)", "Bash(sudo *)",
    "Read(~/.ssh/**)", "Read(~/.aws/**)", "Read(~/.config/gh/**)",
    "Read(**/.env)", "Read(**/.env.*)",
])

_TOOLS = {
    CODE_CC: "Read,Write,Edit,Glob,Grep,Bash,WebSearch,WebFetch",
    RESEARCH_CC: "WebSearch,WebFetch,Read,Write,Edit",
    WRITE_CC: "Read,Write,Edit,Glob,Grep,WebSearch,WebFetch",
}

_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string",
                    "description": "Two to four sentences for a phone screen."},
        "completed": {"type": "boolean"},
        "deliver": {"type": "array", "items": {"type": "string"},
                    "description": "Paths, relative to the working folder, of "
                                   "files to send to the owner."},
    },
    "required": ["summary", "completed"],
}

_UNATTENDED = (
    "You are running unattended for the owner, through his assistant JARVIS. "
    "Nobody can answer a question: make reasonable calls and say what you "
    "assumed. Shell commands run in a sandbox: only this folder is writable "
    "and the network reaches the Python package index only."
)

_DAEDALUS = """\
You are DAEDALUS, the engineer of a personal AI system, working in the
owner's sandbox workspace. Projects live in projects/<name>/; one-off
work goes in scratch/. Before changing an existing project, read its
structure. Commit each working piece to git locally with a clear
message. Never push - publishing is the owner's decision. Compute
answers rather than recalling them: write the code, run it, check the
output. Files the owner should see (a chart, a report, a dataset) go in
"deliver"; project source does not need delivering."""


def code_prompt(task: str, inbox: str | None) -> str:
    files = (f"\n\nFiles for this task are in {inbox}/." if inbox else "")
    return (f"{_DAEDALUS}\n\n{_UNATTENDED}\n\nTask:\n{task}{files}\n\n"
            f"Finish with a summary, whether you completed the task, and "
            f"anything to deliver.")


def research_prompt(task: str) -> str:
    return (f"{ATHENA_PERSONA_V1}\n{_UNATTENDED}\n\nWrite the finished document "
            f"to brief.md in this folder - nothing else goes in that file.\n\n"
            f"Brief:\n{task}\n\nReturn the document's Summary section as the "
            f"summary.")


def write_prompt(task: str, sources: list[str]) -> str:
    listed = ("\n\nSource material, read it first:\n"
              + "\n".join(f"- sources/{name}" for name in sources)) if sources else ""
    return (f"{CALLIOPE_PERSONA_V1}\n{_UNATTENDED}\n\nWrite the finished piece "
            f"to document.md in this folder, as markdown with a # title.{listed}"
            f"\n\nBrief:\n{task}\n\nReturn two or three sentences on what you "
            f"wrote as the summary.")


class HeadlessClaude:
    """`claude -p` in a chosen folder, with the sandbox and rules above."""

    def __init__(self, claude_bin: str) -> None:
        self._bin = claude_bin

    def command(self, prompt: str, tools: str) -> list[str]:
        # NOT --bare: that skips the subscription login.
        return [
            self._bin, "-p", prompt,
            "--output-format", "json",
            "--json-schema", json.dumps(_REPORT_SCHEMA),
            "--permission-mode", "acceptEdits",
            "--permission-prompts", "none",
            "--allowedTools", tools,
            "--disallowedTools", _DENIED,
            "--settings", json.dumps(SANDBOX),
        ]

    async def run(self, prompt: str, tools: str, cwd: Path, timeout_s: int) -> ClaudeResult:
        process = await asyncio.create_subprocess_exec(
            *self.command(prompt, tools),
            cwd=cwd,
            env=_subscription_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except asyncio.CancelledError:
            await _interrupt(process)
            raise
        except asyncio.TimeoutError:
            await _interrupt(process)
            raise RuntimeError(
                f"Claude Code took longer than {timeout_s}s and was interrupted") from None
        return _parse_result(process.returncode or 0, stdout, stderr)


_OUT_OF_USAGE = ("Claude Code is out of usage for now, so this is being "
                 "redone through the API.")


class ClaudeJobs:
    """The three handlers, sharing one runner and one workspace."""

    def __init__(self, claude: HeadlessClaude, workspace: Path) -> None:
        self._claude = claude
        self._workspace = workspace

    async def _run(self, job_type: str, prompt: str, cwd: Path, ctx: JobContext,
                   timeout_s: int) -> tuple[ClaudeResult | None, dict[str, Any]]:
        await ctx.progress("Claude Code starting")
        result = await self._claude.run(prompt, _TOOLS[job_type], cwd, timeout_s)
        try:
            _raise_for(result, job_type)
        except UsageLimited:
            log.warning("claude code out of usage - asking for the API version",
                        extra={"job_id": ctx.job_id})
            return None, {}
        return result, result.structured or {}

    async def code(self, payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, ClaudeCodeIn)
        self._workspace.mkdir(parents=True, exist_ok=True)
        inbox: str | None = None
        if ctx.input_files:
            folder = self._workspace / "scratch" / f"inbox-{ctx.job_id[-8:].lower()}"
            folder.mkdir(parents=True, exist_ok=True)
            for name, content in ctx.input_files.items():
                (folder / Path(name).name).write_bytes(content)
            inbox = str(folder.relative_to(self._workspace))
        result, report = await self._run(
            CODE_CC, code_prompt(payload.task, inbox), self._workspace, ctx, 1740)
        if result is None:
            return ClaudeCodeOut(summary=_OUT_OF_USAGE, completed=False, fallback=True)
        sent = await self._deliver(self._workspace, report.get("deliver") or [], ctx)
        return ClaudeCodeOut(
            summary=str(report.get("summary") or result.text)[:600],
            report=result.text,
            filename=sent[-1] if sent else "",
            completed=bool(report.get("completed", True)),
            cost_usd=result.cost_usd,
        )

    async def research(self, payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, ClaudeCodeIn)
        return await self._document(
            RESEARCH_CC, research_prompt(payload.task), "brief.md", ctx, {})

    async def write(self, payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, ClaudeCodeIn)
        sources = {Path(name).name: data for name, data in ctx.input_files.items()}
        return await self._document(
            WRITE_CC, write_prompt(payload.task, sorted(sources)), "document.md",
            ctx, sources)

    async def _document(self, job_type: str, prompt: str, expected: str,
                        ctx: JobContext, sources: dict[str, bytes]) -> BaseModel:
        folder = Path(tempfile.mkdtemp(prefix="jarvis-cc-"))
        try:
            if sources:
                (folder / "sources").mkdir()
                for name, data in sources.items():
                    (folder / "sources" / name).write_bytes(data)
            result, report = await self._run(job_type, prompt, folder, ctx, 840)
            if result is None:
                return ClaudeCodeOut(summary=_OUT_OF_USAGE, completed=False, fallback=True)
            document = folder / expected
            if not document.is_file() or not document.read_text(errors="replace").strip():
                raise RuntimeError(f"Claude Code finished without writing {expected}")
            markdown = document.read_text(errors="replace")
            filename = _filename_for(markdown, ctx.job_id, expected)
            await ctx.write_artifact(filename, "text/markdown", markdown.encode("utf-8"))
            return ClaudeCodeOut(
                summary=str(report.get("summary") or markdown[:500])[:600],
                report=result.text,
                filename=filename,
                completed=bool(report.get("completed", True)),
                cost_usd=result.cost_usd,
            )
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    async def _deliver(self, root: Path, paths: list[Any], ctx: JobContext) -> list[str]:
        """Send the files Claude Code asked to deliver - only ones inside
        the run folder, only a few, only reasonably sized."""
        sent: list[str] = []
        base = root.resolve()
        for raw in paths[:_MAX_DELIVERIES]:
            target = (root / str(raw)).resolve()
            if not target.is_relative_to(base) or not target.is_file():
                continue
            if target.stat().st_size > _MAX_DELIVERY_BYTES:
                continue
            await ctx.write_artifact(target.name, _mime(target.name), target.read_bytes())
            sent.append(target.name)
        return sent


def _filename_for(markdown: str, job_id: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            slug = re.sub(r"[^a-z0-9]+", "-", line[2:].strip().lower()).strip("-")[:60]
            if slug:
                return f"{slug}.md"
    return f"{Path(fallback).stem}-{job_id[-6:].lower()}.md"


def _mime(filename: str) -> str:
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
        ".csv": "text/csv", ".json": "application/json", ".md": "text/markdown",
        ".txt": "text/plain", ".py": "text/x-python", ".pdf": "application/pdf",
        ".html": "text/html",
    }.get(Path(filename).suffix.lower(), "application/octet-stream")


def register_claude_jobs(
    registry: JobTypeRegistry,
    claude_bin: str,
    workspace: Path,
    capabilities: list[str],
) -> bool:
    """Register the three handlers if this Mac can run them. The Core
    only routes here when the worker advertises "claude-code", so the
    tag and the handlers must agree - a mismatch is logged."""
    if CLAUDE_CODE not in capabilities:
        return False
    if shutil.which(claude_bin) is None:
        log.warning("claude-code advertised but the claude CLI is not on PATH - "
                    "remove the tag or set JARVIS_CLAUDE_BIN", extra={"claude_bin": claude_bin})
        return False
    jobs = ClaudeJobs(HeadlessClaude(claude_bin), workspace)
    for job_type, handler, timeout_s in (
        (CODE_CC, jobs.code, 1800),
        (RESEARCH_CC, jobs.research, 900),
        (WRITE_CC, jobs.write, 900),
    ):
        registry.register(JobTypeSpec(
            type=job_type, input_model=ClaudeCodeIn, output_model=ClaudeCodeOut,
            execution="idempotent", requires=[CLAUDE_CODE],
            timeout_s=timeout_s, handler=handler,
        ))
    log.info("claude code jobs registered", extra={"workspace": str(workspace)})
    return True
