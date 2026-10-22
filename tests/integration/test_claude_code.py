# =============================================================================
# tests/integration/test_claude_code.py - the Claude Code shift
# =============================================================================
#
# The routing promise: Claude Code while the Mac is online, the API when
# it is not, and the API again when a Claude Code job cannot finish. The
# worker promise: the run is sandboxed, publishing is denied, and a plan
# out of usage hands the work back instead of retrying for hours.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.jobs import Job, JobStatus
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.queue.claude_code import ClaudeCodeRouter
from jarvis.core.queue.registry import JobContext, JobTypeRegistry
from jarvis.core.queue.registry_workers import ConnectedWorker, WorkerRegistry
from jarvis.jobs.claude_code import ClaudeCodeIn, register_claude_code_job_metadata
from jarvis.jobs.code import register_code_job_metadata
from jarvis.jobs.research import register_research_jobs
from jarvis.jobs.writing import register_writing_jobs
from jarvis.worker.claude_jobs import ClaudeJobs, HeadlessClaude
from jarvis.worker.reels import ClaudeResult


def _registry() -> JobTypeRegistry:
    registry = JobTypeRegistry()
    register_claude_code_job_metadata(registry)
    register_code_job_metadata(registry)
    register_research_jobs(registry, SimpleNamespace())          # type: ignore[arg-type]
    register_writing_jobs(registry, SimpleNamespace(), SimpleNamespace())  # type: ignore[arg-type]
    return registry


def _mac(workers: WorkerRegistry) -> None:
    workers.register(ConnectedWorker(
        worker_id="macbook", capabilities={"macos", "sandbox-exec", "claude-code"},
        max_concurrency=2, version="0.1.0"))


def _router(db: Database, workers: WorkerRegistry, **kw: Any) -> tuple[ClaudeCodeRouter, JobsRepo]:
    jobs = JobsRepo(db, EventsRepo(db))
    return ClaudeCodeRouter(db, jobs, workers, _registry(), **kw), jobs


async def test_routing_follows_the_mac(db: Database) -> None:
    workers = WorkerRegistry()
    router, _ = _router(db, workers)
    assert router.route("researcher", "solid-state batteries") == (
        "research.brief", {"brief": "solid-state batteries"})
    assert router.route("writer", "an essay", ["A1"]) == (
        "write.document", {"brief": "an essay", "source_artifacts": ["A1"]})
    assert router.route("operator", "book a table") is None   # no Claude Code version

    _mac(workers)
    assert router.route("engineer", "plot the data", ["A1"]) == (
        "code.cc", {"task": "plot the data", "input_artifacts": ["A1"]})
    assert router.route("researcher", "x" * 20)[0] == "research.cc"  # type: ignore[index]


async def _job(jobs: JobsRepo, job_type: str, status: JobStatus = JobStatus.QUEUED,
               result: dict[str, Any] | None = None) -> Job:
    job = Job(type=job_type, status=status, payload={"task": "find out about sodium cells",
                                                     "input_artifacts": []},
              requires=["claude-code"], session_id=new_ulid(), trace_id=new_ulid(),
              result=result, finished_ts=utc_now() if status is JobStatus.SUCCEEDED else None)
    await jobs.create(job)
    return job


async def _children(db: Database, parent: str) -> list[Any]:
    return list(await db.query(
        "SELECT type, payload, session_id, trace_id FROM jobs WHERE parent_job_id = ?",
        (parent,)))


async def test_out_of_usage_is_redone_on_the_api(db: Database) -> None:
    workers = WorkerRegistry()
    _mac(workers)
    router, jobs = _router(db, workers)
    done = await _job(jobs, "research.cc", JobStatus.SUCCEEDED,
                      {"summary": "out of usage", "fallback": True})
    fine = await _job(jobs, "write.cc", JobStatus.SUCCEEDED, {"summary": "ok"})

    assert await router.sweep() == 1
    assert await router.sweep() == 0                     # once, not every tick
    [child] = await _children(db, done.id)
    assert child["type"] == "research.brief"
    assert json.loads(child["payload"]) == {"brief": "find out about sodium cells"}
    assert (child["session_id"], child["trace_id"]) == (done.session_id, done.trace_id)
    assert await _children(db, fine.id) == []


async def test_work_left_queued_for_a_sleeping_mac_moves_to_the_api(db: Database) -> None:
    from datetime import timedelta

    workers = WorkerRegistry()
    router, jobs = _router(db, workers, stuck_after=timedelta(0))
    research = await _job(jobs, "research.cc")
    code = await _job(jobs, "code.cc")

    assert await router.sweep() == 1
    assert (await jobs.get(research.id)).status is JobStatus.CANCELLED  # type: ignore[union-attr]
    assert [c["type"] for c in await _children(db, research.id)] == ["research.brief"]
    # DAEDALUS's API version needs the Mac's sandbox as well - it waits.
    assert (await jobs.get(code.id)).status is JobStatus.QUEUED  # type: ignore[union-attr]

    _mac(workers)
    waiting = await _job(jobs, "write.cc")
    await router.sweep()
    assert (await jobs.get(waiting.id)).status is JobStatus.QUEUED  # type: ignore[union-attr]


# -- the Mac side -------------------------------------------------------------

class FakeClaude(HeadlessClaude):
    def __init__(self, writes: dict[str, str], structured: dict[str, Any] | None,
                 text: str = "done", ok: bool = True) -> None:
        super().__init__("claude")
        self.writes, self.structured, self.text, self.ok = writes, structured, text, ok
        self.calls: list[tuple[str, str, Path]] = []

    async def run(self, prompt: str, tools: str, cwd: Path, timeout_s: int) -> ClaudeResult:
        self.calls.append((prompt, tools, cwd))
        for name, content in self.writes.items():
            (cwd / name).parent.mkdir(parents=True, exist_ok=True)
            (cwd / name).write_text(content)
        return ClaudeResult(exit_code=0 if self.ok else 1, session_id="s1",
                            structured=self.structured, text=self.text,
                            is_error=not self.ok, cost_usd=0.42)


class Ctx:
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.artifacts: list[tuple[str, bytes]] = []
        self.ctx = JobContext(
            job_id="01J00000000000000000000000", trace_id="01J00000000000000000000001",
            checkpoint=None, save_checkpoint=self._noop, progress=self._noop2,
            write_artifact=self._write, input_files=files or {},
        )

    async def _noop(self, _: dict[str, Any]) -> None: ...
    async def _noop2(self, _: str) -> None: ...

    async def _write(self, name: str, mime: str, data: bytes) -> str:
        self.artifacts.append((name, data))
        return new_ulid()


def test_every_run_is_sandboxed_and_cannot_publish() -> None:
    argv = HeadlessClaude("claude").command("hi", "Read,Bash")
    settings = json.loads(argv[argv.index("--settings") + 1])
    assert settings["sandbox"]["enabled"] is True
    assert settings["sandbox"]["allowUnsandboxedCommands"] is False
    assert settings["sandbox"]["failIfUnavailable"] is True
    denied = argv[argv.index("--disallowedTools") + 1]
    assert "Bash(git push *)" in denied and "Bash(gh *)" in denied
    assert "Read(**/.env)" in denied
    assert "--bare" not in argv


async def test_research_comes_back_as_a_document(tmp_path: Path) -> None:
    claude = FakeClaude({"brief.md": "# Sodium Cells Today\n\n## Summary\nCheap."},
                        {"summary": "Cheap, heavy, improving.", "completed": True})
    ctx = Ctx()
    out = await ClaudeJobs(claude, tmp_path).research(ClaudeCodeIn(task="sodium cells now"), ctx.ctx)
    assert out.summary == "Cheap, heavy, improving."            # type: ignore[attr-defined]
    assert out.filename == "sodium-cells-today.md"              # type: ignore[attr-defined]
    assert ctx.artifacts[0][0] == "sodium-cells-today.md"
    prompt, tools, cwd = claude.calls[0]
    assert "ATHENA" in prompt and "brief.md" in prompt and "Bash" not in tools
    assert not cwd.exists()                                     # the run folder is cleaned up


async def test_out_of_usage_asks_for_the_api_instead_of_retrying(tmp_path: Path) -> None:
    claude = FakeClaude({}, None, text="Claude AI usage limit reached", ok=False)
    out = await ClaudeJobs(claude, tmp_path).write(ClaudeCodeIn(task="write the essay"), Ctx().ctx)
    assert out.fallback is True and out.completed is False      # type: ignore[attr-defined]


async def test_daedalus_works_in_the_workspace_and_delivers_only_from_it(tmp_path: Path) -> None:
    claude = FakeClaude(
        {"scratch/chart.png": "png"},
        {"summary": "Plotted.", "completed": True,
         "deliver": ["scratch/chart.png", "../outside.txt", "missing.csv"]})
    (tmp_path.parent / "outside.txt").write_text("secret")
    ctx = Ctx({"data.csv": b"a,b\n1,2\n"})
    out = await ClaudeJobs(claude, tmp_path).code(ClaudeCodeIn(task="plot data.csv"), ctx.ctx)
    assert [name for name, _ in ctx.artifacts] == ["chart.png"]
    assert out.filename == "chart.png"                          # type: ignore[attr-defined]
    prompt, _, cwd = claude.calls[0]
    assert cwd == tmp_path and "DAEDALUS" in prompt
    inbox = next((tmp_path / "scratch").glob("inbox-*"))
    assert (inbox / "data.csv").read_bytes() == b"a,b\n1,2\n"
    assert inbox.name in prompt
