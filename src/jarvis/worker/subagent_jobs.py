# =============================================================================
# src/jarvis/worker/subagent_jobs.py - subagent job types, worker side
# =============================================================================
#
# The handlers for job types the Core knows only as metadata. This is
# where a subagent gets its actual capabilities: the worker's MCP tools,
# bound to the guard, handed to a loop.
#
# Note what the handler does NOT do: it does not choose PROTEUS's tools,
# and PROTEUS cannot widen them. The toolset is constructed here, guarded
# here, and passed in. An agent that could grant itself capabilities
# would make the allowlist decorative.
#
# Screenshots and reports are shipped as artifacts before the job
# reports success, so a vanished worker leaves nothing important behind.
# =============================================================================

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.agentloop.mcp_client import McpHost
from jarvis.agentloop.policies import ENGINEER, OPERATOR, create_guard
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.jobs.code import CodeTaskIn, CodeTaskOut
from jarvis.subagents.engineer import run_engineer
from jarvis.worker.sandbox.runner import run_python
from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings
from jarvis.core.queue.registry import (
    JobContext,
    JobTypeRegistry,
    JobTypeSpec,
    PermanentJobError,
)
from jarvis.jobs.browser import BrowserTaskIn, BrowserTaskOut
from jarvis.llm.layer import LLMLayer
from jarvis.subagents.operator import run_operator

log = get_logger("worker.subagent_jobs")


def register_subagent_jobs(
    registry: JobTypeRegistry,
    llm: LLMLayer,
    mcp: McpHost,
) -> None:
    """Register subagent job handlers on the worker."""
    guard = create_guard()

    async def browser_task(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, BrowserTaskIn)

        # The toolset is built HERE and bound to the guard HERE. PROTEUS
        # receives capabilities; it does not select them.
        toolset = Toolset(guard=guard, actor=OPERATOR)
        toolset.register_mcp_host(mcp)

        if not toolset.specs():
            # No browser server running means this machine advertised a
            # capability it does not have. Retrying will not conjure one.
            raise PermanentJobError(
                "no browser tools available on this worker - is the "
                "playwright MCP server configured and running?"
            )

        await ctx.progress("PROTEUS starting")
        outcome = await run_operator(
            llm, payload.task, toolset,
            trace_id=ctx.trace_id,
            max_steps=payload.max_steps,
        )

        # The full report ships as an artifact; the result carries a
        # chat-sized summary and a pointer.
        await ctx.write_artifact(
            "browser-report.md", "text/markdown",
            (
                f"# Browser task report\n\n"
                f"## Task\n{payload.task}\n\n"
                f"## Report\n{outcome.report}\n\n"
                f"---\nsteps: {outcome.steps_taken}, "
                f"tool calls: {outcome.tool_calls}\n"
            ).encode("utf-8"),
        )
        await ctx.progress("report written")

        summary = outcome.report.strip().split("\n\n")[0][:400]
        if outcome.hit_step_budget:
            summary = (
                f"Ran out of steps after {outcome.steps_taken}, sir. "
                f"{summary}"
            )

        return BrowserTaskOut(
            summary=summary,
            report=outcome.report,
            steps_taken=outcome.steps_taken,
            completed=not outcome.hit_step_budget,
        )

    registry.register(JobTypeSpec(
        type="browser.task",
        input_model=BrowserTaskIn,
        output_model=BrowserTaskOut,
        execution="resumable",
        requires=["browser"],
        timeout_s=900,
        handler=browser_task,
    ))

    async def code_task(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, CodeTaskIn)

        # Files persist across runs WITHIN this task, so the agent can
        # clean data in one step and analyse it in the next. Without
        # this every run would start from nothing and it could never
        # build on its own work. Dies with the task.
        workspace: dict[str, bytes] = {}

        class _RunCodeArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            code: str = Field(min_length=1)

        class _SaveArtifactArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            filename: str
            description: str = ""

        run_number = 0

        async def run_code(args: _RunCodeArgs) -> str:
            nonlocal run_number
            run_number += 1

            result = await run_python(args.code, input_files=dict(workspace))
            workspace.update(result.produced_files)

            # Save EVERY run, including the ones that failed. The final
            # working script says what worked; the sequence says why -
            # which attempt broke, what the error was, and what the fix
            # turned out to be. That is the part worth reading later.
            header = (
                f"# run {run_number} - "
                f"{'succeeded' if result.ok else f'exit {result.exit_code}'}\n"
                f"# task: {payload.task[:200]}\n\n"
            )
            await ctx.write_artifact(
                f"run-{run_number}.py", "text/x-python",
                (header + args.code).encode("utf-8"),
            )

            await ctx.progress(
                f"ran code ({'ok' if result.ok else 'error'})"
            )
            return result.summary()

        async def save_artifact(args: _SaveArtifactArgs) -> str:
            content = workspace.get(args.filename)
            if content is None:
                available = ", ".join(sorted(workspace)) or "(none)"
                return (
                    f"error: no file named {args.filename!r}. "
                    f"Files in the workspace: {available}"
                )
            await ctx.write_artifact(
                args.filename, _guess_mime(args.filename), content
            )
            return f"Saved {args.filename} for the owner."

        toolset = Toolset(guard=guard, actor=ENGINEER)
        toolset.register(InlineTool(
            name="sandbox_run_python",
            description=(
                "Run Python in an isolated sandbox and get back stdout, "
                "stderr, and any files it created. No network access. "
                "Files persist between calls within this task. pandas, "
                "numpy, matplotlib and scipy are available."
            ),
            args_model=_RunCodeArgs,
            handler=run_code,
        ))
        toolset.register(InlineTool(
            name="sandbox_save_artifact",
            description=(
                "Mark a file your code produced as something the owner "
                "should receive - a chart, a cleaned dataset, a document. "
                "Unsaved files are scratch and are discarded."
            ),
            args_model=_SaveArtifactArgs,
            handler=save_artifact,
        ))

        # Input artifacts land in the workspace before the first run:
        # this is the only way data reaches the sandbox, since it cannot
        # go looking for files.
        for name in payload.input_artifacts:
            path = Path(name)
            if path.exists():
                workspace[path.name] = path.read_bytes()

        await ctx.progress("DAEDALUS starting")
        outcome = await run_engineer(
            llm, payload.task, toolset,
            trace_id=ctx.trace_id,
            max_steps=payload.max_steps,
        )

        summary = outcome.report.strip().split("\n\n")[0][:400]
        if outcome.hit_step_budget:
            summary = f"Ran out of steps after {outcome.steps_taken}. {summary}"

        return CodeTaskOut(
            summary=summary,
            report=outcome.report,
            runs=outcome.runs,
            completed=not outcome.hit_step_budget,
        )

    registry.register(JobTypeSpec(
        type="code.task",
        input_model=CodeTaskIn,
        output_model=CodeTaskOut,
        execution="idempotent",
        requires=["sandbox-exec"],
        timeout_s=900,
        handler=code_task,
    ))

    log.info("subagent job types registered", extra={
        "types": ["browser.task", "code.task"],
    })


def _guess_mime(filename: str) -> str:
    """Enough to make Telegram render charts and CSVs sensibly."""
    suffix = Path(filename).suffix.lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
        ".csv": "text/csv", ".json": "application/json",
        ".md": "text/markdown", ".txt": "text/plain",
        ".pdf": "application/pdf", ".xlsx":
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(suffix, "application/octet-stream")