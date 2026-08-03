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
from jarvis.common.capabilities import Gate
from jarvis.core.queue.registry import PausedForApproval
from jarvis.worker.git.operations import GitOperations
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


def _needs_approval(
    ctx: JobContext,
    gate: Gate,
    tool: str,
    summary: str,
    detail: str,
    risk_note: str,
) -> PausedForApproval:
    """Build the exception that pauses a job for the owner's say-so.

    The request itself is raised through the job result rather than
    written directly: a WORKER cannot reach the Core's database, so it
    reports what it needs and the Core creates the approval. Returning
    the exception rather than raising it keeps the call site readable
    as `raise _needs_approval(...)`.
    """
    ctx.pending_approval = {
        "gate": gate.value,
        "actor": ENGINEER,
        "tool": tool,
        "summary": summary,
        "detail": detail,
        "risk_note": risk_note,
    }
    return PausedForApproval(summary)


def register_subagent_jobs(
    registry: JobTypeRegistry,
    llm: LLMLayer,
    mcp: McpHost,
    git_ops: "GitOperations | None" = None,
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

        # -- git tools, when configured ---------------------------------------
        #
        # Registered only if there is a token and an allowlist. A worker
        # without git configuration simply has no git tools, which the
        # model discovers by not seeing them rather than by being told
        # about a capability it cannot use.
        if git_ops is not None:
            class _RepoArgs(BaseModel):
                model_config = ConfigDict(extra="forbid")
                repo: str = Field(description="owner/name, e.g. 'you/project'")

            class _CommitArgs(BaseModel):
                model_config = ConfigDict(extra="forbid")
                repo: str
                message: str = Field(min_length=3)

            class _CreateRepoArgs(BaseModel):
                model_config = ConfigDict(extra="forbid")
                name: str = Field(min_length=1)
                description: str = ""
                private: bool = True

            async def git_clone(args: _RepoArgs) -> str:
                return (await git_ops.clone(args.repo)).summary()

            async def git_status(args: _RepoArgs) -> str:
                return (await git_ops.status(args.repo)).summary()

            async def git_commit(args: _CommitArgs) -> str:
                return (await git_ops.commit(args.repo, args.message)).summary()

            async def git_push(args: _RepoArgs) -> str:
                # THE GATE. Everything built since the guard - the
                # approval service, the Telegram buttons - exists for
                # this line. The job pauses here and resumes only if the
                # owner taps approve, possibly hours later.
                if not ctx.approval_granted:
                    status = await git_ops.status(args.repo)
                    await ctx.save_checkpoint({
                        "pending": "push", "repo": args.repo,
                    })
                    raise _needs_approval(
                        ctx=ctx,
                        gate=Gate.PUBLISH,
                        tool="git_push",
                        summary=f"Push commits to {args.repo}",
                        detail=(
                            f"Repository: {args.repo}\n\n"
                            f"Local state:\n{status.summary()[:1500]}"
                        ),
                        risk_note="This publishes to GitHub and cannot be undone.",
                    )
                return (await git_ops.push(args.repo)).summary()

            async def git_create_repo(args: _CreateRepoArgs) -> str:
                if not ctx.approval_granted:
                    await ctx.save_checkpoint({
                        "pending": "create_repo", "name": args.name,
                    })
                    raise _needs_approval(
                        ctx=ctx,
                        gate=Gate.PUBLISH,
                        tool="git_create_repo",
                        summary=f"Create repository {args.name}",
                        detail=(
                            f"Name: {args.name}\n"
                            f"Visibility: {'private' if args.private else 'PUBLIC'}\n"
                            f"Description: {args.description or '(none)'}"
                        ),
                        risk_note=(
                            "Creates a new repository on the owner's GitHub "
                            "account."
                        ),
                    )
                return (await git_ops.create_repo(
                    args.name, args.description, args.private
                )).summary()

            for name, description, model, handler in (
                ("git_clone",
                 "Clone or update an allowlisted repository into the local "
                 "workspace. Do this before reading or changing anything.",
                 _RepoArgs, git_clone),
                ("git_status",
                 "Show uncommitted changes and the current branch.",
                 _RepoArgs, git_status),
                ("git_commit",
                 "Stage everything and commit locally. Nothing leaves the "
                 "machine until a push.",
                 _CommitArgs, git_commit),
                ("git_push",
                 "Push commits to GitHub. PAUSES for the owner's approval.",
                 _RepoArgs, git_push),
                ("git_create_repo",
                 "Create a new GitHub repository. PAUSES for approval. "
                 "Private unless told otherwise.",
                 _CreateRepoArgs, git_create_repo),
            ):
                toolset.register(InlineTool(
                    name=name, description=description,
                    args_model=model, handler=handler,
                ))

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