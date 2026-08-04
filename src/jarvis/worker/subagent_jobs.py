# =============================================================================
# src/jarvis/worker/subagent_jobs.py - subagent handlers, worker side
# =============================================================================
#
# Where subagents get their actual capabilities: the worker's tools,
# bound to the guard, handed to a loop.
#
# Note what the handlers do NOT do: choose their agent's tools. The
# toolset is constructed here, guarded here, and passed in. An agent
# that could grant itself capabilities would make the allowlist
# decorative.
# =============================================================================

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.agentloop.mcp_client import McpHost
from jarvis.agentloop.policies import ENGINEER, OPERATOR, create_guard
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.common.capabilities import Gate
from jarvis.common.log import get_logger
from jarvis.core.queue.registry import (
    JobContext,
    JobTypeRegistry,
    JobTypeSpec,
    PausedForApproval,
    PermanentJobError,
)
from jarvis.jobs.browser import BrowserTaskIn, BrowserTaskOut
from jarvis.jobs.code import CodeTaskIn, CodeTaskOut
from jarvis.llm.layer import LLMLayer
from jarvis.subagents.engineer import run_engineer
from jarvis.subagents.operator import run_operator
from jarvis.worker.sandbox.runner import run_python
from jarvis.worker.tools.files import FileTools
from jarvis.worker.tools.github import GitTools
from jarvis.worker.workspace import Workspace

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

    A worker cannot write to the Core's database, so it describes what
    it wants and the runner relays it up the wire. Returning rather than
    raising keeps the call site readable as `raise _needs_approval(...)`.
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
    workspace: Workspace,
) -> None:
    """Register subagent job handlers on the worker."""
    guard = create_guard()
    files = FileTools(workspace)
    git = GitTools(workspace)

    # -- PROTEUS --------------------------------------------------------------

    async def browser_task(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, BrowserTaskIn)

        toolset = Toolset(guard=guard, actor=OPERATOR)
        toolset.register_mcp_host(mcp)

        if not toolset.specs():
            raise PermanentJobError(
                "no browser tools on this worker - is the playwright MCP "
                "server configured and running?"
            )

        await ctx.progress("PROTEUS starting")
        outcome = await run_operator(
            llm, payload.task, toolset,
            trace_id=ctx.trace_id, max_steps=payload.max_steps,
        )

        await ctx.write_artifact(
            "browser-report.md", "text/markdown",
            (
                f"# Browser task report\n\n## Task\n{payload.task}\n\n"
                f"## Report\n{outcome.report}\n\n---\n"
                f"steps: {outcome.steps_taken}, tool calls: {outcome.tool_calls}\n"
            ).encode("utf-8"),
        )

        summary = outcome.report.strip().split("\n\n")[0][:400]
        if outcome.hit_step_budget:
            summary = f"Ran out of steps after {outcome.steps_taken}. {summary}"

        return BrowserTaskOut(
            summary=summary, report=outcome.report,
            steps_taken=outcome.steps_taken,
            completed=not outcome.hit_step_budget,
        )

    registry.register(JobTypeSpec(
        type="browser.task",
        input_model=BrowserTaskIn, output_model=BrowserTaskOut,
        execution="resumable", requires=["browser"],
        timeout_s=900, handler=browser_task,
    ))

    # -- DAEDALUS -------------------------------------------------------------

    async def code_task(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, CodeTaskIn)

        # Where this task's execution happens. A project task runs inside
        # its project so scripts see the files being built; a scratch
        # task gets a fresh directory.
        run_dir = {"path": str(workspace.scratch_path())}

        # -- argument models ---------------------------------------------------

        class _NoArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")

        class _PathArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            path: str

        class _WriteArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            path: str
            content: str

        class _EditArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            path: str
            old_text: str = Field(min_length=1)
            new_text: str

        class _RunArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            code: str = Field(min_length=1)

        class _ProjectArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            name: str = Field(min_length=1, max_length=60)
            description: str = ""

        class _ProjectNameArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            project: str

        class _CommitArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            project: str
            message: str = Field(min_length=3)

        class _CreateRepoArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            project: str
            private: bool = True
            description: str = ""

        class _SaveArgs(BaseModel):
            model_config = ConfigDict(extra="forbid")
            path: str
            description: str = ""

        # -- handlers ----------------------------------------------------------

        async def create_project(args: _ProjectArgs) -> str:
            path = workspace.create_project(args.name, args.description)
            run_dir["path"] = str(path)
            result = await git.init(args.name)
            return (
                f"Created projects/{args.name}/ with README.md, .gitignore, "
                f"and src/. Git: {result.summary()[:120]}\n"
                f"Code you run now executes inside this project."
            )

        async def open_project(args: _ProjectNameArgs) -> str:
            try:
                path = workspace.project_path(args.project)
            except Exception as exc:
                return f"error: {exc}"
            if not path.exists():
                return f"error: no project named {args.project}."
            run_dir["path"] = str(path)
            return (
                f"Working in projects/{args.project}/.\n\n"
                + workspace.tree(f"projects/{args.project}")
            )

        async def list_projects(_: _NoArgs) -> str:
            projects = workspace.list_projects()
            if not projects:
                return "No projects yet."
            return "\n".join(
                f"{p['name']} | {p['files']} files | "
                f"{'git' if p['git'] else 'no git'} | {p['modified']}"
                for p in projects
            )

        async def read_file(args: _PathArgs) -> str:
            return files.read(args.path)

        async def write_file(args: _WriteArgs) -> str:
            return files.write(args.path, args.content)

        async def edit_file(args: _EditArgs) -> str:
            return files.edit(args.path, args.old_text, args.new_text)

        async def list_files(args: _PathArgs) -> str:
            return files.list_files(args.path)

        async def tree(args: _PathArgs) -> str:
            return files.tree(args.path)

        async def run_code(args: _RunArgs) -> str:
            """Execute Python inside the current working directory.

            The kernel sandbox still applies - no network, nothing
            outside this directory readable - but the directory PERSISTS,
            so a script can import a module the last one wrote.
            """
            directory = Path(run_dir["path"])
            result = await run_python(
                args.code,
                workdir=directory,
                input_files=ctx.input_files if not directory.exists() else None,
            )
            await ctx.progress(
                f"ran code ({'ok' if result.ok else 'error'})"
            )
            return result.summary()

        async def git_status(args: _ProjectNameArgs) -> str:
            return (await git.status(args.project)).summary()

        async def git_diff(args: _ProjectNameArgs) -> str:
            return (await git.diff(args.project)).summary()

        async def git_commit(args: _CommitArgs) -> str:
            return (await git.commit(args.project, args.message)).summary()

        async def git_push(args: _ProjectNameArgs) -> str:
            if not ctx.approval_granted:
                status = await git.status(args.project)
                log = await git.log(args.project, count=5)
                await ctx.save_checkpoint({
                    "pending": "push", "project": args.project,
                })
                raise _needs_approval(
                    ctx=ctx, gate=Gate.PUBLISH, tool="git_push",
                    summary=f"Push {args.project} to GitHub",
                    detail=(
                        f"Project: {args.project}\n\n"
                        f"Commits:\n{log.summary()[:800]}\n\n"
                        f"State:\n{status.summary()[:600]}"
                    ),
                    risk_note="This publishes to GitHub and cannot be undone.",
                )
            return (await git.push(args.project)).summary()

        async def git_create_repo(args: _CreateRepoArgs) -> str:
            if not ctx.approval_granted:
                await ctx.save_checkpoint({
                    "pending": "create_repo", "project": args.project,
                })
                raise _needs_approval(
                    ctx=ctx, gate=Gate.PUBLISH, tool="git_create_repo",
                    summary=f"Create GitHub repository {args.project}",
                    detail=(
                        f"Name: {args.project}\n"
                        f"Visibility: {'private' if args.private else 'PUBLIC'}\n"
                        f"Description: {args.description or '(none)'}\n\n"
                        f"{workspace.tree(f'projects/{args.project}')[:800]}"
                    ),
                    risk_note="Creates a repository on the owner's account.",
                )
            return (await git.create_repo(
                args.project, args.private, args.description
            )).summary()

        async def save_artifact(args: _SaveArgs) -> str:
            """Mark a file for delivery to the owner."""
            try:
                target = workspace.resolve_safe(args.path)
            except Exception as exc:
                return f"error: {exc}"
            if not target.exists() or not target.is_file():
                return f"error: {args.path} does not exist."
            await ctx.write_artifact(
                target.name, _guess_mime(target.name), target.read_bytes()
            )
            return f"Sent {target.name} to the owner."

        # -- registration ------------------------------------------------------

        toolset = Toolset(guard=guard, actor=ENGINEER)
        for name, description, model, handler in (
            ("sandbox_create_project",
             "Start a new project: folder, README, .gitignore, src/, and a "
             "git repo. Use for anything the owner will keep or extend.",
             _ProjectArgs, create_project),
            ("sandbox_open_project",
             "Work in an existing project. Shows its structure. Do this "
             "before changing anything in a project you did not just make.",
             _ProjectNameArgs, open_project),
            ("sandbox_list_projects",
             "Every project in the workspace.",
             _NoArgs, list_projects),
            ("file_read",
             "Read a file. Path is relative to the workspace, e.g. "
             "projects/my-thing/src/main.py",
             _PathArgs, read_file),
            ("file_write",
             "Create or overwrite a file, making directories as needed.",
             _WriteArgs, write_file),
            ("file_edit",
             "Replace an exact string in a file. Cheaper and safer than "
             "rewriting. The old text must appear exactly once.",
             _EditArgs, edit_file),
            ("file_list",
             "Names and sizes in one directory.",
             _PathArgs, list_files),
            ("file_tree",
             "The structure of a directory. Use this to orient before "
             "working in an existing project.",
             _PathArgs, tree),
            ("sandbox_run_python",
             "Run Python in the current working directory, sandboxed: no "
             "network, nothing outside the directory. Files persist, so "
             "a script can import what an earlier one wrote.",
             _RunArgs, run_code),
            ("git_status",
             "Uncommitted changes and the current branch.",
             _ProjectNameArgs, git_status),
            ("git_diff",
             "What has changed since the last commit. Read this before "
             "committing - it is how you notice unintended changes.",
             _ProjectNameArgs, git_diff),
            ("git_commit",
             "Stage everything and commit locally. Commit each working "
             "piece rather than everything at the end.",
             _CommitArgs, git_commit),
            ("git_push",
             "Push to GitHub. PAUSES for the owner's approval.",
             _ProjectNameArgs, git_push),
            ("git_create_repo",
             "Create a GitHub repository from a project and link it as "
             "origin. PAUSES for approval. Private unless told otherwise.",
             _CreateRepoArgs, git_create_repo),
            ("sandbox_save_artifact",
             "Send a file to the owner - a chart, a report, a dataset. "
             "Project source does not need this; he can open the project.",
             _SaveArgs, save_artifact),
        ):
            toolset.register(InlineTool(
                name=name, description=description,
                args_model=model, handler=handler,
            ))

        # Input files land in scratch so the first script can reach them.
        for filename, content in ctx.input_files.items():
            (Path(run_dir["path"]) / filename).write_bytes(content)

        await ctx.progress("DAEDALUS starting")
        outcome = await run_engineer(
            llm, payload.task, toolset,
            trace_id=ctx.trace_id, max_steps=payload.max_steps,
        )

        summary = outcome.report.strip().split("\n\n")[0][:400]
        if outcome.hit_step_budget:
            summary = f"Ran out of steps after {outcome.steps_taken}. {summary}"

        return CodeTaskOut(
            summary=summary, report=outcome.report,
            runs=outcome.tool_calls,
            completed=not outcome.hit_step_budget,
        )

    registry.register(JobTypeSpec(
        type="code.task",
        input_model=CodeTaskIn, output_model=CodeTaskOut,
        execution="idempotent", requires=["sandbox-exec"],
        timeout_s=1800, handler=code_task,
    ))

    log.info("subagent job types registered", extra={
        "types": ["browser.task", "code.task"],
    })


def _guess_mime(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
        ".csv": "text/csv", ".json": "application/json",
        ".md": "text/markdown", ".txt": "text/plain", ".py": "text/x-python",
        ".pdf": "application/pdf",
    }.get(suffix, "application/octet-stream")
