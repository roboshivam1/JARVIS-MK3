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

from pydantic import BaseModel

from jarvis.agentloop.mcp_client import McpHost
from jarvis.agentloop.policies import OPERATOR, create_guard
from jarvis.agentloop.toolset import Toolset
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
    log.info("subagent job types registered", extra={"types": ["browser.task"]})