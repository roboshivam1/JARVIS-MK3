# =============================================================================
# src/jarvis/core/orchestrator/agent.py - the JARVIS configuration of the
# agent loop
# =============================================================================
#
# No loop logic lives here - that is agentloop/'s job. This file only
# answers three questions and delegates:
#
#   who am I        -> the persona prompt (prompts.py)
#   what can I do   -> the inline toolset built below
#   how do I think  -> the reasoner tier, with a bounded iteration budget
#
# Subagents in later phases are files shaped exactly like this one with
# different answers. If this file ever grows loop mechanics, something
# has gone architecturally wrong.
#
# Current toolset (grows with the phases):
#   get_time_context - the real clock, in the owner's timezone
#   list_jobs        - honest stub: reports that the job system does not
#                      exist yet. A stub the model can CALL and get truth
#                      from beats a prompt line promising a future feature.
# =============================================================================

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from jarvis.agentloop.loop import LoopResult, run_agent_loop
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.common.settings import CoreSettings
from jarvis.core.orchestrator.prompts import assemble_system_prompt
from jarvis.llm.layer import LLMLayer, TextCallback
from jarvis.llm.tiers import Tier

ACTOR = "core.orchestrator"


class _NoArgs(BaseModel):
    """Schema for tools that take nothing. extra=forbid so a model
    hallucinating arguments gets corrected instead of ignored."""

    model_config = ConfigDict(extra="forbid")


class Orchestrator:
    """JARVIS's front mind: persona + tools + tier, applied per turn."""

    def __init__(self, llm: LLMLayer, settings: CoreSettings) -> None:
        self._llm = llm
        self._settings = settings
        self._tools = self._build_toolset()

    def _build_toolset(self) -> Toolset:
        tools = Toolset()

        async def get_time(_: _NoArgs) -> str:
            now = datetime.now(self._settings.tz)
            return now.strftime("%A %d %B %Y, %H:%M %Z")

        tools.register(InlineTool(
            name="get_time_context",
            description=(
                "The current date and time in the owner's timezone. Use "
                "whenever the answer depends on when 'now' is."
            ),
            args_model=_NoArgs,
            handler=get_time,
        ))

        async def list_jobs(_: _NoArgs) -> str:
            # Honest stub until the job queue phase. The model reads this
            # verbatim and can relay the truth instead of inventing jobs.
            return (
                "The background job system is not built yet. There are no "
                "jobs to list. Say so plainly if asked."
            )

        tools.register(InlineTool(
            name="list_jobs",
            description=(
                "List background jobs and their status. Use when the owner "
                "asks what work is queued, running, or finished."
            ),
            args_model=_NoArgs,
            handler=list_jobs,
        ))

        return tools

    async def respond(
        self,
        messages: list[dict[str, Any]],
        *,
        rolling_summary: str,
        trace_id: str,
        on_text: TextCallback | None = None,
    ) -> LoopResult:
        """Produce one reply to an assembled conversation. The session
        manager owns storage and context; this owns only thinking."""
        system = assemble_system_prompt(
            rolling_summary=rolling_summary,
            # profile_doc arrives with the memory phase
        )
        return await run_agent_loop(
            self._llm,
            Tier.REASONER,
            system,
            messages,
            self._tools,
            actor=ACTOR,
            trace_id=trace_id,
            on_text=on_text,
            max_iterations=8,
        )