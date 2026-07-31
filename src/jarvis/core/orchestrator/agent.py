# =============================================================================
# src/jarvis/core/orchestrator/agent.py - the JARVIS configuration of the
# agent loop
# =============================================================================
#
# Still no loop mechanics here - only who JARVIS is, what he can do, and
# how hard he thinks.
#
# The toolset is now built PER TURN, not once at construction. Reason:
# delegation tools must know which conversation they serve, so the job
# they create can report back to the right thread. Session id and trace
# id are baked into the handlers as closures - which also makes
# concurrent conversations naturally safe, since no mutable "current
# turn" state is shared.
#
# Tools:
#   get_time_context - the clock, in the owner's timezone
#   run_subagent     - hand work to a specialist as a durable background
#                      job; returns immediately with a job id
#   list_jobs        - what work exists, by status
#   get_job          - one job in detail, including its result
#   cancel_job       - stop work the owner no longer wants
#
# enqueue_job (direct job-type dispatch from the registry catalogue)
# arrives when there are job types that are NOT subagent work - the
# memory sleep cycle and backups. With one job type registered, two tools
# doing the same thing would only confuse the model.
# =============================================================================

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jarvis.agentloop.loop import LoopResult, run_agent_loop
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.common.ids import is_ulid
from jarvis.common.jobs import Job, JobStatus
from jarvis.common.settings import CoreSettings
from jarvis.core.db.repos.events import EventsRepo
from jarvis.common.facts import FactCategory
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.memory.service import MemoryService
from jarvis.core.orchestrator.prompts import assemble_system_prompt
from jarvis.core.queue.dispatcher import cancel_job as queue_cancel_job
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.llm.layer import LLMLayer, TextCallback
from jarvis.llm.tiers import Tier

ACTOR = "core.orchestrator"

# Subagent name -> the job type that runs it. Grows one line per subagent.
_SUBAGENT_JOBS: dict[str, str] = {
    "researcher": "research.brief",
}


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _RunSubagentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: Literal["researcher"]
    brief: str = Field(
        min_length=10,
        description=(
            "A self-contained task description. It may be read hours later "
            "by someone with no access to this conversation."
        ),
    )
    priority: int = Field(default=5, ge=0, le=9)


class _ListJobsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    only_active: bool = Field(
        default=True,
        description="True lists work still in progress; False includes finished jobs.",
    )


class _JobIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str


class _MemoryStoreArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact: str = Field(
        min_length=3,
        description=(
            "One self-contained sentence that will make sense read alone "
            "in six months. Name the owner rather than writing 'he'."
        ),
    )
    category: Literal[
        "preference", "person", "project", "routine",
        "credential-ref", "world", "other",
    ] = "other"
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="0 trivial, 0.5 ordinary, 0.9 defining.",
    )


class _MemorySearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2)
    k: int = Field(default=6, ge=1, le=15)


class Orchestrator:
    """JARVIS's front mind: persona + tools + tier, applied per turn."""

    def __init__(
        self,
        llm: LLMLayer,
        settings: CoreSettings,
        jobs: JobsRepo,
        events: EventsRepo,
        registry: JobTypeRegistry,
        memory: MemoryService,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._jobs = jobs
        self._events = events
        self._registry = registry
        self._memory = memory

    # -- per-turn toolset -----------------------------------------------------

    def _build_toolset(self, session_id: str, trace_id: str) -> Toolset:
        tools = Toolset()

        async def get_time(_: _NoArgs) -> str:
            return datetime.now(self._settings.tz).strftime("%A %d %B %Y, %H:%M %Z")

        tools.register(InlineTool(
            name="get_time_context",
            description=(
                "The current date and time in the owner's timezone. Use "
                "whenever the answer depends on when 'now' is."
            ),
            args_model=_NoArgs,
            handler=get_time,
        ))

        async def run_subagent(args: _RunSubagentArgs) -> str:
            job_type = _SUBAGENT_JOBS[args.agent]
            spec = self._registry.get(job_type)
            if spec is None:
                return (
                    f"error: {args.agent} is not available in this build "
                    f"(job type {job_type} is not registered)"
                )
            job = Job(
                type=job_type,
                payload={"brief": args.brief},
                priority=args.priority,
                requires=spec.requires,
                session_id=session_id,   # so the result finds its way back
                trace_id=trace_id,       # one trace: turn -> job -> model calls
            )
            await self._jobs.create(job)
            return (
                f"Job {job.id} enqueued for {args.agent}. It runs in the "
                f"background; the owner will be told when it finishes. "
                f"Acknowledge briefly and move on - do not wait."
            )

        tools.register(InlineTool(
            name="run_subagent",
            description=(
                "Hand a task to a specialist as a background job. Returns "
                "immediately with a job id; the work continues without you. "
                "Use for anything taking more than a few seconds. "
                "'researcher' (ATHENA) handles web research and written briefs."
            ),
            args_model=_RunSubagentArgs,
            handler=run_subagent,
        ))

        async def list_jobs(args: _ListJobsArgs) -> str:
            if args.only_active:
                jobs = await self._jobs.live_jobs()
                if not jobs:
                    return "No jobs in progress."
            else:
                jobs = await self._jobs.for_session(session_id, limit=15)
                if not jobs:
                    return "No jobs on record for this conversation."
            lines = [
                f"{j.id} | {j.type} | {j.status}"
                + (f" | attempt {j.attempts}" if j.attempts > 1 else "")
                for j in jobs
            ]
            return "\n".join(lines)

        tools.register(InlineTool(
            name="list_jobs",
            description=(
                "List background jobs. Use when the owner asks what is "
                "running, queued, or recently finished."
            ),
            args_model=_ListJobsArgs,
            handler=list_jobs,
        ))

        async def get_job(args: _JobIdArgs) -> str:
            if not is_ulid(args.job_id):
                return f"error: {args.job_id!r} is not a valid job id"
            job = await self._jobs.get(args.job_id)
            if job is None:
                return f"No job with id {args.job_id}."
            parts = [f"{job.id} | {job.type} | {job.status}"]
            if job.error:
                parts.append(f"error: {job.error}")
            if job.result:
                summary = job.result.get("summary")
                parts.append(
                    f"result summary: {summary}" if summary
                    else f"result keys: {', '.join(job.result)}"
                )
            return "\n".join(parts)

        tools.register(InlineTool(
            name="get_job",
            description=(
                "Details of one job including its result or failure reason. "
                "Use when the owner asks about a specific piece of work."
            ),
            args_model=_JobIdArgs,
            handler=get_job,
        ))

        async def cancel_job(args: _JobIdArgs) -> str:
            if not is_ulid(args.job_id):
                return f"error: {args.job_id!r} is not a valid job id"
            cancelled = await queue_cancel_job(
                self._jobs, self._events, args.job_id
            )
            if cancelled:
                return f"Job {args.job_id} cancelled."
            return (
                f"Job {args.job_id} could not be cancelled - it is already "
                f"finished, or no such job exists."
            )

        tools.register(InlineTool(
            name="cancel_job",
            description="Cancel a job the owner no longer wants.",
            args_model=_JobIdArgs,
            handler=cancel_job,
        ))

        async def memory_store(args: _MemoryStoreArgs) -> str:
            fact = await self._memory.store(
                text=args.fact,
                category=FactCategory(args.category),
                importance=args.importance,
            )
            return f"Stored as fact {fact.id}."

        tools.register(InlineTool(
            name="memory_store",
            description=(
                "Remember one durable fact about the owner's world. Use "
                "when he shares something worth keeping, or asks you to "
                "remember. Not for passing chatter or one-off logistics."
            ),
            args_model=_MemoryStoreArgs,
            handler=memory_store,
        ))

        async def memory_search(args: _MemorySearchArgs) -> str:
            hits = await self._memory.search(args.query, k=args.k)
            if not hits:
                return "Nothing in memory matches that."
            return "\n".join(f"- {h.fact.text}" for h in hits)

        tools.register(InlineTool(
            name="memory_search",
            description=(
                "Search everything known about the owner. Relevant facts "
                "are already provided each turn, so use this only when "
                "something older or more specific is needed than what is "
                "in front of you."
            ),
            args_model=_MemorySearchArgs,
            handler=memory_search,
        ))

        return tools

    # -- one reply ------------------------------------------------------------

    async def respond(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        rolling_summary: str,
        trace_id: str,
        profile_doc: str = "",
        retrieved_memory: str = "",
        on_text: TextCallback | None = None,
    ) -> LoopResult:
        """Produce one reply to an assembled conversation. The session
        manager owns storage, retrieval, and context; this owns only
        thinking."""
        system = assemble_system_prompt(
            profile_doc=profile_doc,
            rolling_summary=rolling_summary,
            retrieved_memory=retrieved_memory,
        )
        
        return await run_agent_loop(
            self._llm,
            Tier.REASONER,
            system,
            messages,
            self._build_toolset(session_id, trace_id),
            actor=ACTOR,
            trace_id=trace_id,
            on_text=on_text,
            max_iterations=8,
        )