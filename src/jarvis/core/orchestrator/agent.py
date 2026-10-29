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
#   reels_recent, reel_details, reel_request, reel_mark_posted
#                    - his own Instagram channel; see core/reels/service.py
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
from jarvis.agentloop.guard import Guard
from jarvis.agentloop.policies import create_guard
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.common.ids import is_ulid
from jarvis.common.jobs import Job, JobStatus
from jarvis.common.settings import CoreSettings
from jarvis.core.db.repos.events import EventsRepo
from zoneinfo import ZoneInfo

from jarvis.common.facts import FactCategory
from jarvis.common.schedules import Schedule, ScheduleKind
from jarvis.common.watchers import Watcher, WatcherKind
from jarvis.core.db.repos.schedules import SchedulesRepo
from jarvis.core.db.repos.watchers import WatchersRepo
from jarvis.core.initiative.engine import next_cron_time
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.memory.service import MemoryService
from jarvis.core.orchestrator.prompts import assemble_system_prompt
from jarvis.core.people.service import ExtractionFailed, PeopleService
from jarvis.core.projects.service import ExtractionFailed as ProjectExtractionFailed
from jarvis.core.projects.service import ProjectService
from jarvis.core.life.service import LifeService
from jarvis.core.money.plutus import ask_plutus
from jarvis.core.money.service import MoneyService
from jarvis.core.life.service import ReportFailed as LifeReportFailed
from jarvis.jobs.projects import DOCUMENT_JOB_TYPE as PROJECT_DOCUMENT_JOB
from jarvis.core.reels.service import BriefRejected, ReelBusy, ReelService
from jarvis.core.queue.dispatcher import cancel_job as queue_cancel_job
from jarvis.core.queue.claude_code import ClaudeCodeRouter
from jarvis.jobs.claude_code import ROUTES
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.llm.layer import LLMLayer, TextCallback
from jarvis.llm.tiers import Tier

ACTOR = "core.orchestrator"

# Subagent name -> the job type that runs it. Grows one line per subagent.
_SUBAGENT_JOBS: dict[str, str] = {
    "researcher": "research.brief",
    "operator": "browser.task",
    "engineer": "code.task",
    "writer": "write.document",
}

# The Claude Code versions, for saying which way a job went.
_CLAUDE_CODE_TYPES = frozenset(cc for cc, _ in ROUTES.values())

class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _RunSubagentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: Literal["researcher", "operator", "engineer", "writer"]
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


class _CreateWatcherArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=3, max_length=60,
        description=(
            "A short descriptive name the owner will recognise weeks "
            "later, e.g. 'sqonion patent status'. Not 'watcher 1'."
        ),
    )
    kind: Literal["web_page", "spend", "job_health", "idle"]
    config: dict[str, Any] = Field(
        description=(
            "Depends on kind. web_page: {'url': '...'}. "
            "spend: {'threshold_inr': 200}. "
            "job_health: {'job_type': 'research.brief', "
            "'consecutive_failures': 2}. "
            "idle: {'event_kind': 'session.turn_user', 'max_idle_hours': 72}."
        ),
    )
    priority: int = Field(
        default=5, ge=0, le=9,
        description=(
            "0-2 reaches him at any hour - use only when work is blocked "
            "or money is involved. 3-5 during waking hours. 6-9 batched "
            "into a digest, which is right for most page-change watchers."
        ),
    )
    note: str = Field(
        default="",
        description=(
            "What the owner actually asked for, in his words. Sent with "
            "every notification, so a hit six weeks later still makes "
            "sense on its own."
        ),
    )


class _WatcherNameArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class _ScheduleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=60)
    cron: str = Field(
        description=(
            "Standard cron in the owner's timezone: '30 3 * * *' is "
            "03:30 daily, '0 9 * * 1' is 09:00 every Monday."
        ),
    )
    job_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class _PeopleLookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        description="A name, nickname, or alias, as the owner said it.",
    )


class _PeopleUpcomingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days: int = Field(default=30, ge=1, le=366,
                      description="How many days ahead to look for dates.")


class _PeopleDebriefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=10,
        description=(
            "What the owner said about the person or people, in his "
            "own words and in full. Do not summarise it."
        ),
    )


class _ProjectNameArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1,
                      description="The project's name or another name he uses for it, "
                                  "or words describing it.")


class _ProjectDebriefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=10,
        description=("What the owner said about his project work, in his own "
                     "words and in full. Do not summarise it."),
    )


class _ProjectNoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="The project's exact name.")
    text: str = Field(min_length=2, description="The log line, as he dictated it.")


class _LifeReportArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=2,
        description=("What he said about his day, in his own words and in full: "
                     "waking, sleep, gym, meals, study, project work, learning, "
                     "things done, targets for another day."),
    )


class _LifeDayArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$",
                             description="YYYY-MM-DD; leave empty for today.")


class _PlutusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(
        min_length=3,
        description=("The owner's question or instruction about his money, in full: "
                     "'how much on food this month?', 'Campus P is the canteen, food', "
                     "'I paid 200 cash for chai'."),
    )


class _ReelRefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str = Field(
        default="last",
        description="The reel's ref (six characters), or 'last' for the newest.",
    )


class _ReelRequestArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea: str = Field(
        min_length=5,
        description=(
            "What the reel is about, in the owner's words. The brief "
            "writer turns it into a premise and beats."
        ),
    )


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
        guard: Guard | None = None,
        artifacts: "ArtifactsRepo | None" = None,
        watchers: "WatchersRepo | None" = None,
        schedules: "SchedulesRepo | None" = None,
        tz: "ZoneInfo | None" = None,
        people: "PeopleService | None" = None,
        reels: "ReelService | None" = None,
        router: "ClaudeCodeRouter | None" = None,
        projects: "ProjectService | None" = None,
        life: "LifeService | None" = None,
        money: "MoneyService | None" = None,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._jobs = jobs
        self._events = events
        self._registry = registry
        self._memory = memory
        self._guard = guard or create_guard()
        self._artifacts = artifacts
        self._watchers = watchers
        self._schedules = schedules
        self._tz = tz or settings.tz
        self._people = people
        self._reels = reels
        self._router = router
        self._projects = projects
        self._life = life
        self._money = money

    # -- per-turn toolset -----------------------------------------------------

    def _build_toolset(self, session_id: str, trace_id: str) -> Toolset:
        tools = Toolset(guard=self._guard, actor=ACTOR)

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
            # The operator's payload field is "task", not "brief".
            payload_key = (
                "brief" if args.agent in ("researcher", "writer") else "task"
            )
            payload: dict[str, Any] = {payload_key: args.brief}
            # Claude Code on the Mac when it is online, the API otherwise.
            routed = (self._router.route(args.agent, args.brief)
                      if self._router is not None else None)
            if routed is not None:
                job_type, payload = routed
            via = (" via Claude Code on the Mac" if job_type in _CLAUDE_CODE_TYPES
                   else "")
            spec = self._registry.get(job_type)
            if spec is None:
                return (
                    f"error: {args.agent} is not available in this build "
                    f"(job type {job_type} is not registered)"
                )
            job = Job(
                type=job_type,
                payload=payload,
                priority=args.priority,
                requires=spec.requires,
                session_id=session_id,   # so the result finds its way back
                trace_id=trace_id,       # one trace: turn -> job -> model calls
            )
            await self._jobs.create(job)
            return (
                f"Job {job.id} enqueued for {args.agent}{via}. It runs in the "
                f"background; the owner will be told when it finishes. "
                f"Acknowledge briefly and move on - do not wait."
            )

        tools.register(InlineTool(
            name="run_subagent",
            description=(
                "Hand a task to a specialist as a background job. Returns "
                "immediately with a job id; the work continues without you. "
                "Use for anything taking more than a few seconds. "
                "'researcher' (ATHENA) handles web research and written "
                "briefs from search results. 'operator' (PROTEUS) drives a "
                "real browser on the owner's machine - use for pages that "
                "need JavaScript, a login, or interaction, and say plainly "
                "that it needs a worker online. 'engineer' (DAEDALUS) "
                "writes and runs Python in a sandbox - use for "
                "calculations, data analysis, charts, and anything where "
                "the answer should be computed rather than recalled. "
                "'writer' (CALLIOPE) produces documents - articles, "
                "reports, long-form pieces. Use when the deliverable is "
                "the writing itself rather than the research behind it."
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
                "remember. Not for passing chatter or one-off logistics. "
                "Facts about other people go to people_debrief instead, "
                "and work on his projects to project_debrief, when those "
                "tools are available."
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

        # -- watchers ---------------------------------------------------------
        #
        # The first thing JARVIS creates that OUTLIVES the conversation.
        # Everything else is a job: it runs, finishes, done. A watcher
        # persists and acts on its own for months, which is why the tool
        # asks for a real name and a note about what was wanted.
        if self._watchers is not None:

            async def create_watcher(args: _CreateWatcherArgs) -> str:
                created = await self._watchers.create(Watcher(
                    name=args.name,
                    kind=WatcherKind(args.kind),
                    config=args.config,
                    priority=args.priority,
                    note=args.note,
                ))
                if not created:
                    return (
                        f"A watcher named {args.name!r} already exists. "
                        f"Pick another name, or remove that one first."
                    )
                return (
                    f"Watching: {args.name}. Checked every fifteen minutes. "
                    f"The first check records a baseline, so the owner will "
                    f"hear about changes from the second check onward."
                )

            tools.register(InlineTool(
                name="create_watcher",
                description=(
                    "Keep an eye on something and tell the owner when it "
                    "changes. Checked every fifteen minutes, indefinitely. "
                    "Use for standing requests - 'tell me if X changes' - "
                    "not for one-off checks, which are jobs. Ask what he "
                    "actually wants to know about if the request is vague: "
                    "a badly aimed watcher fires uselessly forever."
                ),
                args_model=_CreateWatcherArgs,
                handler=create_watcher,
            ))

            async def list_watchers(_: _NoArgs) -> str:
                watchers = await self._watchers.all()
                if not watchers:
                    return "Nothing is being watched."
                lines = []
                for w in watchers:
                    state = "on" if w.enabled else "paused"
                    seen = (
                        w.last_checked_ts.strftime("%d %b %H:%M")
                        if w.last_checked_ts else "never"
                    )
                    lines.append(
                        f"{w.name} | {w.kind.value} | {state} | "
                        f"{w.hit_count} hits | last checked {seen}"
                    )
                return "\n".join(lines)

            tools.register(InlineTool(
                name="list_watchers",
                description="Everything currently being watched.",
                args_model=_NoArgs,
                handler=list_watchers,
            ))

            async def remove_watcher(args: _WatcherNameArgs) -> str:
                if await self._watchers.delete(args.name):
                    return f"Stopped watching {args.name}."
                return f"No watcher named {args.name!r}."

            tools.register(InlineTool(
                name="remove_watcher",
                description="Stop watching something, permanently.",
                args_model=_WatcherNameArgs,
                handler=remove_watcher,
            ))

        # -- schedules --------------------------------------------------------
        #
        # Owed since the scheduler was built: the machinery existed but
        # nothing could create a schedule except boot-time seeding.
        if self._schedules is not None:

            async def create_schedule(args: _ScheduleArgs) -> str:
                spec = self._registry.get(args.job_type)
                if spec is None:
                    available = ", ".join(s.type for s in self._registry.catalogue())
                    return (
                        f"No job type {args.job_type!r}. Available: {available}"
                    )
                try:
                    first = next_cron_time(args.cron, self._tz)
                except Exception as exc:
                    return f"That cron expression is not valid: {exc}"

                created = await self._schedules.ensure(Schedule(
                    name=args.name,
                    kind=ScheduleKind.CRON,
                    cron_expr=args.cron,
                    job_type=args.job_type,
                    job_payload=args.payload,
                    next_fire_ts=first,
                ))
                if not created:
                    return f"A schedule named {args.name!r} already exists."
                return (
                    f"Scheduled: {args.name}, first run "
                    f"{first.astimezone(self._tz).strftime('%d %b at %H:%M')}."
                )

            tools.register(InlineTool(
                name="create_schedule",
                description=(
                    "Run a job repeatedly on a cron schedule. For standing "
                    "routines - a nightly report, a weekly check. Use a "
                    "watcher instead when the point is noticing a change."
                ),
                args_model=_ScheduleArgs,
                handler=create_schedule,
            ))

            async def list_schedules(_: _NoArgs) -> str:
                schedules = await self._schedules.all()
                if not schedules:
                    return "Nothing is scheduled."
                return "\n".join(
                    f"{s.name} | {s.job_type} | "
                    f"{'on' if s.enabled else 'off'} | "
                    f"next {s.next_fire_ts.astimezone(self._tz).strftime('%d %b %H:%M')}"
                    for s in schedules
                )

            tools.register(InlineTool(
                name="list_schedules",
                description="Everything on a recurring schedule.",
                args_model=_NoArgs,
                handler=list_schedules,
            ))

        # -- people -----------------------------------------------------------
        #
        # The people database. Lookup and upcoming read it; debrief only
        # ever STAGES proposals - nothing about a person is written until
        # the owner approves it on the review page.
        if self._people is not None:
            people = self._people

            async def people_lookup(args: _PeopleLookupArgs) -> str:
                return await people.lookup(args.name)

            tools.register(InlineTool(
                name="people_lookup",
                description=(
                    "Everything known about someone in the owner's life, by "
                    "name or nickname. People he mentions are already in front "
                    "of you each turn, minus sensitive details; use this when "
                    "he asks about someone directly or you need more. The "
                    "result includes sensitive facts - use them with discretion."
                ),
                args_model=_PeopleLookupArgs,
                handler=people_lookup,
            ))

            async def people_upcoming(args: _PeopleUpcomingArgs) -> str:
                return await people.upcoming(args.days)

            tools.register(InlineTool(
                name="people_upcoming",
                description=(
                    "Open commitments in both directions - what the owner owes "
                    "people and what they owe him - and birthdays or other "
                    "dates coming up."
                ),
                args_model=_PeopleUpcomingArgs,
                handler=people_upcoming,
            ))

            async def people_debrief(args: _PeopleDebriefArgs) -> str:
                try:
                    outcome = await people.debrief(
                        args.text, trace_id=trace_id, source="conversation")
                except ExtractionFailed as exc:
                    return f"The debrief could not be processed: {exc}"
                return outcome.summary() + " The owner reviews them at /app/people."

            tools.register(InlineTool(
                name="people_debrief",
                description=(
                    "Record what the owner tells you about people: someone he "
                    "met, news about a friend, something someone promised. "
                    "Pass his words in full. This only stages proposals for "
                    "his review - tell him they are waiting. Use this, not "
                    "memory_store, for facts about other people."
                ),
                args_model=_PeopleDebriefArgs,
                handler=people_debrief,
            ))

        # -- projects ---------------------------------------------------------
        #
        # The projects database. Same rule as people: the debrief tool only
        # STAGES proposals for the owner's review. A note is the one direct
        # write - a log line he dictated himself, word for word.
        if self._projects is not None:
            projects = self._projects

            async def projects_list(_: _NoArgs) -> str:
                return await projects.overview()

            tools.register(InlineTool(
                name="projects_list",
                description="Every project the owner has, with status and open work.",
                args_model=_NoArgs,
                handler=projects_list,
            ))

            async def project_lookup(args: _ProjectNameArgs) -> str:
                return await projects.lookup(args.name)

            tools.register(InlineTool(
                name="project_lookup",
                description=(
                    "Everything recorded about one project: decisions and why, "
                    "facts, parts, open tasks, people, recent log. Projects he "
                    "names are summarised in front of you already; use this for "
                    "the full record or to find one by description."
                ),
                args_model=_ProjectNameArgs,
                handler=project_lookup,
            ))

            async def project_debrief(args: _ProjectDebriefArgs) -> str:
                try:
                    outcome = await projects.debrief(
                        args.text, trace_id=trace_id, source="conversation")
                except ProjectExtractionFailed as exc:
                    return f"That could not be processed: {exc}"
                return outcome.summary() + " The owner reviews them at /app/projects."

            tools.register(InlineTool(
                name="project_debrief",
                description=(
                    "Record what the owner tells you about his projects: work "
                    "done, decisions and the reasons, parts bought or needed, "
                    "problems, things still to do, a new project. Pass his words "
                    "in full. This only stages proposals for his review - tell "
                    "him they are waiting."
                ),
                args_model=_ProjectDebriefArgs,
                handler=project_debrief,
            ))

            async def project_note(args: _ProjectNoteArgs) -> str:
                return await projects.note(args.name, args.text)

            tools.register(InlineTool(
                name="project_note",
                description=(
                    "Add one line to a project's log straight away, without "
                    "review. ONLY when he explicitly says to log or note "
                    "something on a project; otherwise use project_debrief."
                ),
                args_model=_ProjectNoteArgs,
                handler=project_note,
            ))

            spec = self._registry.get(PROJECT_DOCUMENT_JOB)
            if spec is not None:

                async def project_document(args: _ProjectNameArgs) -> str:
                    found = await projects.find(args.name)
                    if len(found) != 1:
                        return (f"No project called {args.name!r}." if not found else
                                f"{len(found)} projects match {args.name!r} - be exact.")
                    job = Job(
                        type=PROJECT_DOCUMENT_JOB,
                        payload={"project_id": found[0]["id"]},
                        requires=spec.requires,
                        session_id=session_id,
                        trace_id=trace_id,
                    )
                    await self._jobs.create(job)
                    return (f"Job {job.id}: writing the master document for "
                            f"{found[0]['name']}. He will be sent it when it is done.")

                tools.register(InlineTool(
                    name="project_document",
                    description=(
                        "Write a new version of a project's master document - a "
                        "patent-style write-up of everything recorded about it. "
                        "Runs in the background."
                    ),
                    args_model=_ProjectNameArgs,
                    handler=project_document,
                ))

        # -- the life log -----------------------------------------------------
        #
        # His own record of his days. Reports go straight in (no review),
        # and the last one can always be undone.
        if self._life is not None:
            life = self._life

            async def life_report(args: _LifeReportArgs) -> str:
                try:
                    outcome = await life.report(args.text, trace_id=trace_id)
                except LifeReportFailed as exc:
                    return f"That could not be logged: {exc}"
                return outcome.summary()

            tools.register(InlineTool(
                name="life_report",
                description=(
                    "Log what the owner reports about his day: when he woke or "
                    "slept, gym, meals, study, project work, learning, checklist "
                    "items done, and targets he sets for a day ('tomorrow I want "
                    "to finish a Coursera module'). Use it whenever he reports "
                    "any of these, including in reply to the evening check-in. "
                    "Pass his words in full. Confirm briefly what was logged."
                ),
                args_model=_LifeReportArgs,
                handler=life_report,
            ))

            async def life_day(args: _LifeDayArgs) -> str:
                return await life.status_text(args.date)

            tools.register(InlineTool(
                name="life_day",
                description="One day's checklist, wake and sleep, time logged and meals.",
                args_model=_LifeDayArgs,
                handler=life_day,
            ))

            async def life_week(_: _NoArgs) -> str:
                return await life.week_text()

            tools.register(InlineTool(
                name="life_week",
                description=("How the last seven days went: checklist completion, gym "
                             "days, wake times, hours on study, projects and learning."),
                args_model=_NoArgs,
                handler=life_week,
            ))

            async def life_undo(_: _NoArgs) -> str:
                return await life.undo_last()

            tools.register(InlineTool(
                name="life_undo",
                description="Take back the last thing logged, when he says it was wrong.",
                args_model=_NoArgs,
                handler=life_undo,
            ))

        # -- money ------------------------------------------------------------
        #
        # PLUTUS keeps the books behind its own wall. JARVIS asks; the
        # ledger itself never enters this conversation.
        if self._money is not None:
            money = self._money

            async def plutus(args: _PlutusArgs) -> str:
                return await ask_plutus(self._llm, money, args.request, trace_id=trace_id)

            tools.register(InlineTool(
                name="ask_plutus",
                description=(
                    "PLUTUS keeps the owner's books: payments from his bank account "
                    "(read from the bank's alerts) and cash he mentions. Ask it any "
                    "question about his spending, and pass on any money instruction - "
                    "what a payee is, a cash payment, a correction. Pass his words in "
                    "full and relay its answer. Income is not tracked."
                ),
                args_model=_PlutusArgs,
                handler=plutus,
            ))

        # -- reels ------------------------------------------------------------
        #
        # His own channel. The newest reels are already in his context
        # each turn; these look further, start one, or log it as posted.
        # Starting one is safe to offer: the draft still comes back to
        # the owner for review before anything is rendered.
        if self._reels is not None:
            reels = self._reels

            async def reels_recent(_: _NoArgs) -> str:
                return await reels.status_text(limit=10)

            tools.register(InlineTool(
                name="reels_recent",
                description="The episode log of your channel: recent reels and their state.",
                args_model=_NoArgs,
                handler=reels_recent,
            ))

            async def reel_details(args: _ReelRefArgs) -> str:
                return await reels.describe(args.ref)

            tools.register(InlineTool(
                name="reel_details",
                description=(
                    "Everything about one reel: what was asked for, the "
                    "premise and beats, the script, Sir's notes, the film. "
                    "Use for an older reel, or when the summary in front of "
                    "you is not enough."
                ),
                args_model=_ReelRefArgs,
                handler=reel_details,
            ))

            async def reel_request(args: _ReelRequestArgs) -> str:
                try:
                    outcome = await reels.request(args.idea, record=False)
                except ReelBusy as busy:
                    e = busy.episode
                    return (
                        f"Refused: one reel at a time, and {e.title or e.ref} "
                        f"(ref {e.ref}) is {e.status.value.replace('_', ' ')}."
                    )
                except BriefRejected as exc:
                    return f"Refused: {exc}"
                e = outcome.episode
                reply = (
                    f"Briefed reel {e.ref}: {e.brief.premise} Drafting on the "
                    f"Mac; the draft comes to the owner on Telegram for review."
                )
                if not outcome.worker_online:
                    reply += " The Mac is not connected, so it starts when it is."
                return reply

            tools.register(InlineTool(
                name="reel_request",
                description=(
                    "Start a new reel for your channel when the owner asks "
                    "for one. One at a time; the draft goes to him for "
                    "review before anything is rendered."
                ),
                args_model=_ReelRequestArgs,
                handler=reel_request,
            ))

            async def reel_mark_posted(args: _ReelRefArgs) -> str:
                return await reels.mark_posted(args.ref, record=False)

            tools.register(InlineTool(
                name="reel_mark_posted",
                description=(
                    "Record that the owner has posted a delivered reel. Only "
                    "when he says it is up - you never post anything yourself."
                ),
                args_model=_ReelRefArgs,
                handler=reel_mark_posted,
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