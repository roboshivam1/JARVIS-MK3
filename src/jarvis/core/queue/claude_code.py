# =============================================================================
# src/jarvis/core/queue/claude_code.py - which way a subagent job runs
# =============================================================================
#
# DECIDED AT CREATION. When JARVIS hands work to DAEDALUS, ATHENA or
# CALLIOPE, the router looks at the live fleet: a worker advertising
# "claude-code" is online -> the Claude Code job type; otherwise the API
# one. "Online" means connected, not idle - a busy Mac still takes the
# job next, and that is cheaper than paying for the API now.
#
# AND CORRECTED AFTERWARDS. A loop sweeps for Claude Code jobs that will
# not finish, and starts the API version in their place:
#
#   - the job came back with fallback=true: the owner's plan ran out of
#     usage mid-task. The owner has already been told ("handing this to
#     the API") by the normal finished-job message.
#   - the job is still queued and no Claude Code worker has been online
#     for a while: the Mac went to sleep with work waiting. Withdrawn,
#     and restarted on the API - but only for subagents the Core can run
#     itself. DAEDALUS's API version needs the Mac's sandbox too, so his
#     work simply waits for the Mac.
#
# The replacement is a child job (parent_job_id) in the same session and
# trace, so its result is reported in the same conversation.
# =============================================================================

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from jarvis.common.ids import utc_now
from jarvis.common.jobs import Job, JobStatus
from jarvis.common.log import get_logger
from jarvis.core.db.database import Database
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.core.queue.registry_workers import WorkerRegistry
from jarvis.jobs.claude_code import CLAUDE_CODE, ROUTES

log = get_logger("core.queue.claude_code")

_SWEEP_INTERVAL_S = 30.0
STUCK_AFTER = timedelta(minutes=10)

# Claude Code job type -> API job type.
_API_FOR = {cc: api for cc, api in ROUTES.values()}


def api_payload(api_type: str, task: str, files: list[str]) -> dict[str, Any]:
    """The API job's payload: the same brief, in that job's field names."""
    if api_type == "code.task":
        return {"task": task, "input_artifacts": files}
    if api_type == "write.document":
        return {"brief": task, "source_artifacts": files}
    return {"brief": task}


class ClaudeCodeRouter:
    def __init__(
        self,
        db: Database,
        jobs: JobsRepo,
        workers: WorkerRegistry,
        registry: JobTypeRegistry,
        *,
        stuck_after: timedelta = STUCK_AFTER,
    ) -> None:
        self._db = db
        self._jobs = jobs
        self._workers = workers
        self._registry = registry
        self._stuck_after = stuck_after
        self._offline_since: datetime | None = None

    # -- at creation ----------------------------------------------------------

    def online(self) -> bool:
        return any(CLAUDE_CODE in w.capabilities for w in self._workers.connected())

    def route(self, agent: str, task: str, files: list[str] | None = None,
              ) -> tuple[str, dict[str, Any]] | None:
        """(job type, payload) for this agent, or None when the agent has
        no Claude Code version - the caller then does what it always did."""
        pair = ROUTES.get(agent)
        if pair is None:
            return None
        cc_type, api_type = pair
        files = files or []
        if self.online() and self._registry.get(cc_type) is not None:
            return cc_type, {"task": task, "input_artifacts": files}
        return api_type, api_payload(api_type, task, files)

    # -- afterwards -----------------------------------------------------------

    async def run(self) -> None:
        while True:
            try:
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("claude code sweep failed - trying again shortly")
            await asyncio.sleep(_SWEEP_INTERVAL_S)

    async def sweep(self) -> int:
        """Hand unfinishable Claude Code jobs to the API. Returns how many."""
        moved = 0
        for job in await self._candidates():
            if job.status is JobStatus.SUCCEEDED:
                if (job.result or {}).get("fallback"):
                    moved += await self._hand_over(job, "Claude Code was out of usage")
            elif job.status is JobStatus.QUEUED and await self._stuck(job):
                api_type = _API_FOR[job.type]
                spec = self._registry.get(api_type)
                if spec is None or spec.requires:
                    continue        # the API version needs a worker too: wait
                withdrawn = await self._jobs.transition(
                    job.id, JobStatus.QUEUED, JobStatus.CANCELLED,
                    set_fields={"error": "no Claude Code worker online - "
                                         "handed to the API version"},
                )
                if withdrawn:
                    moved += await self._hand_over(job, "no Claude Code worker online")
        return moved

    async def _candidates(self) -> list[Job]:
        types = list(_API_FOR)
        marks = ", ".join("?" * len(types))
        rows = await self._db.query(
            f"SELECT j.id FROM jobs j WHERE j.type IN ({marks}) "
            f"AND j.status IN ('queued', 'succeeded') "
            f"AND NOT EXISTS (SELECT 1 FROM jobs c WHERE c.parent_job_id = j.id) "
            f"ORDER BY j.id LIMIT 50",
            types,
        )
        found = [await self._jobs.get(str(r["id"])) for r in rows]
        return [job for job in found if job is not None]

    async def _stuck(self, job: Job) -> bool:
        if self.online():
            self._offline_since = None
            return False
        now = utc_now()
        if self._offline_since is None:
            self._offline_since = now
        waited = now - max(job.created_ts, self._offline_since)
        return waited >= self._stuck_after

    async def _hand_over(self, job: Job, why: str) -> int:
        api_type = _API_FOR[job.type]
        if self._registry.get(api_type) is None:
            log.warning("no API version registered", extra={"job_type": api_type})
            return 0
        spec = self._registry.get(api_type)
        assert spec is not None
        replacement = Job(
            type=api_type,
            payload=api_payload(api_type, str(job.payload.get("task", "")),
                                list(job.payload.get("input_artifacts", []))),
            priority=job.priority,
            requires=spec.requires,
            session_id=job.session_id,
            parent_job_id=job.id,
            trace_id=job.trace_id,
        )
        await self._jobs.create(replacement)
        log.info("claude code job handed to the API", extra={
            "job_id": job.id, "replacement": replacement.id, "why": why,
        })
        return 1
