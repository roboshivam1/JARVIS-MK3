# =============================================================================
# src/jarvis/core/initiative/engine.py - things that happen because time
# passed
# =============================================================================
#
# The first trigger in the system that is neither the owner speaking nor
# work finishing. Every few seconds: ask the database what is due,
# enqueue the corresponding job, compute the next occurrence.
#
# CATCH-UP POLICY - at most one. If the Core was off when a schedule was
# due, it fires ONCE on return and the next occurrence is computed from
# now. Firing nothing would mean nightly maintenance silently never runs
# after any downtime; firing every missed occurrence would mean an
# interval schedule dumping dozens of jobs at once after a weekend off.
# Late beats never; fifty-times-late beats neither.
#
# Cron expressions are evaluated in the OWNER'S timezone, then stored as
# UTC. "Nightly at 03:30" means half three in Jaipur, whichever machine
# in whichever datacenter is running the Core.
#
# Each firing mints its own trace id: one schedule firing is one root
# cause, so everything it sets in motion can be followed as a unit.
# =============================================================================

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from croniter import croniter

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.jobs import Job
from jarvis.common.log import get_logger
from jarvis.common.schedules import Schedule, ScheduleKind
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.schedules import SchedulesRepo

log = get_logger("core.initiative.engine")

SOURCE = "core.initiative"
_TICK_INTERVAL_S = 20.0


def next_cron_time(
    cron_expr: str, tz: ZoneInfo, after: datetime | None = None
) -> datetime:
    """Next match of a cron expression, in UTC.

    Evaluated in the owner's LOCAL time, then converted: "03:30 nightly"
    means half three in Jaipur regardless of which datacenter is running
    the Core.
    """
    local = (after or utc_now()).astimezone(tz)
    return croniter(cron_expr, local).get_next(datetime).astimezone(timezone.utc)


def next_occurrence(
    schedule: Schedule,
    tz: ZoneInfo,
    after: datetime | None = None,
) -> datetime:
    """When this schedule should next fire, in UTC."""
    moment = after or utc_now()

    if schedule.kind is ScheduleKind.CRON and schedule.cron_expr:
        return next_cron_time(schedule.cron_expr, tz, moment)

    if schedule.kind is ScheduleKind.INTERVAL and schedule.interval_s:
        return moment + timedelta(seconds=schedule.interval_s)

    # A `once` schedule has no next occurrence; it is disabled after
    # firing. The far-future value keeps it out of due() in the window
    # between firing and disabling.
    return moment + timedelta(days=36500)


class InitiativeEngine:
    """Fires schedules. Grows watchers and follow-ups in a later phase."""

    def __init__(
        self,
        schedules: SchedulesRepo,
        jobs: JobsRepo,
        events: EventsRepo,
        tz: ZoneInfo,
    ) -> None:
        self._schedules = schedules
        self._jobs = jobs
        self._events = events
        self._tz = tz

    async def run(self) -> None:
        """The resident loop: tick, sleep, repeat."""
        while True:
            try:
                await self.tick()
            except Exception:
                # The clock must outlive its own bad days.
                log.error("initiative tick failed", exc_info=True)
            await asyncio.sleep(_TICK_INTERVAL_S)

    async def tick(self) -> int:
        """Fire everything currently due. Returns how many fired."""
        fired = 0
        for schedule in await self._schedules.due():
            try:
                await self._fire(schedule)
                fired += 1
            except Exception:
                # One broken schedule must not stop the others. It stays
                # due and will be retried next tick, loudly each time.
                log.error("schedule failed to fire", exc_info=True, extra={
                    "schedule": schedule.name,
                })
        return fired

    async def _fire(self, schedule: Schedule) -> None:
        trace_id = new_ulid()   # this firing is its own root cause

        job = Job(
            type=schedule.job_type,
            payload=schedule.job_payload,
            priority=schedule.priority,
            trace_id=trace_id,
        )
        await self._jobs.create(job)

        await self._events.append(Event(
            kind=EventKind.INITIATIVE_SCHEDULE_FIRED,
            source=SOURCE,
            job_id=job.id,
            trace_id=trace_id,
            payload={
                "schedule": schedule.name,
                "job_type": schedule.job_type,
                "fire_count": schedule.fire_count + 1,
            },
        ))

        # Advance the clock BEFORE anything else can retry this schedule.
        # Next occurrence is computed from now, not from the missed time,
        # which is what caps catch-up at a single firing.
        following = next_occurrence(schedule, self._tz)
        await self._schedules.mark_fired(schedule.id, following)
        if schedule.kind is ScheduleKind.ONCE:
            await self._schedules.disable(schedule.id)

        log.info("schedule fired", extra={
            "schedule": schedule.name, "job_id": job.id,
            "next_fire_ts": following.isoformat(),
        })