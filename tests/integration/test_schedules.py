# =============================================================================
# tests/integration/test_schedules.py - the clock
# =============================================================================
#
# The two promises that matter: schedules survive restarts (they are rows,
# not timers), and a Core that was switched off does not come back and
# fire a week of missed occurrences at once.
# =============================================================================

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

from jarvis.common.ids import utc_now
from jarvis.common.schedules import Schedule, ScheduleKind
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.schedules import SchedulesRepo
from jarvis.core.initiative.engine import InitiativeEngine, next_cron_time

JAIPUR = ZoneInfo("Asia/Kolkata")


def _interval_schedule(name: str, seconds: int, due_ago_s: int = 0) -> Schedule:
    return Schedule(
        name=name,
        kind=ScheduleKind.INTERVAL,
        interval_s=seconds,
        job_type="test.tick",
        next_fire_ts=utc_now() - timedelta(seconds=due_ago_s),
    )


class TestCronTiming:
    def test_cron_uses_owner_timezone(self) -> None:
        # 03:30 in Jaipur is 22:00 UTC the previous day. If cron were
        # evaluated in UTC, nightly maintenance would run at 09:00 local.
        when = next_cron_time("30 3 * * *", JAIPUR)
        assert when.hour == 22 and when.minute == 0

    def test_cron_is_always_in_the_future(self) -> None:
        assert next_cron_time("*/5 * * * *", JAIPUR) > utc_now()


class TestScheduleStore:
    async def test_ensure_is_idempotent(self, db: Database) -> None:
        # Boot seeds default schedules every time; the second seeding
        # must not duplicate or reset anything.
        repo = SchedulesRepo(db)
        assert await repo.ensure(_interval_schedule("nightly", 3600))
        assert not await repo.ensure(_interval_schedule("nightly", 60))
        stored = await repo.all()
        assert len(stored) == 1
        assert stored[0].interval_s == 3600   # the owner's row survived

    async def test_due_ignores_future_and_disabled(self, db: Database) -> None:
        repo = SchedulesRepo(db)
        await repo.ensure(_interval_schedule("ready", 60, due_ago_s=10))
        future = _interval_schedule("later", 60)
        future = future.model_copy(
            update={"next_fire_ts": utc_now() + timedelta(hours=1)}
        )
        await repo.ensure(future)

        due = await repo.due()
        assert [s.name for s in due] == ["ready"]

        disabled = await repo.all()
        await repo.disable(next(s.id for s in disabled if s.name == "ready"))
        assert await repo.due() == []


class TestFiring:
    async def test_firing_enqueues_a_job(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        schedules = SchedulesRepo(db)
        await schedules.ensure(_interval_schedule("tick", 60, due_ago_s=5))
        engine = InitiativeEngine(schedules, jobs, events, JAIPUR)

        assert await engine.tick() == 1
        queued = await jobs.live_jobs()
        assert len(queued) == 1 and queued[0].type == "test.tick"

        # The firing and the job share one trace: one root cause.
        trail = await events.by_trace(queued[0].trace_id)
        kinds = {e.kind for e in trail}
        assert "initiative.schedule_fired" in kinds
        assert "job.enqueued" in kinds

    async def test_catch_up_fires_once_not_many(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        # Five hours off, a one-minute interval: 300 missed occurrences.
        # Exactly one job may result.
        schedules = SchedulesRepo(db)
        await schedules.ensure(_interval_schedule("frequent", 60, due_ago_s=18000))
        engine = InitiativeEngine(schedules, jobs, events, JAIPUR)

        assert await engine.tick() == 1
        assert await engine.tick() == 0        # nothing left due
        assert len(await jobs.live_jobs()) == 1

        # And the next fire is measured from NOW, not from the missed time.
        stored = (await schedules.all())[0]
        assert stored.next_fire_ts > utc_now()
        assert stored.fire_count == 1

    async def test_once_schedule_disables_itself(
        self, db: Database, jobs: JobsRepo, events: EventsRepo
    ) -> None:
        schedules = SchedulesRepo(db)
        await schedules.ensure(Schedule(
            name="one shot",
            kind=ScheduleKind.ONCE,
            job_type="test.tick",
            next_fire_ts=utc_now() - timedelta(seconds=1),
        ))
        engine = InitiativeEngine(schedules, jobs, events, JAIPUR)

        assert await engine.tick() == 1
        assert await engine.tick() == 0
        stored = (await schedules.all())[0]
        assert not stored.enabled