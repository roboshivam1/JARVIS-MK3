# =============================================================================
# src/jarvis/jobs/health.py - HYGIEIA on a schedule
# =============================================================================
#
#   health.checkin   17:00 - what is left for the day, and what he can
#                    actually eat tonight to cover it.
#   health.review    Sunday - the week against his targets and limits,
#                    with sleep and food spending, and one thing to change.
#
# Both messages are assembled from the records. The check-in may make one
# model call, once, to learn the values of mess menu items it has not
# seen yet; after that it costs nothing.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.notifications import Notification
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.health.service import HealthService
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec

CHECKIN_JOB_TYPE = "health.checkin"
CHECKIN_SCHEDULE = "health check-in"
REVIEW_JOB_TYPE = "health.review"
REVIEW_SCHEDULE = "health weekly review"


class HealthJobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="scheduled")


class HealthJobOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chars: int


def register_health_jobs(registry: JobTypeRegistry, health: HealthService,
                         sessions: SessionsRepo, notifications: NotificationsRepo) -> None:

    async def send(text: str, ctx: JobContext) -> HealthJobOut:
        session = await sessions.get_or_create_default("telegram")
        await notifications.create(Notification(
            client_kind="telegram", text=text, session_id=session.id,
            job_id=ctx.job_id, priority=4, trace_id=ctx.trace_id,
        ))
        return HealthJobOut(chars=len(text))

    async def checkin(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, HealthJobIn)
        return await send(await health.checkin_text(trace_id=ctx.trace_id), ctx)

    async def review(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, HealthJobIn)
        return await send(await health.weekly_review(), ctx)

    for job_type, handler, timeout in ((CHECKIN_JOB_TYPE, checkin, 300),
                                       (REVIEW_JOB_TYPE, review, 120)):
        registry.register(JobTypeSpec(
            type=job_type, input_model=HealthJobIn, output_model=HealthJobOut,
            execution="idempotent", requires=[], default_priority=5,
            timeout_s=timeout, handler=handler,
        ))
