# =============================================================================
# src/jarvis/jobs/life.py - the evening check-in
# =============================================================================
#
# At 22:00 (just before quiet hours) JARVIS sends the day's checklist,
# what has not been logged, and tomorrow's list, and asks what got done.
# The owner answers in the same Telegram thread; JARVIS logs the answer
# with the life_report tool.
#
# No model call: the message is assembled from the database, so it costs
# nothing and cannot say anything that is not true.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.notifications import Notification
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.life.service import LifeService
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec

CHECKIN_JOB_TYPE = "life.checkin"
CHECKIN_SCHEDULE = "evening check-in"


class CheckinIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="scheduled")


class CheckinOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chars: int


def register_life_jobs(registry: JobTypeRegistry, life: LifeService,
                       sessions: SessionsRepo, notifications: NotificationsRepo) -> None:

    async def handle(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, CheckinIn)
        text = await life.checkin_text()
        session = await sessions.get_or_create_default("telegram")
        await notifications.create(Notification(
            client_kind="telegram", text=text, session_id=session.id,
            job_id=ctx.job_id,       # one check-in per job, even if retried
            priority=3, trace_id=ctx.trace_id,
        ))
        return CheckinOut(chars=len(text))

    registry.register(JobTypeSpec(
        type=CHECKIN_JOB_TYPE,
        input_model=CheckinIn,
        output_model=CheckinOut,
        execution="idempotent",
        requires=[],
        default_priority=4,
        timeout_s=60,
        handler=handle,
    ))
