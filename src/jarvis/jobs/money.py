# =============================================================================
# src/jarvis/jobs/money.py - PLUTUS on a schedule
# =============================================================================
#
#   money.daily    each morning: spot new subscriptions, warn about
#                  renewals in the next two days, note payments that
#                  never came. Silent when there is nothing to say.
#   money.review   Sunday: the week's spending against the week before.
#
# Both assemble their message from the ledger - no model call, nothing
# that is not in the books.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.notifications import Notification
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.money.service import MoneyService
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec

DAILY_JOB_TYPE = "money.daily"
DAILY_SCHEDULE = "money daily check"
REVIEW_JOB_TYPE = "money.review"
REVIEW_SCHEDULE = "money weekly review"


class MoneyJobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="scheduled")


class MoneyJobOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sent: bool
    chars: int = 0


def register_money_jobs(registry: JobTypeRegistry, money: MoneyService,
                        sessions: SessionsRepo, notifications: NotificationsRepo) -> None:

    async def send(text: str, ctx: JobContext, priority: int) -> MoneyJobOut:
        if not text:
            return MoneyJobOut(sent=False)
        session = await sessions.get_or_create_default("telegram")
        await notifications.create(Notification(
            client_kind="telegram", text=text, session_id=session.id,
            job_id=ctx.job_id,           # one message per job, even if retried
            priority=priority, trace_id=ctx.trace_id,
        ))
        return MoneyJobOut(sent=True, chars=len(text))

    async def daily(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, MoneyJobIn)
        if money.watchdog is None:
            return MoneyJobOut(sent=False)
        return await send(await money.watchdog.daily(), ctx, priority=4)

    async def review(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, MoneyJobIn)
        return await send(await money.weekly_review(), ctx, priority=4)

    for job_type, handler in ((DAILY_JOB_TYPE, daily), (REVIEW_JOB_TYPE, review)):
        registry.register(JobTypeSpec(
            type=job_type,
            input_model=MoneyJobIn,
            output_model=MoneyJobOut,
            execution="idempotent",
            requires=[],
            default_priority=6,
            timeout_s=120,
            handler=handler,
        ))
