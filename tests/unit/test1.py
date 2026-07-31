import asyncio

from pydantic import BaseModel

from jarvis.common.log import setup_logging
from jarvis.common.schedules import Schedule, ScheduleKind
from jarvis.common.settings import CoreSettings
from jarvis.common.ids import utc_now
from jarvis.core.app import CoreApp
from jarvis.core.queue.registry import JobContext, JobTypeSpec


class TickIn(BaseModel):
    label: str

class TickOut(BaseModel):
    label: str


async def tick_handler(payload: TickIn, ctx: JobContext) -> TickOut:
    print(f"    >>> scheduled job ran: {payload.label}")
    return TickOut(label=payload.label)


async def main() -> None:
    setup_logging("INFO")
    app = CoreApp(CoreSettings())
    app.registry.register(JobTypeSpec(
        type="test.tick", input_model=TickIn, output_model=TickOut,
        execution="idempotent", timeout_s=30, handler=tick_handler,
    ))
    await app.boot()
    assert app.schedules is not None

    await app.schedules.ensure(Schedule(
        name="demo every 30s",
        kind=ScheduleKind.INTERVAL,
        interval_s=30,
        job_type="test.tick",
        job_payload={"label": "heartbeat"},
        next_fire_ts=utc_now(),      # due immediately
    ))
    print(">>> schedule created - watch it fire, then kill and restart me")
    await app.run()


asyncio.run(main())