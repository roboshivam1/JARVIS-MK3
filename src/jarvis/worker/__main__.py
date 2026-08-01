# =============================================================================
# src/jarvis/worker/__main__.py - entry point for `python -m jarvis.worker`
# =============================================================================
#
# Wiring only: settings, logging, job type registration, then run.
#
# Job types registered here are the ones THIS MACHINE can execute. The
# list stays deliberately dull for now - a browser subagent and a code
# sandbox arrive in later batches, once the plumbing beneath them has
# been watched working.
# =============================================================================

from __future__ import annotations

import asyncio
import logging
import sys

from pydantic import BaseModel, ConfigDict

from jarvis.common.log import get_logger, setup_logging
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec
from jarvis.worker.app import WorkerApp
from jarvis.worker.settings import WorkerSettings

# Importing the protocol registers its envelope kinds on this side too.
import jarvis.common.worker_protocol  # noqa: F401


class EchoIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    delay_s: int = 0


class EchoOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    ran_on: str


def build_registry(settings: WorkerSettings) -> JobTypeRegistry:
    """The job types this worker can run."""
    registry = JobTypeRegistry()

    async def echo(payload: BaseModel, ctx: JobContext) -> BaseModel:
        """A deliberately boring job: proves the whole machine turns -
        offer, accept, progress, artifact, result - with nothing at
        stake."""
        assert isinstance(payload, EchoIn)
        for second in range(payload.delay_s):
            await asyncio.sleep(1)
            await ctx.progress(f"working {second + 1}/{payload.delay_s}")
        await ctx.write_artifact(
            "echo.txt", "text/plain",
            f"{payload.message}\n\nproduced on {settings.worker_id}\n".encode(),
        )
        return EchoOut(
            summary=f"Echoed from {settings.worker_id}: {payload.message}",
            ran_on=settings.worker_id,
        )

    registry.register(JobTypeSpec(
        type="worker.echo",
        input_model=EchoIn,
        output_model=EchoOut,
        execution="idempotent",
        requires=["macos"],     # forces it off the Core and onto a worker
        timeout_s=120,
        handler=echo,
    ))
    return registry


def main() -> None:
    setup_logging("INFO")
    log = get_logger("worker.main")

    try:
        settings = WorkerSettings()
    except Exception:
        log.critical("configuration invalid - refusing to start", exc_info=True)
        sys.exit(1)

    logging.getLogger().setLevel(settings.log_level)
    asyncio.run(WorkerApp(settings, build_registry(settings)).run())


if __name__ == "__main__":
    main()