# =============================================================================
# src/jarvis/jobs/worker_types.py - job types the Core knows but cannot run
# =============================================================================
#
# The Core registers these for their METADATA - input schema, timeout,
# capability requirements - so it can validate payloads and construct
# offers. It never executes them: they declare capabilities the Core does
# not have, so the built-in executor passes them by and only a matching
# worker is offered the work.
#
# Handler is None here on purpose. The real implementation lives in the
# worker's own registry, on a machine that can actually do the thing.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from jarvis.core.queue.registry import JobTypeRegistry, JobTypeSpec


class EchoIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    delay_s: int = 0


class EchoOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    ran_on: str


def register_worker_job_types(registry: JobTypeRegistry) -> None:
    """Register worker-only job types for their metadata."""
    registry.register(JobTypeSpec(
        type="worker.echo",
        input_model=EchoIn,
        output_model=EchoOut,
        execution="idempotent",
        requires=["macos"],
        timeout_s=120,
        handler=None,           # runs on the worker, not here
    ))