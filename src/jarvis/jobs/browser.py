# =============================================================================
# src/jarvis/jobs/browser.py - the browser.task job type
# =============================================================================
#
# Binds PROTEUS into the queue. Registered on BOTH sides for different
# reasons: the Core needs the metadata to validate payloads and construct
# offers, the worker needs the handler because that is where the browser
# lives.
#
# requires: ["browser"] - this is what forces the work off the Core and
# onto a machine that actually has a browser. A Core with no browser
# capability simply never picks it up, and the job waits in the queue
# until a capable worker connects. That waiting is normal operation, not
# a degraded mode.
#
# RESUMABLE, not idempotent. Browser state cannot be serialised - you
# cannot snapshot a live page and restore it - so the checkpoint is
# TASK-level: which step was reached and where. On resume PROTEUS
# re-navigates and continues, which is why task briefs should describe
# work that tolerates being restarted partway.
# =============================================================================

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jarvis.core.queue.registry import JobTypeRegistry, JobTypeSpec


class BrowserTaskIn(BaseModel):
    """Input: a self-contained task, written to be read cold."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=10)
    # A cap the caller can lower for simple tasks. The handler enforces
    # its own ceiling regardless.
    max_steps: int = Field(default=25, ge=1, le=60)


class BrowserTaskOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    report: str
    steps_taken: int
    completed: bool          # false when the step budget ran out


def register_browser_job_metadata(registry: JobTypeRegistry) -> None:
    """Core-side registration: metadata only, no handler.

    The Core never runs this - it declares a capability the Core does not
    have - but it must know the shape to validate payloads and build
    offers.
    """
    registry.register(JobTypeSpec(
        type="browser.task",
        input_model=BrowserTaskIn,
        output_model=BrowserTaskOut,
        execution="resumable",
        requires=["browser"],
        default_priority=5,
        timeout_s=900,          # a long web flow is minutes, not seconds
        handler=None,
    ))