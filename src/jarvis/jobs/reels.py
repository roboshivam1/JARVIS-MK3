# =============================================================================
# src/jarvis/jobs/reels.py - the reel.make job type, Core side
# =============================================================================
#
# Registered on the Core for its METADATA only: payload schema, timeout,
# and the capabilities a machine needs. The Core has no GPU and never
# runs it; the handler lives in jarvis.worker.reels, on the Mac.
#
# One job carries a reel from brief to mp4 across several attempts:
#
#   attempt 1   draft (script, voice, stills)  -> pause for review
#   attempt 2   revise on notes                -> pause for review   (repeatable)
#   attempt N   render and mux                 -> succeed with the mp4
#
# Each pause is an approval gate, and every resume is a new lease, which
# counts as an attempt. MAX_ATTEMPTS leaves room for a few rounds of
# notes plus retries through a Claude usage limit, whose exponential
# backoff has to outlast a reset window of several hours.
# =============================================================================

from __future__ import annotations

from jarvis.common.reels import REEL_JOB_TYPE, REEL_REQUIRES, ReelJobIn, ReelJobOut
from jarvis.core.queue.registry import JobTypeRegistry, JobTypeSpec

MAX_ATTEMPTS = 12

# The longest single attempt: a full render of a 40 s reel plus the mux,
# with headroom. Drafting is far shorter; the worker enforces its own
# per-phase limits inside this one.
TIMEOUT_S = 3600


def reel_job_spec() -> JobTypeSpec:
    """The spec, metadata only. The worker builds its own with a handler."""
    return JobTypeSpec(
        type=REEL_JOB_TYPE,
        input_model=ReelJobIn,
        output_model=ReelJobOut,
        execution="resumable",
        requires=list(REEL_REQUIRES),
        default_priority=6,        # background work, never ahead of the owner
        timeout_s=TIMEOUT_S,
        handler=None,
    )


def register_reel_job_metadata(registry: JobTypeRegistry) -> None:
    registry.register(reel_job_spec())
