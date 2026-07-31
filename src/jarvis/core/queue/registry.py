# =============================================================================
# src/jarvis/core/queue/registry.py - the catalogue of job types
# =============================================================================
#
# Nothing runs unless its TYPE is registered here first, with:
#   - typed input and output models (payload/result validated both ways)
#   - required capabilities (empty list = runnable on the Core itself)
#   - a timeout
#   - its re-run behaviour: "idempotent" (safe to rerun blind) or
#     "resumable" (continues from checkpoints). Retry is built into this
#     system - crashes requeue work automatically - so every type must
#     answer "what if this runs 1.5 times?" BEFORE its first run.
#   - for core-runnable types, the handler function itself
#
# The registry is also the orchestrator's menu: its job-enqueue tool will
# expose exactly this catalogue, nothing else.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel

from jarvis.common.log import get_logger

log = get_logger("core.queue.registry")


@dataclass
class JobContext:
    """What a running handler may know and do about its own job."""

    job_id: str
    trace_id: str
    checkpoint: dict[str, Any] | None                      # resume state, if any
    save_checkpoint: Callable[[dict[str, Any]], Awaitable[None]]
    progress: Callable[[str], Awaitable[None]]             # human-readable note
    # (name, mime, content) -> artifact id. Files a handler produces go
    # through here: stored, catalogued, and attached to this job.
    write_artifact: Callable[[str, str, bytes], Awaitable[str]]


# A core-executable handler: validated payload in, output model back.
CoreHandler = Callable[[BaseModel, JobContext], Awaitable[BaseModel]]


@dataclass(frozen=True)
class JobTypeSpec:
    """Everything the queue needs to know about one kind of work."""

    type: str                                   # dotted name, e.g. research.brief
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    execution: Literal["idempotent", "resumable"]
    requires: list[str] = field(default_factory=list)
    default_priority: int = 5
    timeout_s: int = 300
    # Present for requires:[] types the Core can run itself; worker-only
    # types (browser, sandbox) ship their handlers in the worker instead.
    handler: CoreHandler | None = None


class JobTypeRegistry:
    """The single catalogue. Built at boot; frozen in spirit thereafter."""

    def __init__(self) -> None:
        self._specs: dict[str, JobTypeSpec] = {}

    def register(self, spec: JobTypeSpec) -> None:
        if spec.type in self._specs:
            raise RuntimeError(f"job type {spec.type!r} registered twice")
        if not spec.requires and spec.handler is None:
            raise RuntimeError(
                f"job type {spec.type!r} requires no capabilities but has "
                f"no handler - nobody could ever run it"
            )
        self._specs[spec.type] = spec
        log.debug("job type registered", extra={
            "type": spec.type, "requires": spec.requires,
            "execution": spec.execution,
        })

    def get(self, type_name: str) -> JobTypeSpec | None:
        return self._specs.get(type_name)

    def catalogue(self) -> list[JobTypeSpec]:
        return sorted(self._specs.values(), key=lambda s: s.type)