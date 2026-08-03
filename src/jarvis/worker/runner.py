# =============================================================================
# src/jarvis/worker/runner.py - executing one job on a worker
# =============================================================================
#
# The worker's half of job execution. Mirrors the Core's built-in
# executor deliberately: same JobContext, same handler signature, same
# error taxonomy - so a job type can move between Core and worker by
# changing its `requires` list and nothing else.
#
# THE DIFFERENCE IS WHERE STATE LIVES. The Core's executor writes to the
# database directly; this one sends messages. Progress, checkpoints, and
# artifacts all travel to the Core BEFORE the job reports success, which
# is what makes a worker disposable: if this machine vanishes, nothing
# that mattered was only here.
#
# Artifacts upload in chunks with a checksum computed before the first
# byte leaves, so the Core can verify what arrived rather than trusting
# it.
# =============================================================================

from __future__ import annotations

import asyncio
import base64
import hashlib
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from jarvis.common.log import get_logger
from jarvis.llm.anthropic import ProviderError
from jarvis.common.worker_protocol import (
    WORKER_ARTIFACT_BEGIN,
    WORKER_ARTIFACT_CHUNK,
    WORKER_ARTIFACT_END,
    WORKER_JOB_CHECKPOINT,
    WORKER_JOB_PROGRESS,
    WORKER_JOB_RESULT,
    WORKER_JOB_STARTED,
    CoreJobOffer,
    WorkerArtifactBegin,
    WorkerArtifactChunk,
    WorkerArtifactEnd,
    WorkerJobCheckpoint,
    WorkerJobProgress,
    WorkerJobResult,
    WorkerJobStarted,
)
from jarvis.common.worker_protocol import (
    WORKER_NEEDS_APPROVAL,
    WorkerNeedsApproval,
)
from jarvis.core.queue.registry import (
    JobContext,
    JobTypeRegistry,
    PausedForApproval,
    PermanentJobError,
)

log = get_logger("worker.runner")

# Chunk size for artifact upload. Small enough that a single frame stays
# comfortable for any WebSocket implementation, large enough that a
# megabyte does not become a thousand round trips.
_CHUNK_BYTES = 64 * 1024

# (kind, payload) -> sent to the Core.
Sender = Callable[[str, BaseModel], Awaitable[None]]


class JobRunner:
    """Runs offered jobs and reports everything back to the Core."""

    def __init__(self, registry: JobTypeRegistry, send: Sender) -> None:
        self._registry = registry
        self._send = send
        self._running: dict[str, asyncio.Task[None]] = {}

    @property
    def running_job_ids(self) -> list[str]:
        """What the heartbeat reports as still alive."""
        return list(self._running)

    def can_run(self, job_type: str) -> bool:
        spec = self._registry.get(job_type)
        return spec is not None and spec.handler is not None

    def start(self, offer: CoreJobOffer) -> None:
        """Begin a job in its own task, so several can run at once and
        the receive loop never blocks."""
        task = asyncio.create_task(
            self._execute(offer), name=f"job-{offer.job_id[-6:]}"
        )
        self._running[offer.job_id] = task
        task.add_done_callback(lambda _: self._running.pop(offer.job_id, None))

    def cancel(self, job_id: str) -> bool:
        """Stop a running job on the Core's instruction. This is what
        makes cancellation immediate rather than waiting out a lease."""
        task = self._running.get(job_id)
        if task is None:
            return False
        task.cancel()
        return True

    async def _execute(self, offer: CoreJobOffer) -> None:
        """One job, from acceptance to terminal report. Never raises."""
        spec = self._registry.get(offer.type)
        if spec is None or spec.handler is None:
            await self._report(
                offer.job_id, "failed",
                error=f"no handler for {offer.type} on this worker",
                permanent=True,
            )
            return

        await self._send(WORKER_JOB_STARTED, WorkerJobStarted(job_id=offer.job_id))

        async def save_checkpoint(state: dict[str, Any]) -> None:
            await self._send(WORKER_JOB_CHECKPOINT, WorkerJobCheckpoint(
                job_id=offer.job_id, checkpoint=state,
            ))

        async def progress(note: str) -> None:
            await self._send(WORKER_JOB_PROGRESS, WorkerJobProgress(
                job_id=offer.job_id, note=note,
            ))

        async def write_artifact(name: str, mime: str, content: bytes) -> str:
            await self._upload(offer.job_id, name, mime, content)
            # The Core mints the artifact id; the handler gets the name
            # back as a handle. Handlers that need the real id will get
            # it from the job's artifact list after completion.
            return name

        ctx = JobContext(
            job_id=offer.job_id,
            trace_id=offer.trace_id,
            checkpoint=offer.checkpoint,
            save_checkpoint=save_checkpoint,
            progress=progress,
            write_artifact=write_artifact,
            approval_granted=offer.approval_granted,
        )

        try:
            payload = spec.input_model.model_validate(offer.payload)
            async with asyncio.timeout(offer.timeout_s):
                output = await spec.handler(payload, ctx)
            result = spec.output_model.model_validate(
                output, from_attributes=True
            ).model_dump(mode="json")
        except PausedForApproval:
            # Not a failure. The handler wants permission; we relay the
            # request and stop. The Core raises the gate, pauses the job,
            # and re-offers it after the owner answers.
            request = ctx.pending_approval or {}
            await self._send(WORKER_NEEDS_APPROVAL, WorkerNeedsApproval(
                job_id=offer.job_id,
                gate=str(request.get("gate", "unknown")),
                actor=str(request.get("actor", "subagent.engineer")),
                tool=str(request.get("tool", "unknown")),
                summary=str(request.get("summary", "An action needs approval")),
                detail=str(request.get("detail", "")),
                risk_note=str(request.get("risk_note", "")),
            ))
            log.info("job paused for approval", extra={
                "job_id": offer.job_id, "tool": request.get("tool"),
            })
            return
        except asyncio.CancelledError:
            # The Core asked us to stop; it already knows why.
            await self._report(offer.job_id, "failed",
                               error="cancelled", permanent=True)
            return
        except (TimeoutError, asyncio.TimeoutError):
            await self._report(offer.job_id, "failed",
                               error=f"timed out after {offer.timeout_s}s")
            return
        except PermanentJobError as exc:
            await self._report(offer.job_id, "failed",
                               error=str(exc), permanent=True)
            return
        except ProviderError as exc:
            # A malformed request or rejected key fails identically every
            # time; only transient provider trouble deserves a retry.
            await self._report(offer.job_id, "failed",
                               error=str(exc), permanent=exc.permanent)
            return
        except ValidationError as exc:
            # A payload that fails its schema will fail it again.
            await self._report(offer.job_id, "failed",
                               error=f"validation failed: {exc}", permanent=True)
            return
        except Exception as exc:
            log.error("job handler crashed", exc_info=True, extra={
                "job_id": offer.job_id, "type": offer.type,
            })
            await self._report(offer.job_id, "failed",
                               error=f"{type(exc).__name__}: {exc}")
            return

        await self._report(offer.job_id, "succeeded", result=result)

    async def _upload(
        self, job_id: str, name: str, mime: str, content: bytes
    ) -> None:
        """Ship a file to the Core in chunks, checksum first.

        The digest is computed here, before anything is sent, so the Core
        can verify what it assembled rather than trusting the wire.
        """
        await self._send(WORKER_ARTIFACT_BEGIN, WorkerArtifactBegin(
            job_id=job_id, name=name, mime=mime,
            size=len(content), sha256=hashlib.sha256(content).hexdigest(),
        ))
        for seq, start in enumerate(range(0, len(content), _CHUNK_BYTES)):
            chunk = content[start:start + _CHUNK_BYTES]
            await self._send(WORKER_ARTIFACT_CHUNK, WorkerArtifactChunk(
                job_id=job_id, seq=seq,
                data_b64=base64.b64encode(chunk).decode("ascii"),
            ))
        await self._send(WORKER_ARTIFACT_END, WorkerArtifactEnd(job_id=job_id))

    async def _report(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        permanent: bool = False,
    ) -> None:
        """The terminal frame. Sent last, after every artifact, so the
        Core never records success for a job whose files are missing."""
        try:
            await self._send(WORKER_JOB_RESULT, WorkerJobResult(
                job_id=job_id, status=status, result=result,
                error=error, permanent=permanent,
            ))
        except Exception:
            # The link died as we finished. The Core's lease will expire
            # and requeue the job - which is exactly the behaviour we
            # want, and why job types must be idempotent or resumable.
            log.warning("could not report result - link down", extra={
                "job_id": job_id, "status": status,
            })