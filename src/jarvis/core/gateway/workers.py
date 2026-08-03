# =============================================================================
# src/jarvis/core/gateway/workers.py - the worker WebSocket endpoint
# =============================================================================
#
# One coroutine per connected worker, handling its whole lifetime:
# authenticate, register, dispatch offers, relay progress, settle
# results, and clean up on disconnect.
#
# THE DISPATCH LOOP is per-connection rather than global: each worker's
# handler asks "is there queued work I can serve?" every second. That
# keeps offers simple (no central matchmaker deciding between workers)
# and means a worker with spare capacity claims work as fast as it can
# take it. Two workers racing for the same job is resolved by the
# optimistic check in transition(): one wins, the other looks again.
#
# ARTIFACTS ARRIVE IN CHUNKS and are buffered per job until the end
# frame, where size and checksum are verified. A mismatch is rejected -
# which is the entire point of shipping a checksum with a file.
#
# On disconnect nothing is repaired here: the worker's jobs keep their
# leases, and the reclaim loop requeues them when those leases expire.
# Remote death is handled by the same machinery as local death.
# =============================================================================

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from jarvis.common.envelope import Envelope, UnknownKind, make_envelope
from jarvis.common.events import Event, EventKind
from jarvis.common.ids import utc_now
from jarvis.common.jobs import JobStatus, Lease
from jarvis.common.log import get_logger
from jarvis.common.capabilities import Gate
from jarvis.core.approvals.service import ApprovalService
from jarvis.common.worker_protocol import (
    CORE_JOB_CANCEL,
    CORE_JOB_OFFER,
    CORE_WELCOME,
    ERROR_PROTOCOL,
    WORKER_ARTIFACT_BEGIN,
    WORKER_ARTIFACT_CHUNK,
    WORKER_ARTIFACT_END,
    WORKER_HEARTBEAT,
    WORKER_HELLO,
    WORKER_JOB_ACCEPT,
    WORKER_JOB_CHECKPOINT,
    WORKER_JOB_DECLINE,
    WORKER_JOB_PROGRESS,
    WORKER_JOB_RESULT,
    WORKER_JOB_STARTED,
    WORKER_NEEDS_APPROVAL,
    CoreJobCancel,
    CoreJobOffer,
    CoreWelcome,
    ProtocolError,
    WorkerArtifactBegin,
    WorkerArtifactChunk,
    WorkerArtifactEnd,
    WorkerHeartbeat,
    WorkerHello,
    WorkerJobAccept,
    WorkerJobCheckpoint,
    WorkerJobDecline,
    WorkerJobProgress,
    WorkerJobResult,
    WorkerJobStarted,
    WorkerNeedsApproval,
)
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.queue.dispatcher import fail_permanently, requeue_or_fail
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.core.queue.registry_workers import ConnectedWorker, WorkerRegistry

log = get_logger("core.gateway.workers")

SOURCE = "core.workers"
HEARTBEAT_INTERVAL_S = 15
LEASE_TTL_S = 90            # generous: three missed heartbeats before reclaim
_OFFER_POLL_S = 1.0


class _ArtifactBuffer:
    """One in-flight artifact upload, held until its end frame."""

    def __init__(self, begin: WorkerArtifactBegin) -> None:
        self.begin = begin
        self.chunks: dict[int, bytes] = {}

    def add(self, seq: int, data: bytes) -> None:
        self.chunks[seq] = data

    def assemble(self) -> bytes:
        # Sequence-ordered, so out-of-order frames reassemble correctly.
        return b"".join(self.chunks[i] for i in sorted(self.chunks))


class WorkerConnection:
    """Handles one worker's connection from hello to hangup."""

    def __init__(
        self,
        websocket: WebSocket,
        expected_token: str,
        registry: WorkerRegistry,
        jobs: JobsRepo,
        events: EventsRepo,
        artifacts: ArtifactsRepo,
        job_types: JobTypeRegistry,
        approvals: "ApprovalService | None" = None,
    ) -> None:
        self._ws = websocket
        self._expected_token = expected_token
        self._registry = registry
        self._jobs = jobs
        self._events = events
        self._artifacts = artifacts
        self._job_types = job_types
        self._approvals = approvals
        self._worker: ConnectedWorker | None = None
        self._offered: set[str] = set()      # awaiting accept or decline
        self._uploads: dict[str, _ArtifactBuffer] = {}

    async def run(self) -> None:
        """The whole connection lifetime."""
        await self._ws.accept()
        dispatcher: asyncio.Task[None] | None = None
        try:
            if not await self._handshake():
                return
            dispatcher = asyncio.create_task(self._offer_loop())
            await self._receive_loop()
        except WebSocketDisconnect:
            pass
        except Exception:
            log.error("worker connection failed", exc_info=True, extra={
                "worker_id": self._worker.worker_id if self._worker else None,
            })
        finally:
            if dispatcher is not None:
                dispatcher.cancel()
            await self._cleanup()

    # -- handshake ------------------------------------------------------------

    async def _handshake(self) -> bool:
        """First frame must be a valid, authenticated hello."""
        try:
            raw = await asyncio.wait_for(self._ws.receive_json(), timeout=10.0)
            envelope = Envelope.model_validate(raw)
            if envelope.kind != WORKER_HELLO:
                await self._send_error("first frame must be worker.hello")
                return False
            hello = envelope.parse_payload()
            assert isinstance(hello, WorkerHello)
        except (asyncio.TimeoutError, ValidationError, UnknownKind, Exception):
            await self._send_error("invalid handshake")
            return False

        # Timing-safe, same reasoning as the HTTP gateway: comparison
        # duration must not leak how much of a guess was correct.
        if not self._expected_token or not secrets.compare_digest(
            hello.token, self._expected_token
        ):
            log.warning("worker presented a bad token", extra={
                "worker_id": hello.worker_id,
            })
            await self._send_error("invalid token")
            return False

        self._worker = ConnectedWorker(
            worker_id=hello.worker_id,
            capabilities=set(hello.capabilities),
            max_concurrency=hello.max_concurrency,
            version=hello.version,
        )
        self._registry.register(self._worker)
        await self._send(CORE_WELCOME, CoreWelcome(
            heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
            lease_ttl_s=LEASE_TTL_S,
        ))
        await self._events.append(Event(
            kind=EventKind.WORKER_CONNECTED, source=SOURCE,
            trace_id=self._new_trace(),
            payload={
                "worker_id": hello.worker_id,
                "capabilities": sorted(hello.capabilities),
            },
        ))
        return True

    # -- offering work --------------------------------------------------------

    async def _offer_loop(self) -> None:
        """Look for work this worker can take, and offer it."""
        while True:
            try:
                await self._maybe_offer()
            except Exception:
                log.error("offer loop error", exc_info=True)
            await asyncio.sleep(_OFFER_POLL_S)

    async def _maybe_offer(self) -> None:
        worker = self._worker
        if worker is None:
            return
        # Outstanding offers count against capacity: a worker deciding
        # about three jobs must not be handed a fourth.
        in_flight = len(worker.running_job_ids) + len(self._offered)
        if in_flight >= worker.max_concurrency:
            return

        job = await self._jobs.next_queued(capabilities=worker.capabilities)
        if job is None or not job.requires:
            # requires:[] work belongs to the Core's built-in executor;
            # sending it over a network for no reason would be silly.
            return

        now = utc_now()
        won = await self._jobs.transition(
            job.id, JobStatus.QUEUED, JobStatus.LEASED,
            set_fields={
                "lease": Lease(
                    worker_id=worker.worker_id, leased_ts=now,
                    heartbeat_ts=now, ttl_s=LEASE_TTL_S,
                ),
                "attempts": job.attempts + 1,
            },
        )
        if not won:
            return   # another worker or the reclaim loop got there first

        self._offered.add(job.id)
        spec = self._job_types.get(job.type)
        await self._send(CORE_JOB_OFFER, CoreJobOffer(
            job_id=job.id,
            type=job.type,
            payload=job.payload,
            requires=job.requires,
            checkpoint=job.checkpoint,
            timeout_s=spec.timeout_s if spec else 300,
            trace_id=job.trace_id,
            approval_granted=job.checkpoint is not None and job.approval is None,
        ))
        await self._events.append(Event(
            kind=EventKind.JOB_LEASED, source=SOURCE, job_id=job.id,
            session_id=job.session_id, trace_id=job.trace_id,
            payload={"worker": worker.worker_id, "attempt": job.attempts + 1},
        ))

    async def cancel_job(self, job_id: str, reason: str) -> None:
        """Tell the worker to stop a job it is running."""
        await self._send(CORE_JOB_CANCEL, CoreJobCancel(job_id=job_id, reason=reason))

    # -- receiving ------------------------------------------------------------

    async def _receive_loop(self) -> None:
        while True:
            raw = await self._ws.receive_json()
            try:
                envelope = Envelope.model_validate(raw)
                payload = envelope.parse_payload()
            except UnknownKind as exc:
                await self._send_error(f"unsupported kind: {exc}")
                continue
            except ValidationError as exc:
                await self._send_error(f"invalid payload: {exc}")
                continue
            await self._handle(envelope.kind, payload)

    async def _handle(self, kind: str, payload: Any) -> None:
        worker = self._worker
        if worker is None:
            return

        if kind == WORKER_HEARTBEAT:
            assert isinstance(payload, WorkerHeartbeat)
            self._registry.heartbeat(worker.worker_id, payload.running_job_ids)
            await self._extend_leases(payload.running_job_ids)

        elif kind == WORKER_JOB_ACCEPT:
            assert isinstance(payload, WorkerJobAccept)
            self._offered.discard(payload.job_id)
            worker.running_job_ids.add(payload.job_id)

        elif kind == WORKER_JOB_DECLINE:
            assert isinstance(payload, WorkerJobDecline)
            self._offered.discard(payload.job_id)
            await self._return_declined(payload.job_id, payload.reason)

        elif kind == WORKER_JOB_STARTED:
            assert isinstance(payload, WorkerJobStarted)
            job = await self._jobs.get(payload.job_id)
            if job is not None:
                await self._jobs.transition(
                    job.id, JobStatus.LEASED, JobStatus.RUNNING
                )
                await self._events.append(Event(
                    kind=EventKind.JOB_STARTED, source=SOURCE, job_id=job.id,
                    session_id=job.session_id, trace_id=job.trace_id,
                    payload={"worker": worker.worker_id},
                ))

        elif kind == WORKER_JOB_PROGRESS:
            assert isinstance(payload, WorkerJobProgress)
            job = await self._jobs.get(payload.job_id)
            if job is not None:
                await self._events.append(Event(
                    kind=EventKind.JOB_PROGRESS, source=SOURCE, job_id=job.id,
                    session_id=job.session_id, trace_id=job.trace_id,
                    payload={"note": payload.note, "pct": payload.pct},
                ))

        elif kind == WORKER_JOB_CHECKPOINT:
            assert isinstance(payload, WorkerJobCheckpoint)
            await self._jobs.set_checkpoint(payload.job_id, payload.checkpoint)

        elif kind == WORKER_JOB_RESULT:
            assert isinstance(payload, WorkerJobResult)
            worker.running_job_ids.discard(payload.job_id)
            await self._settle(payload)

        elif kind == WORKER_NEEDS_APPROVAL:
            assert isinstance(payload, WorkerNeedsApproval)
            worker.running_job_ids.discard(payload.job_id)
            await self._raise_gate(payload)

        elif kind == WORKER_ARTIFACT_BEGIN:
            assert isinstance(payload, WorkerArtifactBegin)
            self._uploads[payload.job_id] = _ArtifactBuffer(payload)

        elif kind == WORKER_ARTIFACT_CHUNK:
            assert isinstance(payload, WorkerArtifactChunk)
            buffer = self._uploads.get(payload.job_id)
            if buffer is not None:
                buffer.add(payload.seq, base64.b64decode(payload.data_b64))

        elif kind == WORKER_ARTIFACT_END:
            assert isinstance(payload, WorkerArtifactEnd)
            await self._finish_artifact(payload.job_id)

    async def _extend_leases(self, job_ids: list[str]) -> None:
        """A heartbeat pushes each running job's lease forward."""
        worker = self._worker
        if worker is None:
            return
        now = utc_now()
        for job_id in job_ids:
            job = await self._jobs.get(job_id)
            if job is None or job.lease is None:
                continue
            if job.lease.worker_id != worker.worker_id:
                continue   # not this worker's job to keep alive
            await self._jobs.set_lease_heartbeat(job_id, now)

    async def _return_declined(self, job_id: str, reason: str) -> None:
        """A decline costs the job nothing: back to queued, attempt count
        rolled back, because declining is not failing."""
        job = await self._jobs.get(job_id)
        if job is None or job.status is not JobStatus.LEASED:
            return
        await self._jobs.transition(
            job.id, JobStatus.LEASED, JobStatus.QUEUED,
            set_fields={"lease": None, "attempts": max(0, job.attempts - 1)},
        )
        log.info("worker declined job", extra={"job_id": job_id, "reason": reason})

    async def _raise_gate(self, request: WorkerNeedsApproval) -> None:
        """A worker asked permission. Pause the job and ask the owner.

        This is the bridge the approval system was missing: gates fire
        on workers, approvals live here, and workers have no database.
        The service, the outbox, and the Telegram buttons all work
        exactly as built - they only needed a way for a distant process
        to reach them.
        """
        if self._approvals is None:
            log.error("worker raised a gate but approvals are unavailable",
                      extra={"job_id": request.job_id})
            return

        try:
            gate = Gate(request.gate)
        except ValueError:
            log.error("worker raised an unknown gate", extra={
                "job_id": request.job_id, "gate": request.gate,
            })
            return

        await self._approvals.request(
            job_id=request.job_id,
            gate=gate,
            actor=request.actor,
            tool=request.tool,
            summary=request.summary,
            detail=request.detail,
            risk_note=request.risk_note,
        )

    async def _settle(self, result: WorkerJobResult) -> None:
        """Record a terminal report from the worker."""
        job = await self._jobs.get(result.job_id)
        if job is None:
            return

        if result.status == "succeeded":
            moved = await self._jobs.transition(
                job.id, JobStatus.RUNNING, JobStatus.SUCCEEDED,
                set_fields={
                    "result": result.result or {},
                    "lease": None,
                    # Clear any error left by a previous attempt. A job
                    # requeued after a lease expiry carries the reason
                    # in `error`; succeeding later must wipe it, or the
                    # row violates the model's own invariant and becomes
                    # unreadable.
                    "error": None,
                },
            )
            if moved:
                await self._events.append(Event(
                    kind=EventKind.JOB_SUCCEEDED, source=SOURCE, job_id=job.id,
                    session_id=job.session_id, trace_id=job.trace_id,
                    payload={},
                ))
            return

        error = result.error or "worker reported failure"
        if result.permanent:
            await fail_permanently(
                self._jobs, self._events, job, error, expected=JobStatus.RUNNING
            )
        else:
            await requeue_or_fail(
                self._jobs, self._events, job, error, expected=JobStatus.RUNNING
            )

    async def _finish_artifact(self, job_id: str) -> None:
        """Assemble, verify, and store an uploaded file."""
        buffer = self._uploads.pop(job_id, None)
        if buffer is None:
            return
        content = buffer.assemble()

        # Verification is why the checksum travels with the file: a
        # truncated or corrupted upload must not become a stored
        # artifact that something later trusts.
        digest = hashlib.sha256(content).hexdigest()
        if digest != buffer.begin.sha256 or len(content) != buffer.begin.size:
            log.error("artifact upload failed verification", extra={
                "job_id": job_id, "name": buffer.begin.name,
                "expected_size": buffer.begin.size, "got_size": len(content),
            })
            await self._send_error(f"artifact {buffer.begin.name} failed verification")
            return

        artifact = await self._artifacts.write(
            name=buffer.begin.name, mime=buffer.begin.mime,
            content=content, created_by=job_id,
        )
        await self._jobs.add_artifact(job_id, artifact.id)

    # -- plumbing -------------------------------------------------------------

    async def _send(self, kind: str, payload: Any) -> None:
        envelope = make_envelope(kind, payload)
        await self._ws.send_json(envelope.model_dump(mode="json"))

    async def _send_error(self, message: str) -> None:
        try:
            await self._send(ERROR_PROTOCOL, ProtocolError(message=message))
        except Exception:
            pass   # the socket is probably already gone

    def _new_trace(self) -> str:
        from jarvis.common.ids import new_ulid
        return new_ulid()

    async def _cleanup(self) -> None:
        """On disconnect, forget the worker - and repair nothing.

        Its jobs keep their leases, which expire in LEASE_TTL_S and are
        requeued by the reclaim loop. That is the same path a crashed
        Core-local executor takes, so a vanished laptop needs no special
        handling anywhere downstream.
        """
        if self._worker is None:
            return
        worker_id = self._worker.worker_id
        self._registry.unregister(worker_id)
        await self._events.append(Event(
            kind=EventKind.WORKER_DISCONNECTED, source=SOURCE,
            trace_id=self._new_trace(),
            payload={
                "worker_id": worker_id,
                "jobs_in_flight": sorted(self._worker.running_job_ids),
            },
        ))