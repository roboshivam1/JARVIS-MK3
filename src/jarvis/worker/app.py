# =============================================================================
# src/jarvis/worker/app.py - the worker daemon
# =============================================================================
#
# Dials the Core, announces what this machine can do, takes offered work,
# and keeps saying "still here" until something interrupts it. Then dials
# again.
#
# RECONNECT IS THE NORMAL CASE, not error handling. A laptop lid closes,
# wifi drops, the Core restarts for a deploy - all the same event: the
# socket died, wait, dial again. Backoff grows to a minute so an hour of
# Core downtime does not become an hour of hammering.
#
# HEARTBEATS COME FROM HERE, NEVER FROM JOB CODE. A job spending five
# minutes on a model call is BUSY, not dead; if heartbeats were sent from
# inside job execution, that job would look exactly like a crashed worker
# and get requeued while still running. So a separate task ticks on a
# fixed timer regardless of what jobs are doing. The runtime says "I am
# alive"; jobs say "I am progressing". Different claims, different clocks.
# =============================================================================

from __future__ import annotations

import asyncio
import signal

import websockets
from pydantic import BaseModel, ValidationError

from jarvis.common.envelope import Envelope, UnknownKind, make_envelope
from jarvis.common.log import get_logger
from jarvis.common.worker_protocol import (
    CORE_JOB_CANCEL,
    CORE_JOB_OFFER,
    CORE_WELCOME,
    WORKER_HEARTBEAT,
    WORKER_HELLO,
    WORKER_JOB_ACCEPT,
    WORKER_JOB_DECLINE,
    CoreJobCancel,
    CoreJobOffer,
    CoreWelcome,
    WorkerHeartbeat,
    WorkerHello,
    WorkerJobAccept,
    WorkerJobDecline,
)
from jarvis.agentloop.mcp_client import McpHost
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.llm.layer import LLMLayer
from jarvis.worker.mcp_host import load_mcp_configs
from jarvis.worker.runner import JobRunner
from jarvis.worker.settings import WorkerSettings

log = get_logger("worker.app")

_BACKOFF_START_S = 2.0
_BACKOFF_MAX_S = 60.0


class WorkerApp:
    """One worker process: connect, work, reconnect, forever."""

    def __init__(
        self,
        settings: WorkerSettings,
        registry: JobTypeRegistry,
        llm: "LLMLayer | None" = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._llm = llm
        self._stop = asyncio.Event()
        self._heartbeat_interval_s = 15
        self.mcp = McpHost()

    # -- lifecycle ------------------------------------------------------------

    async def run(self) -> None:
        """Dial the Core until told to stop, reconnecting on any failure."""
        self._install_signal_handlers()
        backoff = _BACKOFF_START_S

        log.info("worker starting", extra={
            "worker_id": self._settings.worker_id,
            "core_url": self._settings.core_url,
            "capabilities": self._settings.capabilities,
        })

        # MCP servers start once for the worker's lifetime, not per
        # connection: launching a browser on every reconnect would be
        # slow and pointless.
        await self.mcp.start_all(load_mcp_configs())
        if self.mcp.server_names():
            log.info("mcp servers running", extra={
                "servers": self.mcp.server_names(),
            })

        # Subagent job types register AFTER MCP, because their handlers
        # close over the running host. A worker with no model access
        # skips them rather than accepting work it cannot do.
        if self._llm is not None:
            from pathlib import Path

            from jarvis.worker.git.config import load_git_config
            from jarvis.worker.git.operations import GitOperations
            from jarvis.worker.subagent_jobs import register_subagent_jobs

            git_ops = None
            token = self._settings.github_token.get_secret_value().strip()
            git_config = load_git_config()
            if token and git_config.repos:
                git_ops = GitOperations(
                    git_config, token, Path("data/git-workspace")
                )
                log.info("git capability enabled", extra={
                    "repos": sorted(git_config.repos),
                })

            from jarvis.worker.workspace import Workspace
            workspace = Workspace(Path(self._settings.workspace_dir))
            log.info("workspace ready", extra={"path": str(workspace.root)})

            register_subagent_jobs(
                self._registry, self._llm, self.mcp, workspace
            )

        while not self._stop.is_set():
            try:
                await self._session()
                backoff = _BACKOFF_START_S     # a clean session resets it
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("connection lost - retrying", extra={
                    "error": f"{type(exc).__name__}: {exc}",
                    "retry_in_s": backoff,
                })

            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, _BACKOFF_MAX_S)

        await self.mcp.stop_all()
        log.info("worker stopped")

    def request_stop(self) -> None:
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                log.warning("signal handlers unavailable on this platform")

    # -- one connection -------------------------------------------------------

    async def _session(self) -> None:
        """One full connection: hello, welcome, then work until it ends."""
        async with websockets.connect(self._settings.core_url) as socket:
            self._socket = socket
            runner = JobRunner(self._registry, self._send)

            await self._send(WORKER_HELLO, WorkerHello(
                worker_id=self._settings.worker_id,
                token=self._settings.worker_token.get_secret_value(),
                capabilities=self._settings.capabilities,
                max_concurrency=self._settings.max_concurrency,
            ))

            welcome = await self._expect(CORE_WELCOME, timeout=10.0)
            assert isinstance(welcome, CoreWelcome)
            self._heartbeat_interval_s = welcome.heartbeat_interval_s
            log.info("connected to core", extra={
                "heartbeat_interval_s": welcome.heartbeat_interval_s,
                "lease_ttl_s": welcome.lease_ttl_s,
            })

            heartbeat = asyncio.create_task(
                self._heartbeat_loop(runner), name="heartbeat"
            )
            try:
                await self._receive_loop(runner)
            finally:
                heartbeat.cancel()

    async def _heartbeat_loop(self, runner: JobRunner) -> None:
        """Say "still here" on a fixed timer, whatever the jobs are doing.

        Sent even when idle, so the Core can tell a quiet worker from a
        departed one.
        """
        while True:
            await asyncio.sleep(self._heartbeat_interval_s)
            await self._send(WORKER_HEARTBEAT, WorkerHeartbeat(
                running_job_ids=runner.running_job_ids,
            ))

    async def _receive_loop(self, runner: JobRunner) -> None:
        async for raw in self._socket:
            try:
                envelope = Envelope.model_validate_json(raw)
                payload = envelope.parse_payload()
            except (UnknownKind, ValidationError) as exc:
                # A newer Core may speak kinds this build does not know.
                # Log and carry on rather than dropping the connection.
                log.warning("unparseable frame from core", extra={
                    "error": str(exc)[:200],
                })
                continue

            if envelope.kind == CORE_JOB_OFFER:
                assert isinstance(payload, CoreJobOffer)
                await self._consider(runner, payload)

            elif envelope.kind == CORE_JOB_CANCEL:
                assert isinstance(payload, CoreJobCancel)
                cancelled = runner.cancel(payload.job_id)
                log.info("cancel received", extra={
                    "job_id": payload.job_id, "was_running": cancelled,
                })

    async def _consider(self, runner: JobRunner, offer: CoreJobOffer) -> None:
        """Accept or decline. Declining is a normal answer, not a failure:
        it returns the job to the queue costing it nothing."""
        if not runner.can_run(offer.type):
            await self._send(WORKER_JOB_DECLINE, WorkerJobDecline(
                job_id=offer.job_id,
                reason=f"no handler for {offer.type} on this worker",
            ))
            return
        if len(runner.running_job_ids) >= self._settings.max_concurrency:
            await self._send(WORKER_JOB_DECLINE, WorkerJobDecline(
                job_id=offer.job_id, reason="at capacity",
            ))
            return

        await self._send(WORKER_JOB_ACCEPT, WorkerJobAccept(job_id=offer.job_id))
        runner.start(offer)
        log.info("job accepted", extra={
            "job_id": offer.job_id, "type": offer.type,
        })

    # -- plumbing -------------------------------------------------------------

    async def _send(self, kind: str, payload: BaseModel) -> None:
        envelope = make_envelope(kind, payload)
        await self._socket.send(envelope.model_dump_json())

    async def _expect(self, kind: str, timeout: float) -> BaseModel:
        """Wait for one specific frame - used only for the handshake,
        where the protocol is strictly ordered."""
        raw = await asyncio.wait_for(self._socket.recv(), timeout=timeout)
        envelope = Envelope.model_validate_json(raw)
        if envelope.kind != kind:
            raise RuntimeError(
                f"expected {kind}, got {envelope.kind}: {envelope.payload}"
            )
        return envelope.parse_payload()