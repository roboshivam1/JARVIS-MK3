# =============================================================================
# src/jarvis/core/gateway/http.py - the Core's HTTP door
# =============================================================================
#
# A small FastAPI app created by a factory. The factory receives its
# dependencies ready-made (database, repos, settings) and wires routes;
# importing this module does nothing. The boot sequence runs the app as
# one asyncio task via uvicorn; tests run it in-process with throwaway
# parts.
#
# Phase 1 surface, deliberately tiny:
#   GET /health - unauthenticated liveness probe: "the process is up".
#                 For systemd, uptime monitors, and 2 a.m. curl.
#   GET /status - authed snapshot: uptime, today's spend, conversation
#                 counts. The SAME snapshot the Telegram /status command
#                 renders - computed once here, delivered through any door.
#
# The client WebSocket endpoint from the design docs arrives with the
# first real WS client (web/voice); building it with no client to speak
# to would be untested guesswork. Flagged deferral, not an omission.
# =============================================================================

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse

# Generous but bounded. A dataset large enough to blow this is a
# dataset that should be trimmed before an agent looks at it.
MAX_UPLOAD_BYTES = 50_000_000

from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.gateway.auth import BearerAuth
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.approvals.service import ApprovalService
from jarvis.core.gateway.clients import ClientConnection, ClientRegistry
from jarvis.core.gateway.workers import WorkerConnection
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.sessionmgr import SessionManager
from jarvis.llm.speech import SpeechBackend
from jarvis.llm.transcription import Transcriber
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.core.db.repos.watchers import WatchersRepo
from jarvis.core.queue.registry_workers import WorkerRegistry

log = get_logger("core.gateway")


@dataclass
class GatewayDeps:
    """Everything the gateway needs, handed in by the boot sequence."""

    settings: CoreSettings
    db: Database
    traces: TracesRepo
    started_monotonic: float   # time.monotonic() at boot, for uptime
    # Worker plumbing. Optional so tests can build a gateway without the
    # queue; the endpoint simply refuses connections when absent.
    workers: "WorkerRegistry | None" = None
    jobs: "JobsRepo | None" = None
    events: "EventsRepo | None" = None
    artifacts: "ArtifactsRepo | None" = None
    watchers: "WatchersRepo | None" = None
    transcriber: "Transcriber | None" = None
    speaker: "SpeechBackend | None" = None
    job_types: "JobTypeRegistry | None" = None
    approvals: "ApprovalService | None" = None
    clients: "ClientRegistry | None" = None
    session_mgr: "SessionManager | None" = None
    sessions: "SessionsRepo | None" = None
    artifacts: "ArtifactsRepo | None" = None


async def build_status_snapshot(deps: GatewayDeps) -> dict[str, Any]:
    """The status answer, as data. Shared by this HTTP route and by the
    Telegram /status command - one computation, many doors."""
    uptime_s = int(time.monotonic() - deps.started_monotonic)

    workers = (
        [
            {"id": w.worker_id, "capabilities": sorted(w.capabilities),
             "running": len(w.running_job_ids)}
            for w in deps.workers.connected()
        ]
        if deps.workers else []
    )

    jobs: list[dict[str, Any]] = []
    if deps.jobs is not None:
        for job in await deps.jobs.live_jobs():
            jobs.append({
                "id": job.id, "type": job.type, "status": job.status.value,
            })

    watchers: list[dict[str, Any]] = []
    if deps.watchers is not None:
        for watcher in await deps.watchers.all():
            watchers.append({
                "name": watcher.name,
                "hits": watcher.hit_count,
                "enabled": watcher.enabled,
            })

    sessions_row = await deps.db.query_one(
        "SELECT COUNT(*) AS n FROM sessions"
    )
    turns_row = await deps.db.query_one("SELECT COUNT(*) AS n FROM turns")
    calls_row = await deps.db.query_one("SELECT COUNT(*) AS n FROM llm_calls")

    return {
        "status": "ok",
        "uptime_s": uptime_s,
        "cost_today_usd": round(await deps.traces.cost_today_usd(), 6),
        "sessions": int(sessions_row["n"]) if sessions_row else 0,
        "turns": int(turns_row["n"]) if turns_row else 0,
        "llm_calls": int(calls_row["n"]) if calls_row else 0,
        "workers": workers,
        "jobs": jobs,
        "watchers": watchers,
    }


def create_app(deps: GatewayDeps) -> FastAPI:
    """Build the gateway app around ready-made dependencies."""
    auth = BearerAuth(deps.settings.gateway_token.get_secret_value())
    app = FastAPI(title="jarvis-core", docs_url=None, redoc_url=None)

    @app.get("/app")
    async def client_page() -> FileResponse:
        """The web client.

        Served by the Core rather than deployed separately: the client
        is only ever as reachable as the Core is, which is the honest
        arrangement. On localhost the browser grants microphone access
        without HTTPS, so voice works here today; from a phone it will
        need a certificate.

        Unauthenticated - the PAGE is not a secret. It prompts for a
        token and every request it makes carries one, so nothing behind
        it opens without the secret.
        """
        return FileResponse(
            Path(__file__).parent / "static" / "app.html",
            media_type="text/html",
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Deliberately unauthenticated and nearly information-free:
        # liveness only. Anything more belongs behind auth in /status.
        return {"status": "alive"}

    @app.get("/status", dependencies=[Depends(auth)])
    async def status() -> dict[str, Any]:
        return await build_status_snapshot(deps)

    @app.websocket("/ws/worker")
    async def worker_socket(websocket: WebSocket) -> None:
        # Auth happens INSIDE the connection, in the hello frame, rather
        # than as an HTTP dependency: WebSocket clients cannot set
        # headers reliably across every runtime, and the handshake needs
        # to carry capabilities anyway.
        if (
            deps.workers is None or deps.jobs is None
            or deps.events is None or deps.artifacts is None
            or deps.job_types is None
        ):
            await websocket.close(code=1011)
            return
        connection = WorkerConnection(
            websocket=websocket,
            expected_token=deps.settings.gateway_token.get_secret_value(),
            registry=deps.workers,
            jobs=deps.jobs,
            events=deps.events,
            artifacts=deps.artifacts,
            job_types=deps.job_types,
            approvals=deps.approvals,
        )
        await connection.run()

    @app.websocket("/ws/client")
    async def client_socket(websocket: WebSocket) -> None:
        if (
            deps.clients is None or deps.session_mgr is None
            or deps.sessions is None
        ):
            await websocket.close(code=1011)
            return
        connection = ClientConnection(
            websocket=websocket,
            expected_token=deps.settings.gateway_token.get_secret_value(),
            session_mgr=deps.session_mgr,
            sessions=deps.sessions,
            registry=deps.clients,
            owner_timezone=deps.settings.owner_timezone,
            approvals=deps.approvals,
            transcriber=deps.transcriber,
            speaker=deps.speaker,
        )
        await connection.run()

    @app.post("/upload", dependencies=[Depends(auth)])
    async def upload(file: UploadFile) -> dict[str, Any]:
        """Take a file and make it an artifact.

        Over HTTP rather than the socket: a 10MB file base64-encoded
        through JSON frames is a third larger and needs reassembly we
        would have to write. A multipart POST is one line in the
        browser and the right tool for bytes.

        The artifact id comes back; the client mentions it in its next
        message and the file becomes available to whichever subagent
        needs it.
        """
        if deps.artifacts is None:
            raise HTTPException(status_code=503, detail="artifacts unavailable")

        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds {MAX_UPLOAD_BYTES // 1_000_000}MB",
            )

        artifact = await deps.artifacts.write(
            name=file.filename or "upload",
            mime=file.content_type or "application/octet-stream",
            content=content,
            created_by="client.upload",
        )
        return {
            "artifact_id": artifact.id,
            "name": artifact.name,
            "size": artifact.size,
        }

    @app.get("/artifact/{artifact_id}", dependencies=[Depends(auth)])
    async def download(artifact_id: str) -> FileResponse:
        """Serve an artifact back - charts, briefs, generated documents."""
        if deps.artifacts is None:
            raise HTTPException(status_code=503, detail="artifacts unavailable")
        artifact = await deps.artifacts.get(artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="no such artifact")
        return FileResponse(
            deps.artifacts.path_for(artifact),
            media_type=artifact.mime,
            filename=artifact.name,
        )

    return app