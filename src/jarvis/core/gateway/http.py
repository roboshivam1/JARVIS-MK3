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
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, WebSocket

from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.gateway.auth import BearerAuth
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.gateway.workers import WorkerConnection
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.queue.registry import JobTypeRegistry
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
    job_types: "JobTypeRegistry | None" = None


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
    }


def create_app(deps: GatewayDeps) -> FastAPI:
    """Build the gateway app around ready-made dependencies."""
    auth = BearerAuth(deps.settings.gateway_token.get_secret_value())
    app = FastAPI(title="jarvis-core", docs_url=None, redoc_url=None)

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
        )
        await connection.run()

    return app