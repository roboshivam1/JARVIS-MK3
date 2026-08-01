# =============================================================================
# src/jarvis/core/app.py - the Core daemon: boot, supervise, shutdown
# =============================================================================
#
# The lifecycle, now with real residents:
#
#   boot      - storage, migrations, recovery, then construct the whole
#               component graph: LLM layer (db-backed tracing), the
#               orchestrator, the session manager, gateway, Telegram.
#   supervise - run gateway + Telegram as named asyncio tasks and watch.
#               Stop flag raised -> orderly shutdown. A resident CRASHING
#               also stops the daemon: a loud death plus a clean restart
#               (systemd's job) beats limping along half-alive.
#   shutdown  - stop residents, close storage, exit.
#
# Optional residents degrade gracefully at construction time: no Telegram
# token -> no bridge; no API key -> conversation errors clearly but the
# daemon, gateway, and storage all still run.
#
# boot() deliberately starts NO servers - tests boot and inspect the
# component graph without binding ports. Servers start in run().
# =============================================================================

from __future__ import annotations

import asyncio
import signal
import time

import uvicorn

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import new_ulid
from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings
from jarvis.core.clients.telegram import TelegramBridge
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.gateway.http import GatewayDeps, build_status_snapshot, create_app
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.facts import FactsRepo
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.schedules import SchedulesRepo
from jarvis.core.approvals.service import ApprovalService
from jarvis.common.schedules import Schedule, ScheduleKind
from jarvis.core.initiative.engine import InitiativeEngine, next_cron_time
from jarvis.core.initiative.notifier import Notifier
from jarvis.core.memory.profile import ProfileStore
from jarvis.core.memory.service import MemoryService
from jarvis.llm.embeddings import create_embedder
from jarvis.core.observability.traces import TracesRepo, make_db_trace_sink
from jarvis.core.queue.coreworker import CoreWorker
from jarvis.core.queue.dispatcher import ReclaimLoop
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.core.queue.registry_workers import WorkerRegistry
# Importing the protocol registers its envelope kinds.
import jarvis.common.worker_protocol  # noqa: F401
from jarvis.core.orchestrator.agent import Orchestrator
from jarvis.core.sessionmgr import SessionManager
from jarvis.llm.layer import LLMLayer

log = get_logger("core.app")


class CoreApp:
    """The always-on daemon. One instance per process."""

    def __init__(self, settings: CoreSettings) -> None:
        self.settings = settings
        self.db: Database | None = None
        self.events: EventsRepo | None = None
        self.sessions: SessionsRepo | None = None
        self.traces: TracesRepo | None = None
        self.jobs: JobsRepo | None = None
        self.artifacts: ArtifactsRepo | None = None
        self.notifications: NotificationsRepo | None = None
        self.notifier: Notifier | None = None
        self.memory: MemoryService | None = None
        self.profile: ProfileStore | None = None
        self.schedules: SchedulesRepo | None = None
        self.initiative: InitiativeEngine | None = None
        self.approvals: ApprovalsRepo | None = None
        self.approval_service: ApprovalService | None = None
        self.registry: JobTypeRegistry = JobTypeRegistry()
        self.workers: WorkerRegistry = WorkerRegistry()
        self._reclaim: ReclaimLoop | None = None
        self._core_worker: CoreWorker | None = None
        self.llm: LLMLayer | None = None
        self.session_mgr: SessionManager | None = None
        self.telegram: TelegramBridge | None = None
        self.gateway_deps: GatewayDeps | None = None
        self._uvicorn: uvicorn.Server | None = None
        self._stop = asyncio.Event()
        self._booted = False

    # -- stage 1: boot --------------------------------------------------------

    async def boot(self) -> None:
        """Storage up, schema current, component graph constructed.
        Fast, non-interactive, and server-free (servers start in run())."""
        if self._booted:
            # Booting twice would re-register job types, re-open the
            # database, and duplicate residents. Always a wiring mistake.
            raise RuntimeError("CoreApp.boot() called twice on one instance")
        self._booted = True

        started = time.monotonic()
        self._boot_trace = new_ulid()

        self.settings.ensure_data_dirs()
        self.db = await Database.connect(self.settings.db_path)
        migrations_ran = await self.db.migrate()

        self.events = EventsRepo(self.db)
        self.sessions = SessionsRepo(self.db)
        self.traces = TracesRepo(self.db)
        self.jobs = JobsRepo(self.db, self.events)
        self.artifacts = ArtifactsRepo(self.db, self.settings.artifacts_dir)
        self._reclaim = ReclaimLoop(self.jobs, self.events)
        self._core_worker = CoreWorker(
            self.jobs, self.events, self.registry, self.artifacts
        )
        self.approvals = ApprovalsRepo(self.db)
        self.approval_service = ApprovalService(
            self.approvals, self.jobs, self.events
        )
        self.schedules = SchedulesRepo(self.db)
        self.initiative = InitiativeEngine(
            self.schedules, self.jobs, self.events,
            self.settings.tz, self.registry,
        )
        self.notifications = NotificationsRepo(self.db)
        self.notifier = Notifier(
            self.jobs, self.sessions, self.notifications,
            self.artifacts, self.events, self.approvals,
        )

        await self._recover()

        # Intelligence: every model call traced into the database.
        self.llm = LLMLayer(self.settings, trace_sink=make_db_trace_sink(self.db))
        if not self.llm.available:
            log.warning("no API key - daemon runs, conversation will not")

        # Job types register at boot, after their dependencies exist.
        from jarvis.jobs.maintenance import register_maintenance_jobs
        from jarvis.jobs.research import register_research_jobs
        register_research_jobs(self.registry, self.llm)
        # Worker-executed job types are registered on the Core too, for
        # their metadata (timeout, models) - the Core never runs them,
        # since they declare capabilities it does not have. This is what
        # lets the orchestrator offer work it cannot itself perform.
        from jarvis.jobs.browser import register_browser_job_metadata
        from jarvis.jobs.worker_types import register_worker_job_types
        register_worker_job_types(self.registry)
        register_browser_job_metadata(self.registry)

        register_maintenance_jobs(
            self.registry, self.llm, self.memory, FactsRepo(self.db),
            self.sessions, self.profile, self.events,
        )

        # Seed the nightly sleep cycle. ensure() leaves an existing
        # schedule untouched, so the owner's edits (a moved hour, a
        # disabled row) survive every restart.
        await self.schedules.ensure(Schedule(
            name="nightly memory sleep cycle",
            kind=ScheduleKind.CRON,
            cron_expr=self.settings.sleep_cycle_cron,
            job_type="memory.sleep_cycle",
            job_payload={"reason": "scheduled"},
            priority=8,
            next_fire_ts=next_cron_time(
                self.settings.sleep_cycle_cron, self.settings.tz
            ),
        ))

        # Memory: an embedder (local or hosted, per config), the fact
        # vault, and the standing profile document.
        self.memory = MemoryService(
            FactsRepo(self.db), create_embedder(self.settings)
        )
        self.profile = ProfileStore(self.db)

        orchestrator = Orchestrator(
            self.llm, self.settings, self.jobs, self.events,
            self.registry, self.memory,
        )
        self.session_mgr = SessionManager(
            self.sessions, self.events, orchestrator,
            memory=self.memory, profile=self.profile,
        )

        # Gateway app (server starts in run()).
        self.gateway_deps = GatewayDeps(
            settings=self.settings,
            db=self.db,
            traces=self.traces,
            started_monotonic=started,
            workers=self.workers,
            jobs=self.jobs,
            events=self.events,
            artifacts=self.artifacts,
            job_types=self.registry,
        )

        # Telegram bridge, only if configured.
        token = self.settings.telegram_bot_token.get_secret_value().strip()
        if token:
            self.telegram = TelegramBridge(
                token=token,
                owner_id=self.settings.telegram_owner_id,
                session_mgr=self.session_mgr,
                sessions_repo=self.sessions,
                status_provider=lambda: build_status_snapshot(self.gateway_deps),  # type: ignore[arg-type]
                approval_service=self.approval_service,
                approvals_repo=self.approvals,
            )
        # The bridge is now also a delivery surface for unprompted
            # messages, not just a request/response client.
            self.notifier.register_deliverer("telegram", self.telegram)
        else:
            log.info("no telegram token - bridge disabled")

        await self.events.append(Event(
            kind=EventKind.CORE_STARTED,
            source="core.app",
            trace_id=self._boot_trace,
            payload={
                "migrations_applied": migrations_ran,
                "boot_seconds": round(time.monotonic() - started, 3),
                "telegram": self.telegram is not None,
                "llm_available": self.llm.available,
            },
        ))
        log.info("core booted", extra={
            "trace_id": self._boot_trace,
            "migrations_applied": migrations_ran,
        })

    async def _recover(self) -> None:
        """Inspect stored state and repair anything a previous life left
        mid-flight: every leased/running job is orphaned after a restart
        and goes back through the shared failure path (requeue with
        backoff, or terminal failure if attempts are spent)."""
        assert self._reclaim is not None
        reclaimed = await self._reclaim.reclaim_all_inflight()
        if reclaimed:
            assert self.events is not None
            await self.events.append(Event(
                kind=EventKind.CORE_RECOVERED,
                source="core.app",
                trace_id=self._boot_trace,
                payload={"jobs_reclaimed": reclaimed},
            ))
            log.info("recovery reclaimed jobs", extra={"count": reclaimed})
    # -- stage 2: supervise ---------------------------------------------------

    async def run(self) -> None:
        """boot, start residents, watch until stop or a resident dies."""
        self._install_signal_handlers()
        # Callers may boot first (to enqueue work before residents start);
        # run() boots only if that has not happened yet.
        if not self._booted:
            await self.boot()
        residents: dict[str, asyncio.Task[None]] = {}

        assert self.gateway_deps is not None
        app = create_app(self.gateway_deps)
        config = uvicorn.Config(
            app,
            host=self.settings.gateway_host,
            port=self.settings.gateway_port,
            log_config=None,   # our JSON logging, not uvicorn's
        )
        self._uvicorn = uvicorn.Server(config)
        residents["gateway"] = asyncio.create_task(
            self._uvicorn.serve(), name="gateway"
        )

        if self.telegram is not None:
            residents["telegram"] = asyncio.create_task(
                self.telegram.run(), name="telegram"
            )

        assert self._core_worker is not None and self._reclaim is not None
        residents["core-worker"] = asyncio.create_task(
            self._core_worker.run(), name="core-worker"
        )
        residents["reclaim"] = asyncio.create_task(
            self._reclaim.run(), name="reclaim"
        )
        assert self.notifier is not None
        residents["notifier"] = asyncio.create_task(
            self.notifier.run(), name="notifier"
        )
        assert self.initiative is not None
        residents["initiative"] = asyncio.create_task(
            self.initiative.run(), name="initiative"
        )
        assert self.approval_service is not None
        residents["approvals"] = asyncio.create_task(
            self.approval_service.run(), name="approvals"
        )

        log.info("core running", extra={"residents": list(residents)})

        # Wait for EITHER the stop flag OR any resident finishing (which,
        # for a forever-task, means it crashed).
        stop_wait = asyncio.create_task(self._stop.wait(), name="stop-flag")
        done, _ = await asyncio.wait(
            [stop_wait, *residents.values()],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for name, task in residents.items():
            if task in done and (exc := task.exception()) is not None:
                log.critical("resident crashed - stopping daemon",
                             extra={"resident": name},
                             exc_info=(type(exc), exc, exc.__traceback__))

        await self._stop_residents(residents, stop_wait)
        await self.shutdown()

    async def _stop_residents(
        self,
        residents: dict[str, asyncio.Task[None]],
        stop_wait: asyncio.Task[bool],
    ) -> None:
        stop_wait.cancel()
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True   # uvicorn's polite stop
        for task in residents.values():
            task.cancel()
        await asyncio.gather(*residents.values(), return_exceptions=True)

    def request_stop(self) -> None:
        """Raise the stop flag. Safe from signal handlers and tests."""
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                log.warning("signal handlers unavailable on this platform")

    # -- stage 3: shutdown ----------------------------------------------------

    async def shutdown(self) -> None:
        """Close storage cleanly. Idempotent."""
        if self.db is not None:
            await self.db.close()
            self.db = None
        log.info("core stopped")