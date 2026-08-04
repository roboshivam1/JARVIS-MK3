# =============================================================================
# src/jarvis/core/app.py - the Core daemon: boot, supervise, shutdown
# =============================================================================
#
# The lifecycle, in three stages:
#
#   boot      - storage, migrations, recovery, then the whole component
#               graph. No servers start here, so tests can boot and
#               inspect without binding ports.
#   supervise - run residents and watch them.
#   shutdown  - stop residents, close storage, exit.
#
# RESIDENTS ARE NOT ALL EQUAL, and treating them as if they were has
# taken this daemon down twice for a dropped wifi connection:
#
#   ESSENTIAL - the queue, reclaim, notifier, approvals, initiative.
#     These ARE the Core. One dying leaves the system broken in ways
#     that are worse to limp through than to restart out of, so the
#     daemon exits loudly and systemd brings it back through recovery.
#
#   OPTIONAL - Telegram, and voice later. Surfaces ONTO the Core, not
#     the Core itself. One dying means losing a way to talk to JARVIS
#     while memory, jobs, schedules, and every other surface carry on.
#     These get restarted with backoff.
#
# Same failure taxonomy as everywhere else in this system: retry what
# retrying can fix. It keeps turning out that most failures are
# transient.
# =============================================================================

from __future__ import annotations

import asyncio
import signal
import time

import uvicorn

# Importing the protocols registers their envelope kinds.
import jarvis.common.client_protocol  # noqa: F401
import jarvis.common.worker_protocol  # noqa: F401
from jarvis.common.events import Event, EventKind
from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.log import get_logger
from jarvis.common.schedules import Schedule, ScheduleKind
from jarvis.common.settings import CoreSettings
from jarvis.core.approvals.service import ApprovalService
from jarvis.core.clients.telegram import TelegramBridge
from jarvis.core.db.database import Database
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.facts import FactsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.schedules import SchedulesRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.db.repos.watchers import WatchersRepo
from jarvis.core.gateway.clients import ClientRegistry
from jarvis.core.gateway.http import GatewayDeps, build_status_snapshot, create_app
from jarvis.core.initiative.engine import InitiativeEngine, next_cron_time
from jarvis.core.initiative.notifier import Notifier
from jarvis.core.initiative.policy import NotificationPolicy
from jarvis.core.memory.profile import ProfileStore
from jarvis.core.memory.service import MemoryService
from jarvis.core.observability.traces import TracesRepo, make_db_trace_sink
from jarvis.core.orchestrator.agent import Orchestrator
from jarvis.core.queue.coreworker import CoreWorker
from jarvis.core.queue.dispatcher import ReclaimLoop
from jarvis.core.queue.registry import JobTypeRegistry
from jarvis.core.queue.registry_workers import WorkerRegistry
from jarvis.core.sessionmgr import SessionManager
from jarvis.llm.embeddings import create_embedder
from jarvis.llm.layer import LLMLayer
from jarvis.llm.speech import create_speaker
from jarvis.llm.transcription import Transcriber

log = get_logger("core.app")

# How long an optional resident waits before restarting, and the ceiling.
_RESTART_BASE_S = 5.0
_RESTART_MAX_S = 300.0


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
        self.watchers: WatchersRepo | None = None
        self.initiative: InitiativeEngine | None = None
        self.approvals: ApprovalsRepo | None = None
        self.approval_service: ApprovalService | None = None
        self.llm: LLMLayer | None = None
        self.session_mgr: SessionManager | None = None
        self.telegram: TelegramBridge | None = None
        self.gateway_deps: GatewayDeps | None = None
        self.policy = NotificationPolicy()
        self.transcriber = Transcriber(settings)
        self.speaker = create_speaker(settings)
        self.registry: JobTypeRegistry = JobTypeRegistry()
        self.workers: WorkerRegistry = WorkerRegistry()
        self.clients: ClientRegistry = ClientRegistry()
        self._reclaim: ReclaimLoop | None = None
        self._core_worker: CoreWorker | None = None
        self._uvicorn: uvicorn.Server | None = None
        self._stop = asyncio.Event()
        self._booted = False

    # -- stage 1: boot --------------------------------------------------------

    async def boot(self) -> None:
        """Storage up, schema current, component graph constructed."""
        if self._booted:
            raise RuntimeError("CoreApp.boot() called twice on one instance")
        self._booted = True

        started = time.monotonic()
        self._boot_trace = new_ulid()

        self.settings.ensure_data_dirs()
        self.db = await Database.connect(self.settings.db_path)
        migrations_ran = await self.db.migrate()

        # -- repositories -----------------------------------------------------
        self.events = EventsRepo(self.db)
        self.sessions = SessionsRepo(self.db)
        self.traces = TracesRepo(self.db)
        self.jobs = JobsRepo(self.db, self.events)
        self.artifacts = ArtifactsRepo(self.db, self.settings.artifacts_dir)
        self.notifications = NotificationsRepo(self.db)
        self.approvals = ApprovalsRepo(self.db)
        self.schedules = SchedulesRepo(self.db)
        self.watchers = WatchersRepo(self.db)

        # -- queue ------------------------------------------------------------
        self._reclaim = ReclaimLoop(self.jobs, self.events)
        self._core_worker = CoreWorker(
            self.jobs, self.events, self.registry, self.artifacts
        )

        await self._recover()

        # -- intelligence and memory ------------------------------------------
        self.llm = LLMLayer(self.settings, trace_sink=make_db_trace_sink(self.db))
        if not self.llm.available:
            log.warning("no API key - daemon runs, conversation will not")

        self.memory = MemoryService(
            FactsRepo(self.db), create_embedder(self.settings)
        )
        self.profile = ProfileStore(self.db)

        # -- initiative and approvals -----------------------------------------
        self.approval_service = ApprovalService(
            self.approvals, self.jobs, self.events
        )
        self.initiative = InitiativeEngine(
            self.schedules, self.jobs, self.events,
            self.settings.tz, self.registry,
        )
        self.notifier = Notifier(
            self.jobs, self.sessions, self.notifications,
            self.artifacts, self.events, self.approvals,
            policy=self.policy, tz=self.settings.tz,
        )

        # -- job types, AFTER their dependencies exist ------------------------
        #
        # Registering earlier means a handler closes over None and fails
        # at run time rather than at boot - which is exactly what the
        # sleep cycle did, three identical retries before giving up.
        from jarvis.jobs.backup import register_backup_jobs
        from jarvis.jobs.browser import register_browser_job_metadata
        from jarvis.jobs.code import register_code_job_metadata
        from jarvis.jobs.maintenance import register_maintenance_jobs
        from jarvis.jobs.research import register_research_jobs
        from jarvis.jobs.watch import register_watch_jobs
        from jarvis.jobs.worker_types import register_worker_job_types
        from jarvis.jobs.writing import register_writing_jobs

        register_research_jobs(self.registry, self.llm)
        register_worker_job_types(self.registry)
        register_browser_job_metadata(self.registry)
        register_code_job_metadata(self.registry)
        register_writing_jobs(self.registry, self.llm, self.artifacts)
        register_backup_jobs(self.registry, self.settings)
        register_watch_jobs(
            self.registry, self.watchers, self.notifications,
            self.events, self.traces,
        )
        register_maintenance_jobs(
            self.registry, self.llm, self.memory, FactsRepo(self.db),
            self.sessions, self.profile, self.events,
        )

        # -- conversation -----------------------------------------------------
        orchestrator = Orchestrator(
            self.llm, self.settings, self.jobs, self.events,
            self.registry, self.memory,
            artifacts=self.artifacts,
            watchers=self.watchers, schedules=self.schedules,
            tz=self.settings.tz,
        )
        self.session_mgr = SessionManager(
            self.sessions, self.events, orchestrator,
            memory=self.memory, profile=self.profile,
            artifacts=self.artifacts,
        )

        # -- gateway ----------------------------------------------------------
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
            approvals=self.approval_service,
            clients=self.clients,
            session_mgr=self.session_mgr,
            sessions=self.sessions,
            watchers=self.watchers,
            transcriber=self.transcriber,
            speaker=self.speaker,
        )

        # -- clients ----------------------------------------------------------
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
                policy=self.policy,
                tz=self.settings.tz,
                transcriber=self.transcriber,
            )
            self.notifier.register_deliverer("telegram", self.telegram)
        else:
            log.info("no telegram token - bridge disabled")

        # The web client is a delivery surface too. It refuses when no
        # tab is open, leaving the notification pending for Telegram -
        # an open browser is a bonus, not the only path.
        self.notifier.register_deliverer("web", self.clients)

        await self._seed_schedules()

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

    async def _seed_schedules(self) -> None:
        """Create default schedules if absent. ensure() leaves existing
        rows untouched, so the owner's edits survive every restart."""
        assert self.schedules is not None

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

        await self.schedules.ensure(Schedule(
            name="nightly backup",
            kind=ScheduleKind.CRON,
            cron_expr=self.settings.backup_cron,
            job_type="system.backup",
            job_payload={"reason": "scheduled"},
            priority=8,
            next_fire_ts=next_cron_time(
                self.settings.backup_cron, self.settings.tz
            ),
        ))

        await self.schedules.ensure(Schedule(
            name="watcher tick",
            kind=ScheduleKind.INTERVAL,
            interval_s=900,
            job_type="watch.check",
            job_payload={"reason": "scheduled"},
            priority=7,
            next_fire_ts=utc_now(),
        ))

    async def _recover(self) -> None:
        """Repair anything a previous life left mid-flight. Every
        leased or running job is orphaned after a restart."""
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
        """boot, start residents, watch until stop or an essential death."""
        self._install_signal_handlers()
        if not self._booted:
            await self.boot()

        essential: dict[str, asyncio.Task[None]] = {}
        optional: dict[str, asyncio.Task[None]] = {}

        assert self.gateway_deps is not None
        config = uvicorn.Config(
            create_app(self.gateway_deps),
            host=self.settings.gateway_host,
            port=self.settings.gateway_port,
            log_config=None,
        )
        self._uvicorn = uvicorn.Server(config)

        # The gateway is essential: without it there is no way in at
        # all, for clients or workers.
        essential["gateway"] = asyncio.create_task(
            self._uvicorn.serve(), name="gateway"
        )

        assert self._core_worker is not None and self._reclaim is not None
        assert self.notifier is not None and self.initiative is not None
        assert self.approval_service is not None

        for name, coroutine in (
            ("core-worker", self._core_worker.run()),
            ("reclaim", self._reclaim.run()),
            ("notifier", self.notifier.run()),
            ("initiative", self.initiative.run()),
            ("approvals", self.approval_service.run()),
        ):
            essential[name] = asyncio.create_task(coroutine, name=name)

        # Optional residents are restarted rather than mourned. Telegram
        # losing DNS for thirty seconds has taken this daemon down twice;
        # it is a surface onto the Core, not the Core.
        if self.telegram is not None:
            optional["telegram"] = asyncio.create_task(
                self._keep_alive("telegram", self.telegram.run),
                name="telegram-supervisor",
            )

        log.info("core running", extra={
            "essential": list(essential), "optional": list(optional),
        })

        stop_wait = asyncio.create_task(self._stop.wait(), name="stop-flag")
        done, _ = await asyncio.wait(
            [stop_wait, *essential.values()],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for name, task in essential.items():
            if task in done and (exc := task.exception()) is not None:
                log.critical("essential resident crashed - stopping daemon",
                             extra={"resident": name},
                             exc_info=(type(exc), exc, exc.__traceback__))

        await self._stop_residents({**essential, **optional}, stop_wait)
        await self.shutdown()

    async def _keep_alive(self, name: str, factory) -> None:  # type: ignore[no-untyped-def]
        """Run an optional resident forever, restarting it on failure.

        Backoff grows so a permanently broken resident does not spin,
        and resets on a clean run so a transient failure hours later
        starts from a short wait rather than a long one.
        """
        backoff = _RESTART_BASE_S
        while not self._stop.is_set():
            try:
                await factory()
                return          # clean exit: it was asked to stop
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("optional resident crashed - restarting", extra={
                    "resident": name,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                    "retry_in_s": backoff,
                })
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                return          # stopped while waiting
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, _RESTART_MAX_S)

    async def _stop_residents(
        self,
        residents: dict[str, asyncio.Task[None]],
        stop_wait: asyncio.Task[bool],
    ) -> None:
        stop_wait.cancel()
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
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
