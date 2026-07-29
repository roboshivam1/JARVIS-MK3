# =============================================================================
# src/jarvis/core/app.py - the Core daemon: boot, run, recover, shutdown
# =============================================================================
#
# The lifecycle of the Core, always the same three stages:
#
#   boot     - open storage, migrate schema, run recovery, announce start
#   run      - keep long-running tasks alive until asked to stop
#              (phase 0 has no tasks yet; later phases add gateway,
#               telegram poller, dispatcher, initiative engine here)
#   shutdown - close storage cleanly, exit
#
# Recovery is a NORMAL step of every boot, not a special crash mode. The
# Core never asks "did I crash last time?" - it inspects stored state and
# repairs whatever that state implies (in phase 2: requeue expired job
# leases, retry orphaned running jobs). A fresh boot and a crash-restart
# walk the exact same path; the difference is only what they find.
#
# Stopping: SIGINT (Ctrl-C) and SIGTERM (systemd stop) both raise one
# internal stop flag; the run stage waits on that flag. One mechanism for
# laptop and VPS alike.
# =============================================================================

from __future__ import annotations

import asyncio
import signal
import time

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import new_ulid
from jarvis.common.log import get_logger
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.events import EventsRepo

log = get_logger("core.app")


class CoreApp:
    """The always-on daemon. One instance per process.

    Test-friendly by construction: boot(), shutdown(), and run() are
    separate methods, so tests can boot and shut down programmatically
    without signals or an infinite wait.
    """

    def __init__(self, settings: CoreSettings) -> None:
        self.settings = settings
        self.db: Database | None = None
        self.events: EventsRepo | None = None
        # The stop flag. Raised by signal handlers (or tests); awaited by run().
        self._stop = asyncio.Event()

    # -- stage 1: boot --------------------------------------------------------

    async def boot(self) -> None:
        """Bring the Core to a fully operational state. Must be fast
        (seconds) and must never require human interaction - systemd may
        be rebooting us at 4 a.m. with nobody watching."""
        started = time.monotonic()

        # Each boot is a root cause of its own, so it gets its own trace id:
        # everything this boot does can be found with one query later.
        self._boot_trace = new_ulid()

        self.settings.ensure_data_dirs()

        self.db = await Database.connect(self.settings.db_path)
        migrations_ran = await self.db.migrate()
        self.events = EventsRepo(self.db)

        await self._recover()

        await self.events.append(Event(
            kind=EventKind.CORE_STARTED,
            source="core.app",
            trace_id=self._boot_trace,
            payload={
                "migrations_applied": migrations_ran,
                "boot_seconds": round(time.monotonic() - started, 3),
            },
        ))
        log.info("core started", extra={
            "trace_id": self._boot_trace,
            "db_path": str(self.settings.db_path),
            "migrations_applied": migrations_ran,
        })

    async def _recover(self) -> None:
        """Inspect stored state and repair anything a previous run left
        mid-flight. Runs on EVERY boot.

        Phase 0: there is no job queue or lease state yet, so there is
        nothing that can need repair - this is a documented no-op. Phase 2
        fills it in: requeue expired leases, mark orphaned running jobs
        for retry, reschedule due timers. When recovery actually repairs
        something, it emits a core.recovered event with the counts.
        """
        log.debug("recovery check: nothing recoverable exists in phase 0")

    # -- stage 2: run ---------------------------------------------------------

    async def run(self) -> None:
        """boot, then stay alive until the stop flag is raised, then
        shutdown. This is the method the entry point calls."""
        self._install_signal_handlers()
        await self.boot()

        # Phase 0 has no long-running tasks, so running == waiting.
        # Later phases replace this single wait with task supervision:
        # gateway, telegram poller, dispatcher, initiative engine - all
        # started here, all cancelled on stop.
        log.info("core running - press Ctrl-C to stop")
        await self._stop.wait()

        await self.shutdown()

    def request_stop(self) -> None:
        """Raise the stop flag. Safe to call from signal handlers or tests."""
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                # Some platforms (notably Windows) lack loop signal
                # handlers. Ctrl-C still works there via KeyboardInterrupt;
                # our deployment targets (macOS, Linux) support this fully.
                log.warning("signal handlers unavailable on this platform")

    # -- stage 3: shutdown ----------------------------------------------------

    async def shutdown(self) -> None:
        """Close everything cleanly. Idempotent: safe to call twice.

        Note there is no core.stopped event - the event taxonomy does not
        define one. Absence of activity plus a clean WAL is what a clean
        stop looks like; a crash needs no marker either, because recovery
        reads state, not markers.
        """
        if self.db is not None:
            await self.db.close()
            self.db = None
            self.events = None
        log.info("core stopped")