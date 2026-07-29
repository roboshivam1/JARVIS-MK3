# =============================================================================
# src/jarvis/core/__main__.py - entry point for `python -m jarvis.core`
# =============================================================================
#
# Wiring only: settings -> logging -> CoreApp.run(). No logic lives here;
# an entry point should be too boring to contain bugs.
#
# Order note: logging is installed with a safe default level FIRST, so a
# broken .env still fails as a structured log line instead of a bare
# traceback. Once settings load, the level is corrected to the configured
# one.
# =============================================================================

from __future__ import annotations

import asyncio
import logging
import sys

from jarvis.common.log import get_logger, setup_logging
from jarvis.common.settings import CoreSettings


def main() -> None:
    setup_logging("INFO")
    log = get_logger("core.main")

    try:
        settings = CoreSettings()
    except Exception:
        log.critical("configuration invalid - refusing to start", exc_info=True)
        sys.exit(1)

    logging.getLogger().setLevel(settings.log_level)

    from jarvis.core.app import CoreApp  # import after logging is live
    asyncio.run(CoreApp(settings).run())


if __name__ == "__main__":
    main()