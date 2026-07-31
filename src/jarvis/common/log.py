# =============================================================================
# src/jarvis/common/log.py - structured JSON-lines logging
# =============================================================================
#
# Every process (Core, workers) logs one JSON object per line to stderr.
# Rationale: logs are data to be queried (grep + jq, or shipped elsewhere
# later), not prose. Unstructured print() is banned in this codebase; the
# logger is always available and always structured.
#
# We build ON stdlib logging instead of replacing it so that third-party
# libraries (aiosqlite, fastapi, aiogram) inherit the same formatting when
# they log - hook the root handler once, everything becomes structured.
#
# Usage:
#     from jarvis.common.log import get_logger
#     log = get_logger("core.db")
#     log.info("migration applied", extra={"version": 1})
#
# Fields from `extra` become top-level JSON keys. Standard keys on every
# line: ts (UTC ISO-8601), level, logger, msg. Exceptions add exc_type,
# exc_msg, exc_traceback.
#
# setup_logging() is called exactly once by the process entry point.
# Importing this module has no side effects - same discipline as settings.
# =============================================================================

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, MutableMapping

# Attribute names present on a blank LogRecord: stdlib plumbing, not user
# data. Computed once by making a throwaway record and reading its dict.
# Anything on a real record NOT in this set arrived via `extra` and is ours.
_STDLIB_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0,
        msg="", args=(), exc_info=None,
    ).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Formats one LogRecord as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Merge caller-supplied extra fields. Ours win over nothing; if a
        # caller shadows a standard key (e.g. passes ts) their value is
        # kept under a prefixed name so no information is silently lost.
        for key, value in record.__dict__.items():
            if key in _STDLIB_ATTRS:
                continue
            if key in line:
                line[f"extra_{key}"] = value
            else:
                line[key] = value

        if record.exc_info and record.exc_info[0] is not None:
            etype, evalue, etb = record.exc_info
            line["exc_type"] = etype.__name__
            line["exc_msg"] = str(evalue)
            line["exc_traceback"] = "".join(
                traceback.format_exception(etype, evalue, etb)
            )

        # default=str: never let an unserialisable object (Path, datetime,
        # a Pydantic model passed by accident) crash the logging pipeline.
        # A slightly lossy log line beats an exception inside the logger.
        return json.dumps(line, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Install JSON-lines logging on the root logger. Call once per process,
    from the entry point, before any real work.

    Also routes Python warnings and uncaught exceptions through the same
    structured stream, so no text ever bypasses it.
    """
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()          # idempotent: rerunning replaces, not stacks
    root.addHandler(handler)
    root.setLevel(level)

    logging.captureWarnings(True)

    def _log_uncaught(etype: type[BaseException], value: BaseException, tb: Any) -> None:
        if issubclass(etype, KeyboardInterrupt):
            # Ctrl-C is a normal way to stop a dev run, not a crash report.
            sys.__excepthook__(etype, value, tb)
            return
        logging.getLogger("uncaught").critical(
            "uncaught exception", exc_info=(etype, value, tb)
        )

    sys.excepthook = _log_uncaught


class _SafeLogger(logging.LoggerAdapter[logging.Logger]):
    """A logger that cannot be killed by its own field names.

    stdlib logging reserves attribute names on every record (name, msg,
    module, created, process, ...) and RAISES if an `extra` dict tries to
    use one. That turns a harmless logging mistake into a crash in
    whatever real work was being done at the time - which is exactly
    backwards: logging exists to observe work, never to endanger it.

    Colliding keys are renamed to field_<key> rather than dropped, so no
    information is lost and the collision is visible in the output.
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {
                (f"field_{k}" if k in _STDLIB_ATTRS else k): v
                for k, v in extra.items()
            }
        return msg, kwargs


def get_logger(name: str) -> _SafeLogger:
    """Get a named logger. Names are dotted component paths by convention:
    core.app, core.db, core.gateway, worker.runner."""
    return _SafeLogger(logging.getLogger(name), {})