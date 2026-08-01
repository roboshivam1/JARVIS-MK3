# =============================================================================
# src/jarvis/worker/settings.py - worker configuration
# =============================================================================
#
# A worker's whole identity in config: who it is, where the Core lives,
# what it can do, and how much at once.
#
# Same JARVIS_ prefix and .env mechanism as the Core, because on the
# laptop today both processes read the same file. When the Core moves to
# the VPS, only core_url changes here and nothing else moves.
#
# capabilities is stored as a STRING and exposed as a list. Declaring it
# as list[str] would make pydantic-settings demand a JSON array in .env -
# and it parses that at the SOURCE level, before any validator of ours
# can intervene. A plain string with a property is less clever and
# actually works, which is the better trade in configuration code.
# =============================================================================

from __future__ import annotations

from pydantic import SecretStr

from jarvis.common.settings import CommonSettings


class WorkerSettings(CommonSettings):
    """Settings for a worker process (`python -m jarvis.worker`)."""

    # This machine's stable name. Reconnecting with the same id replaces
    # the old registration, which is what makes a dropped link heal
    # cleanly rather than leaving a ghost in the fleet.
    worker_id: str = "macbook"

    # Where the Core listens. ws:// locally; wss:// once there is a VPS
    # with TLS in front of it.
    core_url: str = "ws://127.0.0.1:8321/ws/worker"

    # The same shared secret the Core's gateway uses.
    worker_token: SecretStr = SecretStr("")

    # What this machine can do, comma-separated. A claim about the
    # environment, not a detection: advertise "browser" only if a
    # browser is genuinely installed. A wrong claim shows up as jobs
    # that fail on arrival, which is the honest failure mode.
    capabilities_raw: str = "macos"

    # How many jobs to run at once. Two is a sane default for a laptop
    # that a human is also using.
    max_concurrency: int = 2

    @property
    def capabilities(self) -> list[str]:
        """The capability tags this worker advertises."""
        return [tag.strip() for tag in self.capabilities_raw.split(",") if tag.strip()]