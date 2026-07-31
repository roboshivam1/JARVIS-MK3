# =============================================================================
# src/jarvis/common/settings.py - typed configuration for every process role
# =============================================================================
#
# All configuration enters the system through this module, from environment
# variables / .env, prefixed JARVIS_ (see .env.example for the reference).
#
# Design constraints:
#   - Validation at the boundary: a bad or missing value stops boot with a
#     precise error. No component ever sees unvalidated config.
#   - common/ has no I/O side effects: importing this module must not touch
#     the filesystem. Directory creation is an explicit method the entry
#     point calls during boot.
#   - Derived paths (db, artifacts) are properties, not fields: the layout
#     of the data directory is a contract that backups depend on, not a
#     per-deployment choice.
#   - Secrets are SecretStr: printing or logging settings can never leak
#     them; code must explicitly call .get_secret_value() at the one place
#     the secret is used.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CommonSettings(BaseSettings):
    """Configuration shared by every process role (Core, worker, tools)."""

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Root of ALL persistent state for this deployment.
    data_dir: Path = Path("./data")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Owner's IANA timezone. Storage is UTC everywhere; this is for
    # owner-facing rendering and scheduling.
    owner_timezone: str = "Asia/Kolkata"

    @field_validator("owner_timezone")
    @classmethod
    def _timezone_must_exist(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except Exception:
            raise ValueError(
                f"JARVIS_OWNER_TIMEZONE={v!r} is not a valid IANA timezone "
                f"(expected something like 'Asia/Kolkata')"
            )
        return v

    @property
    def tz(self) -> ZoneInfo:
        """The owner's timezone as a usable object."""
        return ZoneInfo(self.owner_timezone)


class CoreSettings(CommonSettings):
    """Settings for the Core daemon (`python -m jarvis.core`)."""

    # -- LLM provider --------------------------------------------------------

    # Optional AT BOOT on purpose: a missing key must stop model calls with
    # a clear error, never stop the daemon itself.
    anthropic_api_key: SecretStr | None = None

    # Tier -> model mapping; changing models is a config edit + restart.
    model_reasoner: str = "claude-sonnet-4-6"
    model_utility: str = "claude-haiku-4-5"

    # -- Embeddings ----------------------------------------------------------

    # Which provider computes embeddings. "none" (or a misconfigured
    # provider) means memory falls back to keyword-only rather than
    # failing - degraded memory beats no daemon.
    embedder_provider: Literal["ollama", "voyage", "none"] = "ollama"

    # Interpreted by the selected provider: an Ollama model name, or a
    # Voyage model name.
    model_embedder: str = "nomic-embed-text"

    # Local provider: where the Ollama server listens.
    ollama_base_url: str = "http://localhost:11434"

    # When nightly memory maintenance runs, as a cron expression in the
    # owner's timezone. Default 03:30 - late enough to catch the whole
    # day, early enough to be done before morning.
    sleep_cycle_cron: str = "30 3 * * *"

    # Hosted provider: only needed when embedder_provider is "voyage".
    voyage_api_key: SecretStr | None = None

    # -- Gateway -------------------------------------------------------------

    gateway_host: str = "127.0.0.1"
    gateway_port: int = 8321

    # Shared bearer secret for all authed gateway routes. Empty means
    # "refuse all authed requests" - a forgotten token fails CLOSED, never
    # open.
    gateway_token: SecretStr = SecretStr("")

    # -- Telegram ------------------------------------------------------------

    # Empty token = bridge does not start; Core runs without it.
    telegram_bot_token: SecretStr = SecretStr("")

    # The ONLY Telegram user the bot answers. 0 = unset; the bridge
    # refuses to start with a token but no owner (an open bot is worse
    # than no bot).
    telegram_owner_id: int = 0

    # -- Derived layout - properties, deliberately not fields ----------------

    @property
    def db_path(self) -> Path:
        return self.data_dir / "core.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    def ensure_data_dirs(self) -> None:
        """Create the on-disk layout if absent. Called explicitly by the
        Core's boot sequence - never as an import side effect."""
        for d in (self.data_dir, self.artifacts_dir, self.backups_dir):
            d.mkdir(parents=True, exist_ok=True)