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
    """Configuration shared by every process role (Core, worker, tools).

    Role-specific settings classes inherit from this; each process
    instantiates exactly one settings object at startup and passes it down
    explicitly (no global singleton - explicit wiring keeps tests honest).
    """

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",     # JARVIS_DATA_DIR -> data_dir, etc.
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",           # unrelated env vars in .env are not errors
    )

    # Root of ALL persistent state for this deployment.
    data_dir: Path = Path("./data")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Owner's IANA timezone. Storage is UTC everywhere; this is for
    # owner-facing rendering and scheduling ("nightly" means nightly in
    # Jaipur, not nightly in UTC).
    owner_timezone: str = "Asia/Kolkata"

    @field_validator("owner_timezone")
    @classmethod
    def _timezone_must_exist(cls, v: str) -> str:
        # Fail at boot with a clear message, not at 2 a.m. when the first
        # scheduled job tries to resolve an invalid zone name.
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

    # -- LLM provider (phase 1) ----------------------------------------------

    # Optional AT BOOT on purpose: a missing key must stop model calls with
    # a clear error, never stop the daemon itself. The Core has duties
    # (schedules, storage, status) that owe nothing to any API.
    anthropic_api_key: SecretStr | None = None

    # Tier -> model mapping. Code asks for a tier by name; these decide
    # what each tier currently means. Changing models is a config edit
    # plus restart, never a code change.
    model_reasoner: str = "claude-sonnet-4-6"
    model_utility: str = "claude-haiku-4-5"

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