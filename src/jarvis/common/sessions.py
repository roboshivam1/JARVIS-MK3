# =============================================================================
# src/jarvis/common/sessions.py - Session and Turn models
# =============================================================================
#
# A Session is one conversation thread (a chat window). A Turn is one
# message within it, from the owner or from JARVIS.
#
# Unlike events, sessions and turns are LIVING STATE, not history: the
# rolling summary is rewritten, last_active_ts ticks forward, sessions
# get archived. So these models are not frozen and their tables carry no
# append-only triggers.
#
# rolling_summary is schema-ready but feature-empty for now: the column
# and field exist so context assembly can read them, but nothing writes a
# summary until the memory phase. Empty string means "no summary yet".
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now


class SessionStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class TurnRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Session(BaseModel):
    """One conversation thread."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    client_kind: str                      # "telegram", "web", "voice.mac", ...
    title: str | None = None              # auto-generated later
    status: SessionStatus = SessionStatus.ACTIVE
    rolling_summary: str = ""             # empty until the memory phase
    created_ts: datetime = Field(default_factory=utc_now)
    last_active_ts: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def _id_is_ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("client_kind")
    @classmethod
    def _client_kind_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("client_kind must be non-empty")
        return v

    @field_validator("created_ts", "last_active_ts")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("session timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)


class Turn(BaseModel):
    """One message within a session."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    session_id: str
    role: TurnRole
    content: str                          # voice turns store the transcript
    attachments: list[str] = Field(default_factory=list)   # artifact ids
    job_refs: list[str] = Field(default_factory=list)      # jobs spawned/reported
    llm_call_ids: list[str] = Field(default_factory=list)  # trace linkage
    ts: datetime = Field(default_factory=utc_now)

    @field_validator("id", "session_id")
    @classmethod
    def _ids_are_ulids(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("attachments", "job_refs", "llm_call_ids")
    @classmethod
    def _ref_lists_are_ulids(cls, v: list[str]) -> list[str]:
        for ref in v:
            if not is_ulid(ref):
                raise ValueError(f"reference is not a ULID: {ref!r}")
        return v

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("turn ts must be timezone-aware")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _user_turns_make_no_llm_calls(self) -> "Turn":
        # Model calls happen while PRODUCING a reply; a user turn is input.
        # A user turn claiming llm calls means wiring got crossed somewhere.
        if self.role is TurnRole.USER and self.llm_call_ids:
            raise ValueError("a user turn cannot reference llm calls")
        return self