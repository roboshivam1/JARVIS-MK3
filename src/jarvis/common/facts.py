# =============================================================================
# src/jarvis/common/facts.py - the Fact model
# =============================================================================
#
# A fact is ONE self-contained sentence the system believes about the
# owner's world. Not a conversation, not a summary - a single claim that
# stands alone when read cold months later.
#
# Facts are never deleted, only superseded (replaced by a newer belief)
# or expired (unused and unimportant, faded out). Deletion would destroy
# the evidence trail; status changes keep it. source_event_ids points
# back into the event log, so any belief can be traced to the moment it
# was formed.
#
# The embedding vector is deliberately NOT a field here: it is large,
# rarely needed by callers, and purely a storage/retrieval concern. The
# repository loads it only when searching.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now


class FactCategory(StrEnum):
    """What kind of thing this fact is about. Extending the vocabulary is
    a one-line addition here plus a doc amendment."""

    PREFERENCE = "preference"          # how the owner likes things done
    PERSON = "person"                  # someone in his life
    PROJECT = "project"                # something he is building
    ROUTINE = "routine"                # recurring habits and schedules
    CREDENTIAL_REF = "credential-ref"  # WHERE a credential lives, never the secret
    WORLD = "world"                    # facts about his context, not about him
    OTHER = "other"


class FactStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"   # replaced by a newer, conflicting fact
    EXPIRED = "expired"         # faded out through disuse


class Fact(BaseModel):
    """One durable belief about the owner's world."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    text: str = Field(min_length=3)
    category: FactCategory = FactCategory.OTHER

    # How much this matters (0 trivial .. 1 defining). Drives what
    # graduates into the profile document and what survives decay.
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # How sure the system is. Contradictions lower it; disuse decays it.
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    status: FactStatus = FactStatus.ACTIVE
    supersedes: str | None = None            # the fact this one replaced
    source_event_ids: list[str] = Field(default_factory=list)   # provenance

    created_ts: datetime = Field(default_factory=utc_now)
    last_accessed_ts: datetime = Field(default_factory=utc_now)
    access_count: int = Field(default=0, ge=0)

    # Which embedding model produced this fact's vector. A model change
    # makes old vectors incomparable, so the sleep cycle re-embeds
    # anything whose version is stale.
    embedder_version: str | None = None

    @field_validator("id", "supersedes")
    @classmethod
    def _ids_are_ulids(cls, v: str | None) -> str | None:
        if v is not None and not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator("source_event_ids")
    @classmethod
    def _sources_are_ulids(cls, v: list[str]) -> list[str]:
        for event_id in v:
            if not is_ulid(event_id):
                raise ValueError(f"source event id is not a ULID: {event_id!r}")
        return v

    @field_validator("text")
    @classmethod
    def _text_is_one_claim(cls, v: str) -> str:
        # One fact, one line. Newlines mean several claims crammed
        # together, which breaks dedup and makes retrieval imprecise.
        text = v.strip()
        if "\n" in text:
            raise ValueError("a fact is one self-contained sentence, not a paragraph")
        return text

    @field_validator("created_ts", "last_accessed_ts")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("fact timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)