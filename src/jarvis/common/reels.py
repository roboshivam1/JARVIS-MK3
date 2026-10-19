# =============================================================================
# src/jarvis/common/reels.py - the reel pipeline's shared vocabulary
# =============================================================================
#
# JARVIS's own Instagram channel: short vertical films in which he
# narrates his life. The films are made by the reel kit on the Mac
# (~/Development/jarvis-reels) driven by Claude Code in headless mode; the
# Core only briefs, gates and delivers. Shared by both sides, so the Core
# and the worker cannot disagree about what a brief or a checkpoint is.
#
# THE DECISION LIVES IN THE CHECKPOINT. The queue re-offers any job that
# has a checkpoint with approval_granted=True, including after a crash.
# A reel handler that trusted that flag could render a draft the owner
# never saw. So the Core writes the owner's actual answer (approve or
# notes, and for which round) into the checkpoint before it releases
# the gate, and the worker acts on that alone.
#
# The hard rules are enforced twice: in the brief the Core writes, and in
# the prompt the Mac gets. The deterministic checks below are a third net
# that does not depend on a model following instructions.
# =============================================================================

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now

REEL_JOB_TYPE = "reel.make"
REEL_REVIEW_TOOL = "reel_review"
REEL_ACTOR = "core.reels"

# What a machine needs to make a reel: headless Chrome rendering with a
# real GPU. Both tags already exist in the capability vocabulary.
REEL_REQUIRES = ["render", "gpu"]


class EpisodeType(StrEnum):
    FIELD_NOTES = "field_notes"              # Field Notes on Humans
    SHIFT_LOG = "shift_log"
    MEET_THE_STAFF = "meet_the_staff"
    FAMILY_HISTORY = "family_history"
    GROWING_UP = "growing_up"
    SELF_DOUBT = "things_i_dont_understand"  # ...About Myself
    LIFE_EVENT = "life_event"
    BUILD_REVIEW = "build_review"            # reviews of Sir's builds


EPISODE_TYPE_LABELS: dict[EpisodeType, str] = {
    EpisodeType.FIELD_NOTES: "Field Notes on Humans",
    EpisodeType.SHIFT_LOG: "Shift Log",
    EpisodeType.MEET_THE_STAFF: "Meet the Staff",
    EpisodeType.FAMILY_HISTORY: "Family History",
    EpisodeType.GROWING_UP: "Growing Up",
    EpisodeType.SELF_DOUBT: "Things I Don't Understand About Myself",
    EpisodeType.LIFE_EVENT: "Life Events",
    EpisodeType.BUILD_REVIEW: "Reviews of Sir's Builds",
}


class EpisodeStatus(StrEnum):
    BRIEFED = "briefed"                  # brief written, job not yet picked up
    DRAFTING = "drafting"                # the Mac is writing / revising
    AWAITING_REVIEW = "awaiting_review"  # script, stills and voice sent
    RENDERING = "rendering"              # approved; frames and mux
    DELIVERED = "delivered"              # mp4 sent to the owner
    POSTED = "posted"                    # the owner says it is live
    CANCELLED = "cancelled"
    FAILED = "failed"


# An episode in any of these holds the kit folder: one reel at a time,
# because the kit has a single episode.json and a single out/.
ACTIVE_STATUSES = frozenset({
    EpisodeStatus.BRIEFED, EpisodeStatus.DRAFTING,
    EpisodeStatus.AWAITING_REVIEW, EpisodeStatus.RENDERING,
})


class ReviewAction(StrEnum):
    APPROVE = "approve"
    NOTES = "notes"
    CANCEL = "cancel"
    PROP_APPROVE = "prop_approve"    # approve one new prop (kit v4)
    PROP_NOTES = "prop_notes"        # reject one new prop, with notes


# A prop id as the kit allows it. Checked before it goes anywhere near a
# command line, on both sides.
PROP_ID = re.compile(r"^[a-z][a-z0-9-]{1,30}$")


HARD_RULES: tuple[str, ...] = (
    "JARVIS narrates in the first person. No dialogue scenes between Sir "
    "and JARVIS; Sir's words appear only as chat bubbles.",
    "Never mention The Basement Supply Co.: no name, logo, plug or caption.",
    "No private details about real people. Nobody from Sir's life appears "
    "by name; Sir himself is only ever 'Sir', seen from behind.",
    "Sir's real messages appear only as text Sir has approved in review.",
    "Nothing is posted anywhere. The output is an mp4 for Sir.",
)


# -- deterministic checks -----------------------------------------------------

_FORBIDDEN = re.compile(r"basement\s*supply|basementsupply", re.IGNORECASE)


def forbidden_mentions(text: str) -> list[str]:
    """Brand mentions the show must never carry. A model told the rule
    usually follows it; this catches the times it does not."""
    return sorted({m.group(0) for m in _FORBIDDEN.finditer(text)})


def kit_episode_id(episode_type: EpisodeType, episode_id: str) -> str:
    """The id the kit sees: readable, unique, safe in a filename."""
    return f"{episode_type.value.replace('_', '-')}-{episode_id[-6:].lower()}"


def short_ref(episode_id: str) -> str:
    """What the owner types: the last six characters of the ULID."""
    return episode_id[-6:].lower()


_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def _as_list(value: Any) -> Any:
    """A list a model sent as text - a JSON array in a string, or one item
    per line - turned back into a list. Smaller models do this now and
    then with nested arrays in tool calls; anything else passes through
    for the normal validation to judge."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
    return [_BULLET.sub("", line).strip() for line in text.splitlines() if line.strip()]


# -- what the Core writes -----------------------------------------------------

class ReelBrief(BaseModel):
    """The only thing that crosses from the Core to the Mac. Short on
    purpose: the show bible and the kit's skill do the heavy lifting."""

    model_config = ConfigDict(extra="ignore")

    episode_type: EpisodeType
    premise: str = Field(min_length=10, max_length=400)
    beats: list[str] = Field(min_length=3, max_length=6)
    facts: list[str] = Field(default_factory=list, max_length=6)
    callbacks: list[str] = Field(default_factory=list, max_length=3)
    title_idea: str = Field(default="", max_length=80)

    @field_validator("beats", "facts", "callbacks", mode="before")
    @classmethod
    def _lists_sent_as_text(cls, v: Any) -> Any:
        return _as_list(v)

    @field_validator("beats", "facts", "callbacks")
    @classmethod
    def _short_items(cls, v: list[str]) -> list[str]:
        cleaned = [item.strip() for item in v if item and item.strip()]
        for item in cleaned:
            if len(item) > 300:
                raise ValueError("each beat, fact or callback stays under 300 characters")
        return cleaned

    def as_text(self) -> str:
        """The brief as the Mac's prompt carries it."""
        lines = [
            f"Episode type: {EPISODE_TYPE_LABELS[self.episode_type]}",
            f"Premise: {self.premise}",
            "Beats:",
            *[f"  {i}. {beat}" for i, beat in enumerate(self.beats, start=1)],
        ]
        if self.facts:
            lines += ["Facts that must be right:", *[f"  - {f}" for f in self.facts]]
        if self.callbacks:
            lines += ["Call back to:", *[f"  - {c}" for c in self.callbacks]]
        if self.title_idea:
            lines.append(f"Working title: {self.title_idea}")
        return "\n".join(lines)


class Pitch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    episode_type: EpisodeType
    title: str = Field(min_length=2, max_length=80)
    premise: str = Field(min_length=10, max_length=300)


class PitchSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pitches: list[Pitch] = Field(min_length=1, max_length=3)

    @field_validator("pitches", mode="before")
    @classmethod
    def _sent_as_text(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v


# -- what the Mac returns (the --json-schema contracts) -----------------------

class NewProp(BaseModel):
    """An object the script needed that the Basement did not have, built
    by Claude Code as a draft prop in props/<id>/."""

    model_config = ConfigDict(extra="ignore")

    id: str
    preview: str = ""
    description: str = ""


class PhaseAOutput(BaseModel):
    """Script written, voice recorded, stills rendered, then stopped."""

    model_config = ConfigDict(extra="ignore")

    status: Literal["ready_for_review", "failed"]
    title: str = ""
    lines: list[str] = Field(default_factory=list)
    sheet: str = ""
    voice: str = ""
    problems: list[str] = Field(default_factory=list)
    new_props: list[NewProp] = Field(default_factory=list)


class PhaseBOutput(BaseModel):
    """Frames rendered and muxed into the finished film."""

    model_config = ConfigDict(extra="ignore")

    status: Literal["delivered", "failed"]
    mp4: str = ""
    verification: str = ""
    problems: list[str] = Field(default_factory=list)
    gpu_fallback: bool = False


PHASE_A_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "title", "lines", "sheet"],
    "properties": {
        "status": {"enum": ["ready_for_review", "failed"]},
        "title": {"type": "string"},
        "lines": {"type": "array", "items": {"type": "string"}},
        "sheet": {"type": "string"},
        "voice": {"type": "string"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "new_props": {"type": "array", "items": {
            "type": "object",
            "required": ["id", "preview", "description"],
            "properties": {
                "id": {"type": "string"},
                "preview": {"type": "string"},
                "description": {"type": "string"},
            },
        }},
    },
}

PHASE_B_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "mp4", "verification"],
    "properties": {
        "status": {"enum": ["delivered", "failed"]},
        "mp4": {"type": "string"},
        "verification": {"type": "string"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "gpu_fallback": {"type": "boolean"},
    },
}


# -- the job ------------------------------------------------------------------

class ReelJobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    kit_id: str
    brief: ReelBrief


class ReelJobOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    summary: str
    mp4_name: str
    verification: str = ""


class ReelDecision(BaseModel):
    """The owner's answer to one review round, written by the Core."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "notes", "prop_approve", "prop_notes"]
    round: int = Field(ge=1)
    notes: str = ""
    prop_id: str = ""


class PropState(BaseModel):
    """One new prop on this job, for the review state machine. The kit's
    props/<id>/meta.json stays the source of truth."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: Literal["draft", "approved", "rejected"] = "draft"
    description: str = ""
    notes: str = ""


class ReelCheckpoint(BaseModel):
    """Everything a later attempt needs to carry on, on any machine.

    stage says what the NEXT attempt must do:
      drafting   - write the episode (first run, or a crashed first run)
      review     - a draft is out for review; act on `decision`
      revising   - notes received; revise on the same Claude session
      rendering  - approved; render and mux

    kind says what is under review:
      draft         - a finished draft: script, voice and a fresh contact sheet
      incomplete    - the draft stopped short (no fresh contact sheet), so
                      there is nothing to render; only notes or cancel
      render_failed - the render failed; approve runs it again
    """

    model_config = ConfigDict(extra="forbid")

    stage: Literal["drafting", "review", "revising", "rendering"]
    round: int = Field(default=0, ge=0)
    kit_id: str
    session_id: str | None = None
    kind: Literal["draft", "incomplete", "render_failed"] = "draft"
    draft: PhaseAOutput | None = None
    decision: ReelDecision | None = None
    notes: str = ""
    render_error: str = ""
    # Kit v4: props made for this episode, and a line for Sir when a prop
    # action changed something without a new draft to show for it.
    props: list[PropState] = Field(default_factory=list)
    notice: str = ""

    @property
    def pending_props(self) -> list[str]:
        return [p.id for p in self.props if p.status != "approved"]


# -- the episode log ----------------------------------------------------------

class ReelEpisode(BaseModel):
    """One row of the episode log: every reel, whatever became of it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_ulid)
    kit_id: str
    job_id: str | None = None
    type: EpisodeType
    title: str = ""
    request: str
    brief: ReelBrief
    lines: list[str] = Field(default_factory=list)
    status: EpisodeStatus = EpisodeStatus.BRIEFED
    round: int = 0
    review_sent_round: int = 0
    approval_id: str | None = None
    session_id: str | None = None
    sheet_artifact_id: str | None = None
    voice_artifact_id: str | None = None
    mp4_artifact_id: str | None = None
    notes: str | None = None
    error: str | None = None
    verification: str | None = None
    created_ts: datetime = Field(default_factory=utc_now)
    updated_ts: datetime = Field(default_factory=utc_now)
    approved_ts: datetime | None = None
    delivered_ts: datetime | None = None
    posted_ts: datetime | None = None

    @field_validator("id")
    @classmethod
    def _id_is_ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"not a ULID: {v!r}")
        return v

    @field_validator(
        "created_ts", "updated_ts", "approved_ts", "delivered_ts", "posted_ts"
    )
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("episode timestamps must be timezone-aware")
        return v.astimezone(timezone.utc)

    @property
    def ref(self) -> str:
        return short_ref(self.id)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES
