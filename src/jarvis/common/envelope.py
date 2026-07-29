# =============================================================================
# src/jarvis/common/envelope.py — the wire format (doc 02 §2)
# =============================================================================
#
# Every WebSocket message in the system — Core⇄worker, Core⇄client, both
# directions — is an Envelope. There is no second wire format.
#
# TWO-STAGE VALIDATION (the central design idea):
#
#   Stage 1: Envelope.model_validate(raw)      — outer frame only.
#            Succeeds even if `kind` is unknown or `payload` is nonsense.
#            This is what lets a receiver answer "error.unsupported_kind"
#            with the message id, instead of crashing (doc 02 rule).
#
#   Stage 2: envelope.parse_payload()          — inner payload, validated
#            against the Pydantic model registered for this `kind`.
#            Raises cleanly if the kind is unregistered or payload invalid.
#
# THE KIND REGISTRY:
# kind string (e.g. "client.user_message") → payload model class.
# Modules that own protocol kinds register them at import time. Exactly one
# model per kind; double registration is a programming error and fails loud.
# The registry also serves as machine-readable protocol documentation:
# `registered_kinds()` lists everything this build can speak.
#
# Versioning: kinds are versioned by the envelope's `v` field collectively,
# not individually (doc 02). v=1 until a breaking protocol change.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.common.ids import is_ulid, new_ulid, utc_now

PROTOCOL_VERSION = 1

# ── Kind registry ────────────────────────────────────────────────────────────

_KIND_REGISTRY: dict[str, type[BaseModel]] = {}


class UnknownKind(Exception):
    """Raised by parse_payload() for a kind this build does not speak.
    Receivers catch this and reply error.unsupported_kind — never crash."""


def register_kind(kind: str, payload_model: type[BaseModel]) -> None:
    """Bind a kind string to its payload model. Called at import time by
    the module that owns the kind (gateway kinds in Phase 1, worker kinds
    in Phase 2). Double registration = two modules claiming the same kind
    = a bug we refuse to paper over."""
    if kind in _KIND_REGISTRY:
        raise RuntimeError(
            f"envelope kind {kind!r} registered twice "
            f"(existing: {_KIND_REGISTRY[kind].__name__}, "
            f"new: {payload_model.__name__})"
        )
    _KIND_REGISTRY[kind] = payload_model


def registered_kinds() -> tuple[str, ...]:
    """Every kind this build can validate — the protocol surface, listable."""
    return tuple(sorted(_KIND_REGISTRY))


# ── The Envelope itself ──────────────────────────────────────────────────────

class Envelope(BaseModel):
    """The outer frame of every wire message (doc 02 §2, field-exact)."""

    # forbid: an envelope with unexpected top-level fields is a malformed or
    # hostile message, not something to silently accept.
    model_config = ConfigDict(extra="forbid")

    v: int = PROTOCOL_VERSION
    id: str = Field(default_factory=new_ulid)
    ts: datetime = Field(default_factory=utc_now)
    kind: str
    trace_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _id_is_ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError(f"envelope id is not a ULID: {v!r}")
        return v

    @field_validator("ts")
    @classmethod
    def _ts_is_utc(cls, v: datetime) -> datetime:
        # Naive timestamps are ambiguous; reject at the boundary. Aware
        # non-UTC timestamps are normalised — storage is UTC (doc 02).
        if v.tzinfo is None:
            raise ValueError("envelope ts must be timezone-aware (UTC)")
        return v.astimezone(timezone.utc)

    @field_validator("kind")
    @classmethod
    def _kind_shape(cls, v: str) -> str:
        # Kinds are dotted lowercase paths: "client.user_message",
        # "worker.job_progress". Shape-checked here; existence is checked
        # at parse_payload time (stage 2), not here (stage 1 must accept
        # unknown kinds).
        parts = v.split(".")
        if len(parts) < 2 or not all(
            p and all(c.islower() or c.isdigit() or c == "_" for c in p)
            for p in parts
        ):
            raise ValueError(
                f"kind must be dotted lowercase (e.g. 'client.user_message'), "
                f"got {v!r}"
            )
        return v

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def parse_payload(self) -> BaseModel:
        """Validate payload against the model registered for this kind.

        Raises:
            UnknownKind          — kind not in this build's registry.
            pydantic.ValidationError — payload doesn't match the kind's model.
        Both are expected, catchable conditions at the receiving edge.
        """
        model = _KIND_REGISTRY.get(self.kind)
        if model is None:
            raise UnknownKind(self.kind)
        return model.model_validate(self.payload)


def make_envelope(
    kind: str,
    payload: BaseModel,
    trace_id: str | None = None,
) -> Envelope:
    """Build an outgoing envelope from a typed payload.

    The sending side of stage-2 discipline: you can only make an envelope
    from a payload model registered for that kind — catching 'right data,
    wrong kind string' mistakes at send time, in the sender's stack trace,
    rather than at the receiver where debugging is twice removed.
    """
    expected = _KIND_REGISTRY.get(kind)
    if expected is None:
        raise UnknownKind(kind)
    if not isinstance(payload, expected):
        raise TypeError(
            f"kind {kind!r} expects payload {expected.__name__}, "
            f"got {type(payload).__name__}"
        )
    return Envelope(
        kind=kind,
        trace_id=trace_id,
        payload=payload.model_dump(mode="json"),
    )