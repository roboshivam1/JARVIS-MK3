# =============================================================================
# src/jarvis/common/ids.py — ULID generation and inspection
# =============================================================================
#
# ULID = Universally Unique Lexicographically Sortable Identifier.
# Layout (128 bits total, encoded as 26 chars of Crockford base32):
#
#     48 bits: Unix time in MILLISECONDS   ── makes IDs time-ordered
#     80 bits: randomness                  ── makes IDs unique
#
# Constraint this module exists to uphold (doc 02): everything in the system
# is identified by a ULID string, and sorting those strings sorts records
# chronologically. The event log and job queue rely on that property.
#
# Monotonicity: two IDs minted in the same millisecond must still sort in
# creation order. We follow the ULID spec's approach — same millisecond →
# previous random value + 1 instead of fresh randomness.
#
# Hand-rolled (no dependency) by owner-approved proposal: the spec is tiny,
# stable, and worth understanding, since these are the system's primary keys.
# =============================================================================

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone

# Crockford base32: no I, L, O, U — avoids lookalike characters and accidental
# profanity. This exact alphabet is required by the ULID spec; using standard
# base32 would produce IDs other ULID tooling cannot read.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

# Decode table: char → 5-bit value. Built once at import.
_DECODE = {c: i for i, c in enumerate(_ALPHABET)}

_TIMESTAMP_BITS = 48
_RANDOM_BITS = 80
_MAX_TIMESTAMP = (1 << _TIMESTAMP_BITS) - 1   # year ~10889 — not our problem
_MAX_RANDOM = (1 << _RANDOM_BITS) - 1


def _encode(value: int) -> str:
    """Encode a 128-bit int as exactly 26 Crockford base32 characters.

    26 chars x 5 bits = 130 bits of capacity for 128 bits of data; the top
    2 bits are always zero, which is why every ULID starts with 0–7.
    Fixed width is essential: variable-length encoding would break
    lexicographic ordering.
    """
    chars = []
    for _ in range(26):
        chars.append(_ALPHABET[value & 0b11111])   # take the low 5 bits
        value >>= 5
    return "".join(reversed(chars))                # most significant char first


class _MonotonicGenerator:
    """Mints ULIDs that are strictly increasing, even within one millisecond.

    State: the (timestamp, random) pair of the last ID issued. If the clock
    still reads the same millisecond, we reuse that random value + 1 rather
    than rolling fresh randomness — guaranteeing sort order matches creation
    order. Fresh milliseconds get fresh randomness.

    The lock makes this safe if a worker subprocess or thread ever imports
    it; on the single-threaded Core it is uncontended and effectively free.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_ts: int = -1
        self._last_rand: int = 0

    def new(self) -> str:
        with self._lock:
            ts = int(time.time() * 1000)

            if ts <= self._last_ts:
                # Same millisecond (or clock went backwards — NTP adjustments
                # happen; we hold the timestamp rather than emit an ID that
                # sorts before its predecessor).
                ts = self._last_ts
                rand = self._last_rand + 1
                if rand > _MAX_RANDOM:
                    # 2^80 IDs in one millisecond — practically unreachable,
                    # but an infinite loop on overflow would be worse than
                    # a rare 1 ms stall.
                    ts += 1
                    rand = int.from_bytes(os.urandom(10), "big")
            else:
                rand = int.from_bytes(os.urandom(10), "big")  # 10 bytes = 80 bits

            self._last_ts = ts
            self._last_rand = rand
            return _encode((ts << _RANDOM_BITS) | rand)


_generator = _MonotonicGenerator()


def new_ulid() -> str:
    """Mint a new ULID. The only ID factory in the system — every event,
    job, session, turn, and artifact ID comes from this function."""
    return _generator.new()


def ulid_timestamp(ulid: str) -> datetime:
    """Extract the creation time embedded in a ULID (UTC).

    Useful for debugging and for the event log: the ID itself tells you
    when the record was minted, independent of any ts column.
    """
    if len(ulid) != 26:
        raise ValueError(f"not a ULID (length {len(ulid)}, expected 26): {ulid!r}")
    value = 0
    for char in ulid.upper():
        try:
            value = (value << 5) | _DECODE[char]
        except KeyError:
            raise ValueError(f"invalid ULID character {char!r} in {ulid!r}") from None
    ms = value >> _RANDOM_BITS
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def is_ulid(value: str) -> bool:
    """Cheap shape check (length + alphabet). Used by model validators in
    doc-02 schemas to reject malformed IDs at the boundary."""
    return len(value) == 26 and all(c in _DECODE for c in value.upper())

def utc_now() -> datetime:
    """The system's single definition of 'now': timezone-aware UTC.

    Every model default and every stored timestamp uses this. Python's
    naive `datetime.now()` is banned in this codebase — a naive datetime
    is a bug that hasn't happened yet (it means 'whatever timezone the
    host happens to be in', which differs between your laptop, the VPS,
    and a future worker).
    """
    return datetime.now(timezone.utc)