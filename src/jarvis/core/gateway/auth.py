# =============================================================================
# src/jarvis/core/gateway/auth.py - one door policy for the gateway
# =============================================================================
#
# Single-user auth: every authed request must present the shared bearer
# secret. No accounts, no sessions, no signup - "is this the owner or an
# owner-authorised device" is the entire question.
#
# Two properties worth knowing:
#   - Fails CLOSED: an empty configured token rejects everything, so a
#     half-configured deployment is locked, not open.
#   - Timing-safe comparison: ordinary == quits at the first wrong
#     character, so response TIME leaks how much of a guess was right.
#     secrets.compare_digest takes constant time regardless.
#
# Per-device tokens (the devices table) arrive with workers, when there
# is more than one device story to tell. Flagged as a deliberate phase-1
# simplification.
# =============================================================================

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request

from jarvis.common.log import get_logger

log = get_logger("core.gateway.auth")


class BearerAuth:
    """Callable FastAPI dependency: checks the Authorization header on
    every route that includes it."""

    def __init__(self, expected_token: str) -> None:
        self._expected = expected_token

    async def __call__(self, request: Request) -> None:
        if not self._expected:
            # No token configured: refuse authed routes entirely.
            log.warning("authed request refused - no gateway token configured")
            raise HTTPException(status_code=401, detail="gateway not configured")

        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="missing bearer token")

        if not secrets.compare_digest(token, self._expected):
            log.warning("bad gateway token", extra={"path": request.url.path})
            raise HTTPException(status_code=401, detail="invalid token")