# =============================================================================
# src/jarvis/core/initiative/notifier.py - the only mouth that speaks
# unprompted
# =============================================================================
#
# THE choke point: nothing else in the system may message the owner
# unprompted. Two scans and one delivery pass, each safe to re-run after
# a crash:
#
#   scan jobs      - finished work belonging to a conversation gets a
#                    pending notification
#   scan approvals - pending gates the owner has not been asked about get
#                    a pending notification carrying the approval id
#   deliver        - pending rows are judged by POLICY, then sent,
#                    deferred, or batched into a digest
#
# Duplicates are impossible by construction: unique indexes on job_id and
# approval_id mean a crash mid-scan cannot produce two copies of the same
# message, and neither scan needs bookkeeping of its own.
#
# Delivered messages are recorded as assistant turns, so JARVIS knows
# what he already told the owner and the owner can reply to it naturally.
#
# DEFERRED IS NOT DROPPED. A notification held during quiet hours keeps
# its pending status and simply is not due yet. Silence must mean
# "nothing happened", never "something happened and I decided not to
# say" - the owner cannot tell those apart, and the uncertainty poisons
# everything else the system tells him.
# =============================================================================

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import utc_now
from jarvis.common.jobs import Job, JobStatus
from jarvis.common.log import get_logger
from jarvis.common.notifications import Notification
from jarvis.common.sessions import Turn, TurnRole
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.initiative.policy import (
    Decision,
    NotificationPolicy,
    compose_digest,
)

log = get_logger("core.initiative.notifier")

SOURCE = "core.initiative"
_POLL_INTERVAL_S = 3.0

# Approvals reach the owner ahead of everything else: work is PAUSED
# until he answers, so waiting until morning costs more than the
# interruption does.
_APPROVAL_PRIORITY = 1


class Deliverer(Protocol):
    """What a client must implement to receive unprompted messages."""

    async def deliver(
        self,
        text: str,
        file_path: Path | None = None,
        file_name: str | None = None,
        approval_id: str | None = None,
    ) -> None: ...


class Notifier:
    """Turns finished work and pending questions into messages that
    actually reach the owner - at a time he will tolerate."""

    def __init__(
        self,
        jobs: JobsRepo,
        sessions: SessionsRepo,
        notifications: NotificationsRepo,
        artifacts: ArtifactsRepo,
        events: EventsRepo,
        approvals: ApprovalsRepo | None = None,
        policy: NotificationPolicy | None = None,
        tz: ZoneInfo | None = None,
    ) -> None:
        self._jobs = jobs
        self._sessions = sessions
        self._notifications = notifications
        self._artifacts = artifacts
        self._events = events
        self._approvals = approvals
        # The policy object is SHARED with the Telegram bridge, which
        # mutates its snooze field via /quiet. Two copies would mean the
        # snooze silently did nothing.
        self._policy = policy or NotificationPolicy()
        self._tz = tz or ZoneInfo("UTC")
        self._deliverers: dict[str, Deliverer] = {}

    def register_deliverer(self, client_kind: str, deliverer: Deliverer) -> None:
        """Attach a client surface. Called at boot for each live client."""
        self._deliverers[client_kind] = deliverer
        log.info("deliverer registered", extra={"client_kind": client_kind})

    # -- the resident loop ----------------------------------------------------

    async def run(self) -> None:
        while True:
            try:
                await self.scan_finished_jobs()
                await self.scan_pending_approvals()
                await self.deliver_pending()
            except Exception:
                # The mouth must outlive its own bad days.
                log.error("notifier cycle failed", exc_info=True)
            await asyncio.sleep(_POLL_INTERVAL_S)

    # -- scan: finished work --------------------------------------------------

    async def scan_finished_jobs(self) -> int:
        """Queue notifications for newly finished conversational jobs."""
        queued = 0
        for job_id in await self._notifications.job_ids_awaiting_notification():
            try:
                job = await self._jobs.get(job_id)
            except Exception:
                # A row that fails model validation - corrupted state
                # left by two writers racing, say - must not wedge the
                # scan. Without this, one bad row loops this subsystem
                # forever and nothing else in the outbox is delivered.
                log.error("skipping unreadable job row", exc_info=True,
                          extra={"job_id": job_id})
                continue

            if job is None or job.session_id is None:
                continue
            session = await self._sessions.get(job.session_id)
            if session is None:
                continue

            text, artifact_id = self._compose(job)
            created = await self._notifications.create(Notification(
                client_kind=session.client_kind,
                text=text,
                session_id=job.session_id,
                job_id=job.id,
                artifact_id=artifact_id,
                priority=3 if job.status is JobStatus.FAILED else 5,
                trace_id=job.trace_id,
            ))
            if created:
                queued += 1
        return queued

    def _compose(self, job: Job) -> tuple[str, str | None]:
        """Message text and any file to send with it."""
        if job.status is JobStatus.FAILED:
            return (
                f"That job failed, sir - {job.type}. Reason: "
                f"{job.error or 'unknown'}."
            ), None

        result = job.result or {}
        summary = result.get("summary")

        # A result may name its artifact explicitly (Core-run jobs mint
        # the id themselves). Worker-run jobs cannot: the Core mints ids
        # on upload, so the handler never learns them. Fall back to the
        # job's own artifact list, which the upload path populates.
        artifact_id = result.get("artifact_id")
        if artifact_id is None and job.artifacts:
            artifact_id = job.artifacts[-1]

        if summary:
            return f"Finished, sir. {summary}", artifact_id
        return f"Job {job.id} finished ({job.type}).", artifact_id

    # -- scan: pending questions ----------------------------------------------

    async def scan_pending_approvals(self) -> int:
        """Queue a question for every gate the owner has not seen yet."""
        if self._approvals is None:
            return 0

        queued = 0
        pending = await self._notifications.approval_ids_awaiting_notification()
        for approval_id in pending:
            request = await self._approvals.get(approval_id)
            if request is None or not request.is_open:
                continue

            job = await self._jobs.get(request.job_id)
            client_kind = "telegram"     # the default surface
            session_id = None
            if job is not None and job.session_id is not None:
                session = await self._sessions.get(job.session_id)
                if session is not None:
                    client_kind = session.client_kind
                    session_id = session.id

            # The EXACT action, not a summary of it. An approval prompt
            # that hides the specifics trains the owner to tap yes
            # without reading, which is worse than no gate at all.
            text = (
                f"Permission needed, sir.\n\n"
                f"{request.summary}\n\n"
                f"{request.detail}"
            )
            if request.risk_note:
                text += f"\n\nRisk: {request.risk_note}"

            created = await self._notifications.create(Notification(
                client_kind=client_kind,
                text=text,
                session_id=session_id,
                job_id=request.job_id,
                approval_id=request.id,
                priority=_APPROVAL_PRIORITY,
                trace_id=request.trace_id,
            ))
            if created:
                queued += 1
        return queued

    # -- deliver --------------------------------------------------------------

    async def deliver_pending(self) -> int:
        """Send queued notifications, subject to policy."""
        sent = 0
        now = utc_now()
        digestible: list[Notification] = []

        for note in await self._notifications.pending():
            decision = self._policy.decide(note, now, self._tz)

            if decision is Decision.DEFER:
                until = self._policy.next_waking_moment(now, self._tz)
                await self._notifications.defer(note.id, until)
                await self._events.append(Event(
                    kind=EventKind.INITIATIVE_NOTIFICATION_SUPPRESSED,
                    source=SOURCE, session_id=note.session_id,
                    job_id=note.job_id, trace_id=note.trace_id,
                    payload={"reason": "quiet hours",
                             "until": until.isoformat()},
                ))
                continue

            if decision is Decision.DIGEST:
                digestible.append(note)
                continue

            if await self._send_one(note):
                sent += 1

        # Batched items go out together once enough accumulate, or once
        # the oldest has waited long enough. One message about fourteen
        # things gets read; fourteen buzzes get muted.
        if digestible:
            sent += await self._maybe_send_digest(digestible, now)

        return sent

    async def _maybe_send_digest(
        self, notifications: list[Notification], now: datetime
    ) -> int:
        """Send a batch, if there is enough of one yet."""
        oldest = min(n.ts for n in notifications)
        window = timedelta(minutes=self._policy.digest_window_minutes)
        if len(notifications) < 3 and now - oldest < window:
            return 0        # not enough yet, and not old enough

        deliverer = self._deliverers.get(notifications[0].client_kind)
        if deliverer is None:
            return 0

        try:
            await deliverer.deliver(compose_digest(notifications))
        except Exception:
            log.warning("digest delivery failed, will retry", exc_info=True)
            return 0

        for note in notifications:
            await self._notifications.mark_delivered(note.id)
        log.info("digest delivered", extra={"count": len(notifications)})
        return 1

    async def _send_one(self, note: Notification) -> bool:
        """Deliver one notification. Returns whether it went out.

        Never raises: a failed send leaves the row pending so the next
        cycle retries it, and a transient network failure must not eat
        the message.
        """
        # Every surface, not just the one that spawned the message.
        # Sessions are shared across clients by owner decision, so
        # "which surface does this belong to" no longer has a single
        # answer - and a result delivered only where the owner is not
        # looking is a result he does not get.
        deliverer = self._deliverers.get(note.client_kind)
        others = [
            d for kind, d in self._deliverers.items()
            if kind != note.client_kind
        ]
        for extra in others:
            try:
                await extra.deliver(
                    note.text, None, None, note.approval_id
                )
            except Exception:
                pass        # best effort; the primary surface is what counts

        if deliverer is None:
            # Nothing can ever deliver this: settle it rather than
            # retrying forever.
            await self._notifications.mark_suppressed(
                note.id, f"no client registered for {note.client_kind}"
            )
            await self._events.append(Event(
                kind=EventKind.INITIATIVE_NOTIFICATION_SUPPRESSED,
                source=SOURCE, session_id=note.session_id,
                job_id=note.job_id, trace_id=note.trace_id,
                payload={"reason": "no client",
                         "client_kind": note.client_kind},
            ))
            return False

        file_path: Path | None = None
        file_name: str | None = None
        if note.artifact_id:
            artifact = await self._artifacts.get(note.artifact_id)
            if artifact is not None:
                file_path = self._artifacts.path_for(artifact)
                file_name = artifact.name

        try:
            await deliverer.deliver(
                note.text, file_path, file_name, note.approval_id
            )
        except Exception:
            log.warning("delivery failed, will retry",
                        exc_info=True, extra={"notification_id": note.id})
            return False

        await self._notifications.mark_delivered(note.id)
        await self._events.append(Event(
            kind=EventKind.INITIATIVE_NOTIFICATION_SENT,
            source=SOURCE, session_id=note.session_id,
            job_id=note.job_id, trace_id=note.trace_id,
            payload={"client_kind": note.client_kind,
                     "approval": note.approval_id is not None},
        ))
        await self._record_as_turn(note)
        return True

    async def _record_as_turn(self, note: Notification) -> None:
        """A delivered notification IS JARVIS speaking, so it belongs in
        the conversation - his next reply then knows what he already
        said, and the owner can respond to it naturally."""
        if note.session_id is None:
            return
        await self._sessions.append_turn(Turn(
            session_id=note.session_id,
            role=TurnRole.ASSISTANT,
            content=note.text,
            job_refs=[note.job_id] if note.job_id else [],
        ))
