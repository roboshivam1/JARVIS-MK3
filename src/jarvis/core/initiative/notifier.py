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
#   deliver        - pending rows go out through the client that owns
#                    their session. Approval-carrying rows render with
#                    buttons; the rest render as plain messages.
#
# Duplicates are impossible by construction: unique indexes on job_id and
# approval_id mean a crash mid-scan cannot produce two copies of the same
# message, and neither scan needs bookkeeping of its own.
#
# Delivered messages are recorded as assistant turns, so JARVIS knows
# what he already told the owner and the owner can reply to it naturally.
#
# Phase-2 policy still applies: deliver everything, immediately. Quiet
# hours, digests, and priority routing replace ONLY the policy decision
# inside this file - approvals will be the exception that always wakes
# the owner, which is exactly the sort of rule this choke point exists
# to make expressible in one place.
# =============================================================================

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from jarvis.common.events import Event, EventKind
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

log = get_logger("core.initiative.notifier")

SOURCE = "core.initiative"
_POLL_INTERVAL_S = 3.0

# Approvals are the one thing that should reach the owner ahead of
# everything else: work is paused until he answers.
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
    actually reach the owner."""

    def __init__(
        self,
        jobs: JobsRepo,
        sessions: SessionsRepo,
        notifications: NotificationsRepo,
        artifacts: ArtifactsRepo,
        events: EventsRepo,
        approvals: ApprovalsRepo | None = None,
    ) -> None:
        self._jobs = jobs
        self._sessions = sessions
        self._notifications = notifications
        self._artifacts = artifacts
        self._events = events
        self._approvals = approvals
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
            job = await self._jobs.get(job_id)
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
        for approval_id in await self._notifications.approval_ids_awaiting_notification():
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
        """Send queued notifications through their client surface."""
        sent = 0
        for note in await self._notifications.pending():
            deliverer = self._deliverers.get(note.client_kind)
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
                continue

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
                # Stays pending: the next cycle tries again. A transient
                # network failure must not eat the message.
                log.warning("delivery failed, will retry",
                            exc_info=True, extra={"notification_id": note.id})
                continue

            await self._notifications.mark_delivered(note.id)
            await self._events.append(Event(
                kind=EventKind.INITIATIVE_NOTIFICATION_SENT,
                source=SOURCE, session_id=note.session_id,
                job_id=note.job_id, trace_id=note.trace_id,
                payload={"client_kind": note.client_kind,
                         "approval": note.approval_id is not None},
            ))
            await self._record_as_turn(note)
            sent += 1
        return sent

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