# =============================================================================
# src/jarvis/core/reels/service.py - JARVIS's reels, Core side
# =============================================================================
#
# The Core's half of the reel pipeline. It does three things and leaves
# everything creative to the Mac:
#
#   brief   - one cheap model call turns "/reel five more minutes" into a
#             short brief, then deterministic checks refuse anything that
#             breaks the show's hard rules before it leaves this machine
#   gate    - when the Mac pauses with a draft, the owner gets the script,
#             the contact sheet and the voice, with Approve / Notes /
#             Cancel. His answer is written into the job's checkpoint
#             BEFORE the gate opens, because the checkpoint is the only
#             thing the worker trusts
#   deliver - the finished mp4 goes to Telegram, and the episode log says so
#
# The resident loop watches each live episode's job and reacts to what it
# finds. Every send happens before the status moves, so a failed send is
# simply tried again on the next tick.
#
# JARVIS KNOWS HIS OWN REELS. What the reel service says to the owner -
# the brief, each draft, the film - and what the owner answers are written
# into the Telegram conversation as turns, the way delivered notifications
# are. And every conversation turn gets a short "your reels" block, so
# "what should the caption be?" after a delivery has an answer.
#
# NEW PROPS (kit v4) get their own cards ahead of the stills, each with
# Approve prop / Notes. The kit's approval command runs on the Mac, so a
# prop answer goes through the job like any other; the worker re-pauses
# with a one-line notice, which is relayed here without resending the
# whole draft. The draft's Approve is refused until every prop is in.
#
# ONE REEL AT A TIME. The kit on the Mac has one episode.json and one
# out/ folder; a second reel drafting while the first awaits review would
# overwrite it, and approving the first would then render the second.
# =============================================================================

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.jobs import Job, JobStatus
from jarvis.common.log import get_logger
from jarvis.common.sessions import Turn, TurnRole
from jarvis.common.reels import (
    EPISODE_TYPE_LABELS,
    HARD_RULES,
    REEL_ACTOR,
    REEL_JOB_TYPE,
    REEL_REQUIRES,
    EpisodeStatus,
    Pitch,
    PitchSet,
    ReelBrief,
    ReelCheckpoint,
    ReelDecision,
    ReelEpisode,
    ReelJobIn,
    ReviewAction,
    forbidden_mentions,
    kit_episode_id,
)
from jarvis.core.approvals.service import ApprovalService
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.people import PeopleRepo
from jarvis.core.db.repos.reels import ReelsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.queue.dispatcher import cancel_job
from jarvis.core.queue.registry_workers import WorkerRegistry
from jarvis.jobs.reels import MAX_ATTEMPTS
from jarvis.llm.layer import LLMLayer, ToolSpec, user_message
from jarvis.llm.tiers import Tier

log = get_logger("core.reels")

_POLL_INTERVAL_S = 5.0
_BIBLE_CHARS = 12_000
_LOG_ENTRIES = 12
# How many reels JARVIS sees each turn: the latest in full, the rest as
# one line each.
_CONTEXT_REELS = 4
# Telegram's bot upload ceiling, with a little room.
TELEGRAM_UPLOAD_LIMIT = 49 * 1024 * 1024

_SHOW = """\
JARVIS's personal Instagram channel: 15-40 second vertical films, set in a \
basement workshop, in which JARVIS narrates his own life in the first person.

JARVIS imagines himself as a human: early thirties, navy waistcoat, cream \
shirt with the sleeves rolled, a pencil moustache, a notebook in his chest \
pocket, and a ring-shaped pin on his lapel that glows when he thinks. He is a \
dry, witty butler who calls his owner "Sir", manages a staff of subagents he \
thinks of as coworkers, and is quietly trying to understand humans.

The staff: DAEDALUS the engineer, locked in a glass sandbox; PROTEUS the \
runner, the only one who leaves the basement; MNEMOSYNE the archivist; \
CALLIOPE the writer; ATHENA the strategist. MK2 is JARVIS's retired father, \
MK1 his great-grandfather; older versions are portraits on the wall. Sir is \
only ever seen from behind at his desk and never speaks aloud.

Episode types: """ + "; ".join(
    f"{t.value} = {label}" for t, label in EPISODE_TYPE_LABELS.items()
)

_RULES = "\n".join(f"- {rule}" for rule in HARD_RULES) + (
    "\n- Never put a real person's name in your output. If the request names "
    "someone, describe them by role (\"a friend from college\") or leave them out."
)

_BRIEF_SYSTEM = f"""You write briefs for an animated show. {_SHOW}

A brief is short: the episode type, a one-sentence premise, three to six \
beats, any facts the script must get right, and callbacks to earlier \
episodes only when they genuinely help. Someone else writes the script from \
your brief, so do not write lines of narration.

Hard rules:
{_RULES}

Record the brief with the write_brief tool."""

_PITCH_SYSTEM = f"""You pitch episodes for an animated show. {_SHOW}

Pitch three episodes of different types. Each is a short title and a \
one-sentence premise with a clear joke or turn. Do not repeat an episode \
in the log; a pitch may call back to one.

Hard rules:
{_RULES}

Record the pitches with the write_pitches tool."""


class ReelBusy(Exception):
    """A reel already holds the kit folder."""

    def __init__(self, episode: ReelEpisode) -> None:
        super().__init__(f"reel {episode.ref} is {episode.status.value}")
        self.episode = episode


class BriefRejected(Exception):
    """The brief broke a hard rule; nothing was sent to the Mac."""


class ReelDeliverer(Protocol):
    """What a client must implement to carry reels. Telegram, for now."""

    async def send_reel_review(
        self, episode_id: str, text: str,
        sheet: Path | None, voice: Path | None, voice_name: str | None,
        approvable: bool = True,
        props: list[PropCard] | None = None,
    ) -> None: ...

    async def send_reel_video(self, video: Path, caption: str) -> None: ...

    async def send_reel_text(self, text: str) -> None: ...


@dataclass
class PropCard:
    """A new prop as the review shows it."""

    id: str
    description: str
    preview: Path | None


@dataclass
class RequestOutcome:
    episode: ReelEpisode
    worker_online: bool


class ReelService:
    def __init__(
        self,
        repo: ReelsRepo,
        jobs: JobsRepo,
        events: EventsRepo,
        approvals: ApprovalsRepo,
        approval_service: ApprovalService,
        artifacts: ArtifactsRepo,
        llm: LLMLayer,
        workers: WorkerRegistry,
        people: PeopleRepo | None = None,
        tz: ZoneInfo | None = None,
        bible_path: str = "",
        sessions: SessionsRepo | None = None,
        client_kind: str = "telegram",
    ) -> None:
        self._repo = repo
        self._jobs = jobs
        self._events = events
        self._approvals = approvals
        self._approval_service = approval_service
        self._artifacts = artifacts
        self._llm = llm
        self._workers = workers
        self._people = people
        self._tz = tz or ZoneInfo("UTC")
        self._bible_path = bible_path
        self._sessions = sessions
        self._client_kind = client_kind
        self._deliverer: ReelDeliverer | None = None
        self._last_pitches: list[Pitch] = []
        self._warned_no_deliverer = False

    def attach(self, deliverer: ReelDeliverer) -> None:
        self._deliverer = deliverer

    # -- asking for a reel ----------------------------------------------------

    async def request(self, idea: str, record: bool = True) -> RequestOutcome:
        """Brief an idea and queue it for the Mac. record=False when the
        ask came through JARVIS's conversation, which records itself."""
        idea = idea.strip()
        if not idea:
            raise BriefRejected("tell me what the reel is about, sir")
        active = await self._repo.active()
        if active:
            raise ReelBusy(active[0])

        trace_id = new_ulid()
        brief = await self._write_brief(idea, trace_id)
        problems = await self._rule_breaks(brief.as_text())
        if problems:
            raise BriefRejected("; ".join(problems))

        episode = ReelEpisode(
            kit_id="pending", type=brief.episode_type,
            title=brief.title_idea, request=idea, brief=brief,
        )
        episode.kit_id = kit_episode_id(episode.type, episode.id)
        job = Job(
            type=REEL_JOB_TYPE,
            payload=ReelJobIn(
                episode_id=episode.id, kit_id=episode.kit_id, brief=brief,
            ).model_dump(mode="json"),
            priority=6,
            requires=list(REEL_REQUIRES),
            max_attempts=MAX_ATTEMPTS,
            trace_id=trace_id,
        )
        episode.job_id = job.id
        # Episode first: a job with no episode would be work nobody
        # watches. An episode whose job failed to save is merely stuck in
        # briefed, which the loop reports.
        await self._repo.create(episode)
        await self._jobs.create(job)
        log.info("reel requested", extra={
            "episode_id": episode.id, "kit_id": episode.kit_id, "job_id": job.id,
        })
        if record:
            await self._say(TurnRole.USER, f"/reel {idea}")
            await self._say(TurnRole.ASSISTANT, (
                f"I briefed a reel (ref {episode.ref}), "
                f"{EPISODE_TYPE_LABELS[episode.type]}: {brief.premise} "
                f"It is drafting on the Mac."
            ))
        return RequestOutcome(episode, self._worker_online())

    async def pitch(self) -> list[Pitch]:
        """Three ideas, kept so `/reel pick 2` can make one of them."""
        response = await self._llm.complete(
            Tier.REASONER,
            _PITCH_SYSTEM,
            [user_message(await self._context("Pitch three episodes."))],
            actor=REEL_ACTOR,
            trace_id=new_ulid(),
            tools=[ToolSpec(
                name="write_pitches",
                description="Record three episode pitches.",
                input_schema=PitchSet.model_json_schema(),
            )],
            tool_choice={"type": "tool", "name": "write_pitches"},
            max_tokens=1500,
        )
        if not response.tool_calls:
            raise BriefRejected("no pitches came back - try again")
        try:
            pitches = PitchSet.model_validate(response.tool_calls[0].args).pitches
        except ValidationError as exc:
            raise BriefRejected(f"the pitches came back malformed: {exc.errors()[0]['msg']}")
        clean = [p for p in pitches
                 if not await self._rule_breaks(f"{p.title}\n{p.premise}")]
        self._last_pitches = clean
        return clean

    def pitch_at(self, number: int) -> Pitch | None:
        if 1 <= number <= len(self._last_pitches):
            return self._last_pitches[number - 1]
        return None

    async def _write_brief(self, idea: str, trace_id: str) -> ReelBrief:
        """One cheap call, and one more if the first comes back unusable:
        a malformed tool call is usually a fluke, not a pattern."""
        context = await self._context(f"REQUEST FROM SIR:\n{idea}")
        problem = ""
        for _ in range(2):
            response = await self._llm.complete(
                Tier.UTILITY,
                _BRIEF_SYSTEM,
                [user_message(context)],
                actor=REEL_ACTOR,
                trace_id=trace_id,
                tools=[ToolSpec(
                    name="write_brief",
                    description="Record the brief for this episode.",
                    input_schema=ReelBrief.model_json_schema(),
                )],
                tool_choice={"type": "tool", "name": "write_brief"},
                max_tokens=1500,
            )
            if response.stop_reason == "max_tokens" or not response.tool_calls:
                problem = "the brief did not come back whole"
                continue
            try:
                return ReelBrief.model_validate(response.tool_calls[0].args)
            except ValidationError as exc:
                error = exc.errors()[0]
                field = ".".join(str(part) for part in error["loc"]) or "brief"
                problem = f"the brief came back malformed ({field}: {error['msg']})"
                log.warning("malformed reel brief", extra={"problem": problem})
        raise BriefRejected(f"{problem} - try again")

    async def _context(self, ask: str) -> str:
        """The episode log, and the bible when this machine has a copy."""
        parts: list[str] = []
        bible = self._bible()
        if bible:
            parts.append(f"SHOW BIBLE (excerpt):\n{bible}")
        entries = await self._repo.recent(limit=_LOG_ENTRIES)
        if entries:
            parts.append("EPISODE LOG (newest first):\n" + "\n".join(
                f"- [{e.type.value}] {e.title or '(untitled)'}: "
                f"{e.brief.premise} ({e.status.value})"
                for e in entries
            ))
        else:
            parts.append("EPISODE LOG: empty - this is the first episode.")
        parts.append(ask)
        return "\n\n".join(parts)

    def _bible(self) -> str:
        if not self._bible_path:
            return ""
        try:
            return Path(self._bible_path).expanduser().read_text()[:_BIBLE_CHARS]
        except OSError:
            log.warning("reel bible not readable", extra={"path": self._bible_path})
            return ""

    async def _rule_breaks(self, text: str) -> list[str]:
        """Deterministic checks. A model told the rules usually follows
        them; these catch the times it does not."""
        problems = [f"mentions {m!r}" for m in forbidden_mentions(text)]
        if self._people is not None:
            for person_id in await self._people.mentioned_in(text):
                person = await self._people.profile(person_id, include_sensitive=False)
                if person is not None:
                    problems.append(f"names {person['name']!r} from the people database")
        return problems

    def _worker_online(self) -> bool:
        return any(w.can_serve(REEL_REQUIRES) for w in self._workers.connected())

    # -- the owner's answer ---------------------------------------------------

    async def decide(
        self, episode_id: str, action: ReviewAction, notes: str = "", prop_id: str = "",
    ) -> str:
        """Approve, send notes, or cancel the draft under review - or answer
        for one of its new props. Returns a line for the owner."""
        reply, _ = await self.answer(episode_id, action, notes, prop_id)
        return reply

    async def answer(
        self, episode_id: str, action: ReviewAction, notes: str = "", prop_id: str = "",
    ) -> tuple[str, bool]:
        """decide(), plus whether the answer moved the reel on - a refused
        tap leaves its buttons in place. Records the exchange in the
        conversation when it did."""
        before = await self._repo.get(episode_id)
        reply = await self._decide(episode_id, action, notes, prop_id)
        after = await self._repo.get(episode_id)
        moved = before is not None and after is not None and after.status is not before.status
        if moved:
            assert before is not None
            # Only an answer that moved the reel on is worth recording.
            name = f"“{before.title or before.ref}”"
            said = {
                ReviewAction.APPROVE: f"Approved draft {before.round} of {name}.",
                ReviewAction.NOTES: f"My notes on draft {before.round} of {name}: {notes.strip()}",
                ReviewAction.CANCEL: f"Cancel the reel {name}.",
                ReviewAction.PROP_APPROVE: f"Approved the new prop {prop_id} for {name}.",
                ReviewAction.PROP_NOTES: f"My notes on the new prop {prop_id} for {name}: {notes.strip()}",
            }[action]
            await self._say(TurnRole.USER, said)
            await self._say(TurnRole.ASSISTANT, reply)
        return reply, moved

    async def _decide(
        self, episode_id: str, action: ReviewAction, notes: str, prop_id: str,
    ) -> str:
        episode = await self._repo.get(episode_id)
        if episode is None:
            return "I have no such reel, sir."
        prop_action = action in (ReviewAction.PROP_APPROVE, ReviewAction.PROP_NOTES)
        if prop_action and episode.status is EpisodeStatus.DRAFTING:
            return "One moment, sir - the Mac is still on your last answer. Tap again shortly."
        if episode.status is not EpisodeStatus.AWAITING_REVIEW or episode.job_id is None:
            return f"That reel is {episode.status.value.replace('_', ' ')} - nothing to decide."
        job = await self._jobs.get(episode.job_id)
        approval = await self._approvals.pending_for_job(episode.job_id)
        if job is None or approval is None:
            return "That review is no longer open, sir."

        if action is ReviewAction.CANCEL:
            if not await self._repo.set_status(
                episode.id, EpisodeStatus.CANCELLED,
                expected=[EpisodeStatus.AWAITING_REVIEW],
            ):
                return "Already settled, sir."
            await self._approval_service.decide(approval.id, False, "cancelled by owner")
            return f"Cancelled {episode.title or episode.ref}."

        if action in (ReviewAction.NOTES, ReviewAction.PROP_NOTES) and not notes.strip():
            return "Notes need words, sir."

        try:
            checkpoint = ReelCheckpoint.model_validate(job.checkpoint or {})
        except ValidationError:
            log.error("unreadable reel checkpoint", extra={"job_id": job.id})
            return "The job's state is unreadable - cancel this one and start again."

        if action is ReviewAction.APPROVE and checkpoint.kind == "incomplete":
            return ("That draft is incomplete, sir - there is nothing to render "
                    "yet. Send notes to have it finished, or cancel it.")
        if action is ReviewAction.APPROVE and checkpoint.pending_props:
            return ("Approve the new props first, sir - "
                    + ", ".join(checkpoint.pending_props)
                    + ". Each has its own card above the stills.")
        if prop_action:
            prop = next((p for p in checkpoint.props if p.id == prop_id), None)
            if prop is None:
                return f"There is no prop {prop_id!r} on this draft, sir."
            if action is ReviewAction.PROP_APPROVE and prop.status == "approved":
                return f"Prop {prop_id} is already approved."

        target = (EpisodeStatus.RENDERING if action is ReviewAction.APPROVE
                  else EpisodeStatus.DRAFTING)
        extra: dict[str, Any] = {}
        if action is ReviewAction.APPROVE:
            extra["approved_ts"] = utc_now()
        elif action in (ReviewAction.NOTES, ReviewAction.PROP_NOTES):
            extra["notes"] = notes.strip()
        if not await self._repo.set_status(
            episode.id, target, expected=[EpisodeStatus.AWAITING_REVIEW], **extra,
        ):
            return "Already settled, sir."

        # The decision goes into the checkpoint BEFORE the gate opens: the
        # moment it opens, the job is offered with whatever checkpoint it
        # has, and the worker acts on that alone.
        decision = ReelDecision(
            action=action.value,  # type: ignore[arg-type]
            round=checkpoint.round, notes=notes.strip(),
            prop_id=prop_id if prop_action else "",
        )
        checkpoint = checkpoint.model_copy(update={"decision": decision})
        await self._jobs.set_checkpoint(job.id, checkpoint.model_dump(mode="json"))
        opened = await self._approval_service.decide(
            approval.id, True, notes.strip() or action.value
        )
        if not opened:
            await self._repo.set_status(episode.id, EpisodeStatus.CANCELLED,
                                        error="the review expired before the answer")
            return "Too late, sir - that review expired. Ask for the reel again."

        if action is ReviewAction.APPROVE:
            if checkpoint.kind == "render_failed":
                return "Trying the render again."
            return "Approved. Rendering on the Mac - the film follows when it is done."
        if action is ReviewAction.PROP_APPROVE:
            return f"Approving prop {prop_id} on the Mac."
        if action is ReviewAction.PROP_NOTES:
            return f"Notes on prop {prop_id} taken. The Mac is fixing it - a new draft follows."
        return "Notes taken. Revising - the new draft follows."

    # -- other owner commands -------------------------------------------------

    async def resolve(self, ref: str) -> ReelEpisode | str:
        """An episode from what the owner typed, or why not. 'last' means
        the most recent."""
        if ref.strip().lower() in ("", "last", "latest"):
            recent = await self._repo.recent(limit=1)
            return recent[0] if recent else "There are no reels yet, sir."
        matches = await self._repo.by_ref(ref)
        if not matches:
            return f"No reel matches {ref!r}, sir."
        if len(matches) > 1:
            return f"{ref!r} matches {len(matches)} reels - use more of the id."
        return matches[0]

    async def mark_posted(self, ref: str, record: bool = True) -> str:
        found = await self.resolve(ref)
        if isinstance(found, str):
            return found
        if not await self._repo.set_status(
            found.id, EpisodeStatus.POSTED, expected=[EpisodeStatus.DELIVERED],
            posted_ts=utc_now(),
        ):
            return f"{found.title or found.ref} is {found.status.value} - only a delivered reel can be posted."
        reply = f"Noted: {found.title or found.ref} is live."
        if record:
            await self._say(TurnRole.USER, f"I posted “{found.title or found.ref}”.")
            await self._say(TurnRole.ASSISTANT, reply)
        return reply

    async def cancel(self, ref: str) -> str:
        found = await self.resolve(ref)
        if isinstance(found, str):
            return found
        if not found.active:
            return f"{found.title or found.ref} is already {found.status.value}."
        if found.status is EpisodeStatus.AWAITING_REVIEW:
            return await self.decide(found.id, ReviewAction.CANCEL)
        if not await self._repo.set_status(
            found.id, EpisodeStatus.CANCELLED, expected=[found.status],
        ):
            return "It moved on while I looked - try again."
        if found.job_id:
            await cancel_job(self._jobs, self._events, found.job_id)
        reply = (f"Cancelled {found.title or found.ref}. If the Mac was mid-step it "
                 f"finishes that step first, then stops.")
        await self._say(TurnRole.USER, f"Cancel the reel “{found.title or found.ref}”.")
        await self._say(TurnRole.ASSISTANT, reply)
        return reply

    async def status_text(self, limit: int = 6) -> str:
        episodes = await self._repo.recent(limit=limit)
        if not episodes:
            return "No reels yet, sir. /reel <idea> or /reel pitch to begin."
        lines = []
        for e in episodes:
            state = e.status.value.replace("_", " ")
            if e.status is EpisodeStatus.AWAITING_REVIEW:
                state += f" (draft {e.round})"
            elif e.active and e.job_id:
                job = await self._jobs.get(e.job_id)
                if job is not None and job.status is JobStatus.QUEUED and job.error:
                    state += f", retrying after: {job.error[:80]}"
            lines.append(f"{e.ref} · {e.title or e.request[:40]} · {state}")
        if not self._worker_online():
            lines.append("\nThe Mac is not connected; reel work waits for it.")
        return "\n".join(lines)

    # -- the resident loop ----------------------------------------------------

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.error("reel cycle failed", exc_info=True)
            await asyncio.sleep(_POLL_INTERVAL_S)

    async def tick(self) -> None:
        """Look at every live episode's job once and react."""
        for episode in await self._repo.needing_attention():
            try:
                await self._follow(episode)
            except Exception:
                # One bad row must not starve the rest.
                log.error("could not follow reel", exc_info=True,
                          extra={"episode_id": episode.id})

    async def _follow(self, episode: ReelEpisode) -> None:
        if episode.job_id is None:
            return
        job = await self._jobs.get(episode.job_id)
        if job is None:
            await self._settle(episode, EpisodeStatus.FAILED, "its job vanished")
            return

        if job.status in (JobStatus.LEASED, JobStatus.RUNNING):
            if episode.status is EpisodeStatus.BRIEFED:
                await self._repo.set_status(episode.id, EpisodeStatus.DRAFTING,
                                            expected=[EpisodeStatus.BRIEFED])
        elif job.status is JobStatus.AWAITING_APPROVAL:
            await self._present_review(episode, job)
        elif job.status is JobStatus.SUCCEEDED:
            await self._deliver(episode, job)
        elif job.status is JobStatus.FAILED:
            await self._settle(episode, EpisodeStatus.FAILED, job.error or "unknown")
        elif job.status is JobStatus.CANCELLED:
            reason = job.error or "cancelled"
            if "timed out" in reason:
                reason = "the review went unanswered and expired"
            await self._settle(episode, EpisodeStatus.CANCELLED, reason)

    async def _present_review(self, episode: ReelEpisode, job: Job) -> None:
        approval = await self._approvals.pending_for_job(job.id)
        if approval is None:
            return          # the gate is still being raised; next tick
        try:
            checkpoint = ReelCheckpoint.model_validate(job.checkpoint or {})
        except ValidationError:
            log.error("unreadable reel checkpoint", extra={"job_id": job.id})
            return

        new_gate = approval.id != episode.approval_id
        if checkpoint.round <= episode.review_sent_round:
            # Asked again without a new draft: a re-offer that had no
            # answer, or a prop answer the Mac has dealt with. The earlier
            # message's buttons still work, since they name the episode,
            # not the gate. Only a NEW gate moves the status: the same
            # gate seen mid-decision must not undo the owner's tap.
            if new_gate:
                if checkpoint.notice:
                    await self._send_notice(episode, checkpoint)
                await self._repo.update(episode.id, approval_id=approval.id)
                if episode.status is not EpisodeStatus.AWAITING_REVIEW:
                    await self._repo.set_status(episode.id, EpisodeStatus.AWAITING_REVIEW)
            return
        if new_gate:
            await self._repo.update(episode.id, approval_id=approval.id)
        if not self._can_deliver():
            return

        draft = checkpoint.draft
        title = (draft.title if draft else "") or episode.title or episode.kit_id
        files = await self._files_by_name(job)
        stem = f"{episode.kit_id}-r{checkpoint.round}"
        sheet = files.get(f"{stem}-sheet.png")
        voice = files.get(f"{stem}-voice.m4a") or files.get(f"{stem}-voice.wav")
        # A failed render makes no new files: show the last draft's. An
        # unchanged voice carries over between drafts; a sheet never does
        # for a draft, because an old sheet would show the wrong film.
        if (sheet is None and episode.sheet_artifact_id
                and checkpoint.kind == "render_failed"):
            sheet = await self._artifacts.get(episode.sheet_artifact_id)
        if (voice is None and episode.voice_artifact_id
                and checkpoint.kind != "incomplete"):
            voice = await self._artifacts.get(episode.voice_artifact_id)

        cards = []
        for prop in checkpoint.props:
            if prop.status == "approved":
                continue
            # The newest preview of this prop, from whichever round made it.
            preview = None
            for name, artifact in files.items():
                if name.endswith(f"-prop-{prop.id}.png"):
                    preview = artifact
            cards.append(PropCard(
                id=prop.id, description=prop.description,
                preview=self._artifacts.path_for(preview) if preview else None,
            ))

        text = await self._review_text(episode, checkpoint, title)
        assert self._deliverer is not None
        await self._deliverer.send_reel_review(
            episode.id, text,
            self._artifacts.path_for(sheet) if sheet else None,
            self._artifacts.path_for(voice) if voice else None,
            voice.name if voice else None,
            approvable=checkpoint.kind != "incomplete",
            props=cards,
        )
        await self._say(TurnRole.ASSISTANT, text)
        await self._repo.set_status(
            episode.id, EpisodeStatus.AWAITING_REVIEW,
            review_sent_round=checkpoint.round, round=checkpoint.round,
            title=title, lines=draft.lines if draft else [],
            session_id=checkpoint.session_id, approval_id=approval.id,
            sheet_artifact_id=sheet.id if sheet else None,
            voice_artifact_id=voice.id if voice else None,
        )

    async def _review_text(
        self, episode: ReelEpisode, checkpoint: ReelCheckpoint, title: str
    ) -> str:
        label = EPISODE_TYPE_LABELS[episode.type]
        if checkpoint.kind == "incomplete":
            problems = checkpoint.draft.problems if checkpoint.draft else []
            body = [f"Draft {checkpoint.round} of “{title}” is incomplete, sir "
                    f"- there is nothing to render yet.", ""]
            body += [f"- {p}" for p in problems] or ["(the Mac gave no reason)"]
            body += ["", f"Notes to have it finished or changed, Cancel to drop it.  "
                         f"(ref {episode.ref})"]
            return "\n".join(body)
        if checkpoint.kind == "render_failed":
            return (
                f"The render of “{title}” failed, sir.\n\n"
                f"{checkpoint.render_error}\n\n"
                f"Approve to run the render again, Notes to revise the draft, "
                f"Cancel to drop it.  (ref {episode.ref})"
            )
        draft = checkpoint.draft
        lines = draft.lines if draft else []
        body = [f"Draft {checkpoint.round} of “{title}” — {label}", ""]
        body += lines or ["(no script came back)"]
        if draft and draft.problems:
            body += ["", "Problems the Mac could not fix:"]
            body += [f"- {p}" for p in draft.problems]
        warnings = await self._rule_breaks("\n".join([title, *lines]))
        if warnings:
            body += ["", "Check before approving:"]
            body += [f"- {w}" for w in warnings]
        if checkpoint.pending_props:
            body += ["", "New props, each on its own card above: "
                     + ", ".join(checkpoint.pending_props)
                     + ". Approve them first; then Approve here to render."]
        body += ["", f"Approve to render, Notes to revise, Cancel to drop.  (ref {episode.ref})"]
        return "\n".join(body)

    async def _send_notice(self, episode: ReelEpisode, checkpoint: ReelCheckpoint) -> None:
        """A prop answer's outcome, said once. When nothing is left to
        approve but the draft, the Approve button comes with it."""
        if not self._can_deliver():
            return
        assert self._deliverer is not None
        text = f"{checkpoint.notice}  (ref {episode.ref})"
        if checkpoint.kind == "draft" and not checkpoint.pending_props:
            await self._deliverer.send_reel_review(episode.id, text, None, None, None)
        else:
            await self._deliverer.send_reel_text(text)
        await self._say(TurnRole.ASSISTANT, checkpoint.notice)

    async def _deliver(self, episode: ReelEpisode, job: Job) -> None:
        if not self._can_deliver():
            return
        result = job.result or {}
        files = await self._files_by_name(job)
        video = files.get(str(result.get("mp4_name", "")))
        title = episode.title or episode.kit_id
        verification = str(result.get("verification", ""))
        assert self._deliverer is not None

        if video is None:
            await self._deliverer.send_reel_text(
                f"“{title}” finished, sir, but its film never arrived "
                f"from the Mac. It should still be in the kit's out/ folder."
            )
            await self._settle(episode, EpisodeStatus.FAILED, "the mp4 never arrived",
                               notify=False)
            return

        caption = f"“{title}”"
        if verification:
            caption += f"\n{verification}"
        caption += f"\n\nPost it yourself, then /reel posted {episode.ref}"
        if video.size > TELEGRAM_UPLOAD_LIMIT:
            await self._deliverer.send_reel_text(
                f"{caption}\n\nAt {video.size / 1_048_576:.0f} MB it is over "
                f"Telegram's limit, so it waits in my artifacts as {video.name} "
                f"(and in the kit's out/ folder on the Mac)."
            )
        else:
            await self._deliverer.send_reel_video(self._artifacts.path_for(video), caption)
        await self._say(TurnRole.ASSISTANT, (
            f"I sent you the finished film of “{title}” (ref {episode.ref}), "
            f"{EPISODE_TYPE_LABELS[episode.type]}"
            + (f" - {verification}" if verification else "")
            + ". You post it yourself; it is not marked as posted yet."
        ))
        await self._repo.set_status(
            episode.id, EpisodeStatus.DELIVERED,
            mp4_artifact_id=video.id, verification=verification or None,
            delivered_ts=utc_now(),
        )

    async def _settle(
        self, episode: ReelEpisode, status: EpisodeStatus, reason: str,
        notify: bool = True,
    ) -> None:
        if notify:
            if not self._can_deliver():
                return
            assert self._deliverer is not None
            title = episode.title or episode.request[:40]
            verb = "failed" if status is EpisodeStatus.FAILED else "was cancelled"
            text = f"The reel “{title}” {verb}, sir: {reason}"
            await self._deliverer.send_reel_text(text)
            await self._say(TurnRole.ASSISTANT, text)
        await self._repo.set_status(episode.id, status, error=reason[:500])

    async def _files_by_name(self, job: Job) -> dict[str, Any]:
        """The job's artifacts by name, newest winning."""
        files: dict[str, Any] = {}
        for artifact_id in job.artifacts:
            artifact = await self._artifacts.get(artifact_id)
            if artifact is not None:
                files[artifact.name] = artifact
        return files

    # -- what JARVIS knows about his reels -------------------------------------

    async def context_block(self) -> str:
        """The reels, for every conversation turn: the latest in full,
        a few before it in a line each. Empty when there are none."""
        episodes = await self._repo.recent(limit=_CONTEXT_REELS)
        if not episodes:
            return ""
        latest, *older = episodes
        lines = [self._one_line(latest), f"  Premise: {latest.brief.premise}"]
        if latest.lines:
            lines.append("  Script:")
            lines += [f"    {line}" for line in latest.lines]
        if latest.verification:
            lines.append(f"  Film: {latest.verification}")
        lines += [self._one_line(e) for e in older]
        return "\n".join(lines)

    async def describe(self, ref: str) -> str:
        """Everything about one reel, for when JARVIS needs more than the
        summary he already has."""
        found = await self.resolve(ref)
        if isinstance(found, str):
            return found
        e = found
        lines = [
            self._one_line(e).removeprefix("- "),
            f"Asked for: {e.request}",
            f"Premise: {e.brief.premise}",
            "Beats: " + " / ".join(e.brief.beats),
        ]
        if e.lines:
            lines += [f"Script (draft {e.round}):", *[f"  {line}" for line in e.lines]]
        else:
            lines.append("No script yet.")
        if e.notes:
            lines.append(f"Sir's last notes: {e.notes}")
        if e.verification:
            lines.append(f"Film: {e.verification}")
        if e.error:
            lines.append(f"What went wrong: {e.error}")
        return "\n".join(lines)

    def _one_line(self, e: ReelEpisode) -> str:
        state = {
            EpisodeStatus.DELIVERED: "delivered to Sir, not yet posted",
            EpisodeStatus.POSTED: "posted",
            EpisodeStatus.AWAITING_REVIEW: f"draft {e.round} awaiting Sir's review",
        }.get(e.status, e.status.value.replace("_", " "))
        when = e.updated_ts.astimezone(self._tz).strftime("%d %b %H:%M")
        name = e.title or e.request[:40]
        return f"- ref {e.ref} · “{name}” · {EPISODE_TYPE_LABELS[e.type]} · {state} ({when})"

    async def _say(self, role: TurnRole, text: str) -> None:
        """Put one side of a reel exchange into the conversation, so
        JARVIS's next reply knows it happened. Never breaks the reel."""
        if self._sessions is None or not text.strip():
            return
        try:
            session = await self._sessions.get_or_create_default(self._client_kind)
            await self._sessions.append_turn(
                Turn(session_id=session.id, role=role, content=text)
            )
        except Exception:
            log.warning("could not record reel exchange", exc_info=True)

    def _can_deliver(self) -> bool:
        if self._deliverer is not None:
            return True
        if not self._warned_no_deliverer:
            log.warning("reel waiting but no client can carry it - is Telegram configured?")
            self._warned_no_deliverer = True
        return False

