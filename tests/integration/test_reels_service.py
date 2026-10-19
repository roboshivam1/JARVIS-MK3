# =============================================================================
# tests/integration/test_reels_service.py - the Core side of the reels
# =============================================================================
#
# Real database, real job queue, real approval service; a fake model and a
# fake Telegram. The worker is played by hand: tests move the job through
# the same states a real Mac would, and check what the owner sees.
#
# The promises that matter:
#   - a brief that breaks a hard rule never becomes a job
#   - one reel at a time: a second request is refused while one is live
#   - a draft reaches the owner once, with its stills and voice, and the
#     generic approval notification stays out of the way
#   - the owner's answer is in the checkpoint BEFORE the job is released
#   - a double tap settles once; the film is delivered once
#   - an incomplete draft arrives without Approve and cannot be approved
#   - what passes between the owner and the reel service lands in his
#     Telegram conversation, and JARVIS sees his reels every turn
#   - new props come as cards before the stills; the draft cannot be
#     approved until each prop is; a prop answer's outcome is said once
# =============================================================================

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.common.capabilities import Gate
from jarvis.common.ids import utc_now
from jarvis.common.jobs import JobStatus, Lease
from jarvis.common.reels import (
    EpisodeStatus,
    ReelCheckpoint,
    ReviewAction,
)
from jarvis.common.sessions import TurnRole
from jarvis.core.approvals.service import ApprovalService
from jarvis.core.db.database import Database
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.jobs import JobsRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.reels import ReelsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.initiative.notifier import Notifier
from jarvis.core.queue.registry_workers import ConnectedWorker, WorkerRegistry
from jarvis.core.reels.service import BriefRejected, ReelBusy, ReelService
from jarvis.llm.layer import ToolCall

BRIEF = {
    "episode_type": "field_notes",
    "premise": "Sir says five more minutes at 1:12 a.m. and means ninety.",
    "beats": ["The promise", "The evidence", "The verdict"],
    "facts": ["It was 1:12 a.m."],
    "title_idea": "Five more minutes",
}


class FakeLLM:
    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []
        self.prompts: list[str] = []

    async def complete(self, tier: Any, system: str, messages: list[dict[str, Any]],
                       **kwargs: Any) -> Any:
        self.prompts.append(messages[0]["content"])
        name = kwargs["tool_choice"]["name"]
        args = self.replies.pop(0)
        return SimpleNamespace(stop_reason="tool_use",
                               tool_calls=[ToolCall(id="t1", name=name, args=args)])


class FakeTelegram:
    def __init__(self) -> None:
        self.reviews: list[dict[str, Any]] = []
        self.videos: list[tuple[Path, str]] = []
        self.texts: list[str] = []

    async def send_reel_review(self, episode_id: str, text: str, sheet: Path | None,
                               voice: Path | None, voice_name: str | None,
                               approvable: bool = True,
                               props: list[Any] | None = None) -> None:
        self.reviews.append({"episode_id": episode_id, "text": text, "sheet": sheet,
                             "voice": voice, "voice_name": voice_name,
                             "approvable": approvable, "props": props or []})

    async def send_reel_video(self, video: Path, caption: str) -> None:
        self.videos.append((video, caption))

    async def send_reel_text(self, text: str) -> None:
        self.texts.append(text)


class FakePeople:
    """Knows one person, Riya, the way the people repo would."""

    async def mentioned_in(self, text: str) -> list[str]:
        return ["P1"] if "riya" in text.lower() else []

    async def profile(self, person_id: str, include_sensitive: bool = True) -> dict[str, Any]:
        return {"name": "Riya Sharma"}


@pytest.fixture
def world(db: Database, tmp_path: Path) -> SimpleNamespace:
    events = EventsRepo(db)
    jobs = JobsRepo(db, events)
    approvals = ApprovalsRepo(db)
    approval_service = ApprovalService(approvals, jobs, events)
    artifacts = ArtifactsRepo(db, tmp_path / "artifacts")
    workers = WorkerRegistry()
    llm = FakeLLM()
    telegram = FakeTelegram()
    repo = ReelsRepo(db)
    sessions = SessionsRepo(db)
    service = ReelService(repo, jobs, events, approvals, approval_service, artifacts,
                          llm, workers, people=FakePeople(),  # type: ignore[arg-type]
                          sessions=sessions)
    service.attach(telegram)
    notifier = Notifier(jobs, sessions, NotificationsRepo(db),
                        artifacts, events, approvals)
    return SimpleNamespace(db=db, events=events, jobs=jobs, approvals=approvals,
                           sessions=sessions,
                           approval_service=approval_service, artifacts=artifacts,
                           workers=workers, llm=llm, telegram=telegram, repo=repo,
                           service=service, notifier=notifier)


# -- playing the Mac ----------------------------------------------------------

async def _lease(w: SimpleNamespace, job_id: str) -> None:
    job = await w.jobs.get(job_id)
    now = utc_now()
    await w.jobs.transition(job_id, JobStatus.QUEUED, JobStatus.LEASED, set_fields={
        "lease": Lease(worker_id="macbook", leased_ts=now, heartbeat_ts=now, ttl_s=90),
        "attempts": job.attempts + 1,
    })
    await w.jobs.transition(job_id, JobStatus.LEASED, JobStatus.RUNNING)


async def _draft(w: SimpleNamespace, job_id: str, kit_id: str, round_no: int,
                 title: str = "Five more minutes", lines: list[str] | None = None,
                 kind: str = "draft", props: list[dict[str, Any]] | None = None) -> None:
    """What the worker does: upload, checkpoint, ask. An incomplete draft
    made no contact sheet."""
    await _lease(w, job_id)
    files = [("voice.m4a", "audio/mp4", b"M4A")]
    if kind == "draft":
        files.insert(0, ("sheet.png", "image/png", b"PNG"))
    for prop in props or []:
        files.append((f"prop-{prop['id']}.png", "image/png", b"PROP"))
    for suffix, mime, data in files:
        artifact = await w.artifacts.write(f"{kit_id}-r{round_no}-{suffix}", mime,
                                           data, created_by=job_id)
        await w.jobs.add_artifact(job_id, artifact.id)
    job = await w.jobs.get(job_id)
    previous = ReelCheckpoint.model_validate(job.checkpoint) if job.checkpoint else None
    checkpoint = ReelCheckpoint(
        stage="review", round=round_no, kit_id=kit_id, kind=kind,  # type: ignore[arg-type]
        session_id="sess-1" if previous is None else previous.session_id,
        draft={"status": "ready_for_review" if kind == "draft" else "failed",
               "title": title,
               "lines": lines or ["At one twelve a.m., Sir said five more minutes."],
               "sheet": "out/stills/sheet.png",
               "problems": [] if kind == "draft" else [
                   "Denied by the allow-list: `npm run voice 2>&1 | tail -20`"]},
        props=props or [],
    )
    await w.jobs.set_checkpoint(job_id, checkpoint.model_dump(mode="json"))
    await w.approval_service.request(
        job_id=job_id, gate=Gate.SPEND, actor="worker.reels", tool="reel_review",
        summary=f"Reel draft {round_no} ready for review: {title}", detail="...",
    )


async def _finish(w: SimpleNamespace, job_id: str, kit_id: str) -> None:
    await _lease(w, job_id)
    artifact = await w.artifacts.write(f"{kit_id}.mp4", "video/mp4", b"MP4",
                                       created_by=job_id)
    await w.jobs.add_artifact(job_id, artifact.id)
    await w.jobs.transition(job_id, JobStatus.RUNNING, JobStatus.SUCCEEDED, set_fields={
        "result": {"episode_id": "x", "summary": "done", "mp4_name": f"{kit_id}.mp4",
                   "verification": "30.2s 1080x1920 ok"},
        "lease": None,
    })


# -- asking ---------------------------------------------------------------------

async def test_request_queues_a_reel_job(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    outcome = await world.service.request("five more minutes")

    episode = outcome.episode
    assert episode.status is EpisodeStatus.BRIEFED
    assert episode.kit_id.startswith("field-notes-")
    assert not outcome.worker_online

    job = await world.jobs.get(episode.job_id)
    assert job.type == "reel.make"
    assert job.requires == ["render", "gpu"]
    assert job.max_attempts >= 10
    assert job.session_id is None                      # the generic notifier stays out
    assert job.payload["kit_id"] == episode.kit_id
    assert "EPISODE LOG: empty" in world.llm.prompts[0]


async def test_worker_online_is_reported(world: SimpleNamespace) -> None:
    world.workers.register(ConnectedWorker(
        worker_id="macbook", capabilities={"macos", "render", "gpu"},
        max_concurrency=2, version="0.1.0"))
    world.llm.replies.append(BRIEF)
    assert (await world.service.request("five more minutes")).worker_online


async def test_brand_mention_is_refused(world: SimpleNamespace) -> None:
    world.llm.replies.append({**BRIEF, "beats": ["A", "B", "A plug for Basement Supply"]})
    with pytest.raises(BriefRejected, match="Basement Supply"):
        await world.service.request("a shop plug")
    assert await world.repo.recent() == []


async def test_real_names_are_refused(world: SimpleNamespace) -> None:
    world.llm.replies.append({**BRIEF, "premise": "Riya teaches JARVIS to bake bread."})
    with pytest.raises(BriefRejected, match="Riya Sharma"):
        await world.service.request("riya and the bread")


async def test_one_reel_at_a_time(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    first = (await world.service.request("five more minutes")).episode
    with pytest.raises(ReelBusy) as busy:
        await world.service.request("another one")
    assert busy.value.episode.id == first.id
    assert world.llm.replies == []                     # refused before any model call


# -- reviewing ------------------------------------------------------------------

async def test_draft_reaches_the_owner_once(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _lease(world, episode.job_id)
    await world.service.tick()
    assert (await world.repo.get(episode.id)).status is EpisodeStatus.DRAFTING

    await world.jobs.transition(episode.job_id, JobStatus.RUNNING, JobStatus.QUEUED,
                                set_fields={"lease": None})
    await _draft(world, episode.job_id, episode.kit_id, 1)
    await world.service.tick()
    await world.service.tick()

    assert len(world.telegram.reviews) == 1
    review = world.telegram.reviews[0]
    assert review["sheet"].name.endswith("-r1-sheet.png")
    assert review["voice_name"].endswith("-r1-voice.m4a")
    assert "five more minutes" in review["text"]
    assert f"(ref {episode.ref})" in review["text"]

    loaded = await world.repo.get(episode.id)
    assert loaded.status is EpisodeStatus.AWAITING_REVIEW
    assert loaded.round == 1 and loaded.session_id == "sess-1"

    # The generic "Permission needed" message must not double up.
    assert await world.notifier.scan_pending_approvals() == 0


async def test_draft_warnings_are_shown(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _draft(world, episode.job_id, episode.kit_id, 1,
                 lines=["Riya says hello.", "Brought to you by Basement Supply."])
    await world.service.tick()
    text = world.telegram.reviews[0]["text"]
    assert "Check before approving" in text
    assert "Riya Sharma" in text and "Basement Supply" in text


async def test_notes_then_approve_write_the_checkpoint_first(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _draft(world, episode.job_id, episode.kit_id, 1)
    await world.service.tick()

    reply = await world.service.decide(episode.id, ReviewAction.NOTES, "Slower ending.")
    assert "Notes taken" in reply
    job = await world.jobs.get(episode.job_id)
    assert job.status is JobStatus.QUEUED
    assert job.checkpoint["decision"] == {"action": "notes", "round": 1,
                                          "notes": "Slower ending.", "prop_id": ""}
    loaded = await world.repo.get(episode.id)
    assert loaded.status is EpisodeStatus.DRAFTING and loaded.notes == "Slower ending."

    await _draft(world, episode.job_id, episode.kit_id, 2, title="Five more minutes, slower")
    await world.service.tick()
    assert len(world.telegram.reviews) == 2
    assert world.telegram.reviews[1]["sheet"].name.endswith("-r2-sheet.png")

    reply = await world.service.decide(episode.id, ReviewAction.APPROVE)
    assert "Approved" in reply
    job = await world.jobs.get(episode.job_id)
    assert job.checkpoint["decision"]["action"] == "approve"
    assert job.checkpoint["decision"]["round"] == 2
    assert (await world.repo.get(episode.id)).status is EpisodeStatus.RENDERING

    # A second tap on the old message settles nothing.
    assert "nothing to decide" in await world.service.decide(episode.id, ReviewAction.APPROVE)

    await _finish(world, episode.job_id, episode.kit_id)
    await world.service.tick()
    await world.service.tick()
    assert len(world.telegram.videos) == 1
    video, caption = world.telegram.videos[0]
    assert video.name == f"{episode.kit_id}.mp4"
    assert "1080x1920" in caption and f"/reel posted {episode.ref}" in caption
    delivered = await world.repo.get(episode.id)
    assert delivered.status is EpisodeStatus.DELIVERED

    assert "live" in await world.service.mark_posted(episode.ref)
    assert (await world.repo.get(episode.id)).status is EpisodeStatus.POSTED


async def test_cancel_from_review(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _draft(world, episode.job_id, episode.kit_id, 1)
    await world.service.tick()

    assert "Cancelled" in await world.service.decide(episode.id, ReviewAction.CANCEL)
    assert (await world.jobs.get(episode.job_id)).status is JobStatus.CANCELLED
    await world.service.tick()
    assert world.telegram.texts == []                  # his own cancel needs no report

    # The kit is free again.
    world.llm.replies.append(BRIEF)
    await world.service.request("another one")


async def test_reask_without_new_draft_is_not_resent(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _draft(world, episode.job_id, episode.kit_id, 1)
    await world.service.tick()

    # A generic approve (say, from /approvals) releases the job with no
    # decision; the Mac re-raises the gate for the same round.
    approval = await world.approvals.pending_for_job(episode.job_id)
    await world.approval_service.decide(approval.id, True)
    await _lease(world, episode.job_id)
    await world.approval_service.request(
        job_id=episode.job_id, gate=Gate.SPEND, actor="worker.reels", tool="reel_review",
        summary="again", detail="...")
    await world.service.tick()

    assert len(world.telegram.reviews) == 1
    loaded = await world.repo.get(episode.id)
    assert loaded.status is EpisodeStatus.AWAITING_REVIEW
    assert loaded.approval_id == (await world.approvals.pending_for_job(episode.job_id)).id
    assert "Approved" in await world.service.decide(episode.id, ReviewAction.APPROVE)


async def test_failure_is_reported_once(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _lease(world, episode.job_id)
    await world.jobs.transition(episode.job_id, JobStatus.RUNNING, JobStatus.FAILED,
                                set_fields={"error": "Claude Code is not logged in",
                                            "lease": None})
    await world.service.tick()
    await world.service.tick()
    assert len(world.telegram.texts) == 1
    assert "not logged in" in world.telegram.texts[0]
    assert (await world.repo.get(episode.id)).status is EpisodeStatus.FAILED


async def test_status_and_refs(world: SimpleNamespace) -> None:
    assert "No reels yet" in await world.service.status_text()
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    text = await world.service.status_text()
    assert episode.ref in text and "Mac is not connected" in text
    assert (await world.service.resolve("last")).id == episode.id
    assert isinstance(await world.service.resolve("zzzzzz"), str)


async def test_pitches_skip_rule_breakers(world: SimpleNamespace) -> None:
    world.llm.replies.append({"pitches": [
        {"episode_type": "shift_log", "title": "Night shift",
         "premise": "JARVIS logs a night where nothing happened, loudly."},
        {"episode_type": "life_event", "title": "Shop day",
         "premise": "JARVIS reviews a Basement Supply drop."},
    ]})
    pitches = await world.service.pitch()
    assert [p.title for p in pitches] == ["Night shift"]
    assert world.service.pitch_at(1).title == "Night shift"
    assert world.service.pitch_at(2) is None


def test_reel_buttons_fit_telegram() -> None:
    from jarvis.core.clients.telegram import _reel_keyboard
    keyboard = _reel_keyboard("01JABCDEFGHJKMNPQRSTVWXYZ0")
    for button in keyboard.inline_keyboard[0]:
        assert button.callback_data is not None
        assert len(button.callback_data.encode()) <= 64
    incomplete = _reel_keyboard("01JABCDEFGHJKMNPQRSTVWXYZ0", approvable=False)
    assert [b.text for b in incomplete.inline_keyboard[0]] == ["Notes", "Cancel"]



async def test_incomplete_draft_cannot_be_approved(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _draft(world, episode.job_id, episode.kit_id, 1)
    await world.service.tick()
    await world.service.decide(episode.id, ReviewAction.NOTES, "Add the staff.")

    await _draft(world, episode.job_id, episode.kit_id, 2, kind="incomplete")
    await world.service.tick()
    review = world.telegram.reviews[-1]
    assert review["approvable"] is False
    assert review["sheet"] is None                     # round 1's sheet is not this draft's
    assert "incomplete" in review["text"] and "npm run voice" in review["text"]

    reply = await world.service.decide(episode.id, ReviewAction.APPROVE)
    assert "incomplete" in reply
    assert (await world.repo.get(episode.id)).status is EpisodeStatus.AWAITING_REVIEW
    job = await world.jobs.get(episode.job_id)
    assert job.status is JobStatus.AWAITING_APPROVAL   # the gate is still closed

    assert "Notes taken" in await world.service.decide(
        episode.id, ReviewAction.NOTES, "Finish it.")


async def test_the_conversation_hears_about_reels(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _draft(world, episode.job_id, episode.kit_id, 1)
    await world.service.tick()
    await world.service.decide(episode.id, ReviewAction.APPROVE)
    await _finish(world, episode.job_id, episode.kit_id)
    await world.service.tick()

    session = await world.sessions.get_or_create_default("telegram")
    turns = await world.sessions.recent_turns(session.id, limit=20)
    said = [(t.role, t.content) for t in turns]
    assert said[0] == (TurnRole.USER, "/reel five more minutes")
    assert said[1][0] is TurnRole.ASSISTANT and episode.ref in said[1][1]
    assert any(role is TurnRole.ASSISTANT and "At one twelve a.m." in text
               for role, text in said)
    assert any(role is TurnRole.USER and text.startswith("Approved draft 1")
               for role, text in said)
    assert "finished film" in said[-1][1] and episode.ref in said[-1][1]

    context = await world.service.context_block()
    assert context.startswith(f"- ref {episode.ref}")
    assert "delivered to Sir, not yet posted" in context
    assert "At one twelve a.m., Sir said five more minutes." in context
    assert "30.2s 1080x1920" in context

    details = await world.service.describe("last")
    assert "Asked for: five more minutes" in details and "Beats:" in details


async def test_a_refused_tap_is_not_recorded(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes", record=False)).episode
    await _draft(world, episode.job_id, episode.kit_id, 1, kind="incomplete")
    await world.service.tick()
    await world.service.decide(episode.id, ReviewAction.APPROVE)

    session = await world.sessions.get_or_create_default("telegram")
    turns = await world.sessions.recent_turns(session.id, limit=20)
    assert [t.role for t in turns] == [TurnRole.ASSISTANT]   # the review, nothing else


async def test_lists_sent_as_text_are_accepted(world: SimpleNamespace) -> None:
    world.llm.replies.append({**BRIEF,
                              "beats": '["The promise", "The evidence", "The verdict"]',
                              "facts": "1. It was 1:12 a.m.\n- He said five more minutes"})
    episode = (await world.service.request("five more minutes")).episode
    assert episode.brief.beats == ["The promise", "The evidence", "The verdict"]
    assert episode.brief.facts == ["It was 1:12 a.m.", "He said five more minutes"]


async def test_a_malformed_brief_is_asked_for_again(world: SimpleNamespace) -> None:
    world.llm.replies += [{**BRIEF, "beats": 7}, BRIEF]
    episode = (await world.service.request("five more minutes")).episode
    assert episode.brief.beats == BRIEF["beats"]

    world.llm.replies += [{**BRIEF, "beats": 7}, {**BRIEF, "beats": 7}]
    await world.service.cancel(episode.ref)
    with pytest.raises(BriefRejected, match="beats"):
        await world.service.request("another one")



async def _repause(w: SimpleNamespace, job_id: str, **changes: Any) -> None:
    """What the worker does after a prop answer: same round, new gate."""
    await _lease(w, job_id)
    job = await w.jobs.get(job_id)
    checkpoint = ReelCheckpoint.model_validate(
        {**job.checkpoint, "decision": None, **changes})
    await w.jobs.set_checkpoint(job_id, checkpoint.model_dump(mode="json"))
    await w.approval_service.request(
        job_id=job_id, gate=Gate.SPEND, actor="worker.reels", tool="reel_review",
        summary="again", detail="...")


async def test_new_props_are_approved_before_the_draft(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    bike = {"id": "bike", "status": "draft", "description": "A red road bike."}
    await _draft(world, episode.job_id, episode.kit_id, 1, props=[bike])
    await world.service.tick()

    review = world.telegram.reviews[-1]
    assert [card.id for card in review["props"]] == ["bike"]
    assert review["props"][0].preview.name.endswith("-r1-prop-bike.png")
    assert "Approve them first" in review["text"]

    reply, moved = await world.service.answer(episode.id, ReviewAction.APPROVE)
    assert "Approve the new props first" in reply and not moved
    assert (await world.jobs.get(episode.job_id)).status is JobStatus.AWAITING_APPROVAL

    assert "no prop" in await world.service.decide(
        episode.id, ReviewAction.PROP_APPROVE, prop_id="lamp")
    reply, moved = await world.service.answer(
        episode.id, ReviewAction.PROP_APPROVE, prop_id="bike")
    assert moved and "Approving prop bike" in reply
    job = await world.jobs.get(episode.job_id)
    assert job.checkpoint["decision"] == {"action": "prop_approve", "round": 1,
                                          "notes": "", "prop_id": "bike"}
    assert "still on your last answer" in await world.service.decide(
        episode.id, ReviewAction.PROP_APPROVE, prop_id="bike")

    # The Mac approves it and asks again, same round, with a notice.
    await _repause(world, episode.job_id,
                   props=[{**bike, "status": "approved"}],
                   notice="Prop bike approved. Every prop is approved - approve the draft to render.")
    await world.service.tick()
    await world.service.tick()
    assert len(world.telegram.reviews) == 2                 # the notice, once, with buttons
    notice = world.telegram.reviews[-1]
    assert notice["text"].startswith("Prop bike approved") and notice["sheet"] is None
    assert (await world.repo.get(episode.id)).status is EpisodeStatus.AWAITING_REVIEW

    assert "Approved. Rendering" in await world.service.decide(episode.id, ReviewAction.APPROVE)


async def test_prop_notes_reach_the_checkpoint(world: SimpleNamespace) -> None:
    world.llm.replies.append(BRIEF)
    episode = (await world.service.request("five more minutes")).episode
    await _draft(world, episode.job_id, episode.kit_id, 1,
                 props=[{"id": "bike", "status": "draft", "description": "A bike."}])
    await world.service.tick()

    assert "Notes need words" in await world.service.decide(
        episode.id, ReviewAction.PROP_NOTES, "  ", prop_id="bike")
    reply = await world.service.decide(
        episode.id, ReviewAction.PROP_NOTES, "Make it blue.", prop_id="bike")
    assert "Notes on prop bike taken" in reply
    job = await world.jobs.get(episode.job_id)
    assert job.checkpoint["decision"]["action"] == "prop_notes"
    assert job.checkpoint["decision"]["notes"] == "Make it blue."
    assert (await world.repo.get(episode.id)).status is EpisodeStatus.DRAFTING


def test_prop_buttons_fit_telegram() -> None:
    from jarvis.core.clients.telegram import _prop_keyboard
    keyboard = _prop_keyboard("01JABCDEFGHJKMNPQRSTVWXYZ0", "a" + "b" * 30)
    for button in keyboard.inline_keyboard[0]:
        assert button.callback_data is not None
        assert len(button.callback_data.encode()) <= 64
