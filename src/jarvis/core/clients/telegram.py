# =============================================================================
# src/jarvis/core/clients/telegram.py - the Telegram bridge
# =============================================================================
#
# A thin adapter between Telegram and the Core. It renders and relays; it
# never thinks. All intelligence stays in the Core.
#
# Mechanics:
#   - Long-polling: no public address or webhook needed, so the bridge
#     works identically on a laptop and a VPS.
#   - Streamed replies: Telegram cannot append to a message, so a
#     placeholder is sent and EDITED as text arrives, throttled to one
#     edit per ~1.2 s (Telegram pushes back near 1/s).
#   - Owner lock: the FIRST check on every update is the sender's numeric
#     id. Strangers get silence and a log line.
#
# APPROVALS: gated actions arrive as messages with inline buttons. A tap
# sends a small hidden payload (approve:<ulid> / reject:<ulid>) rather
# than a visible message - typing a job id on a phone is how mistakes
# happen. Once answered, the message is edited to show the decision and
# the keyboard is REMOVED: a live approve button on a settled question
# invites a second tap that silently does nothing.
#
# Commands: /start, /new, /status, /approvals, /quiet, /reel, /skip
#
# REELS: drafts arrive as the contact sheet, the voice track and the
# script, with Approve / Notes / Cancel. The buttons name the EPISODE,
# not the approval gate, so they keep working if the gate is re-raised.
# Notes are whatever the owner sends next, typed or spoken.
# =============================================================================

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramUnauthorizedError
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from jarvis.common.ids import new_ulid
from jarvis.common.log import get_logger
from jarvis.common.reels import EPISODE_TYPE_LABELS, REEL_REVIEW_TOOL, ReviewAction
from jarvis.core.approvals.service import ApprovalService
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.schedules import SchedulesRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.initiative.brief import BRIEF_SCHEDULE, BriefService
from jarvis.core.initiative.engine import next_cron_time
from jarvis.core.life.service import LifeService
from jarvis.core.initiative.policy import NotificationPolicy
from jarvis.llm.transcription import Transcriber
from jarvis.core.initiative.policy import NotificationPolicy
from jarvis.core.initiative.policy import NotificationPolicy
from jarvis.core.reels.service import BriefRejected, PropCard, ReelBusy, ReelService
from jarvis.core.sessionmgr import SessionManager

log = get_logger("core.telegram")

CLIENT_KIND = "telegram"
_EDIT_INTERVAL_S = 1.2
_TG_LIMIT = 4096
_STREAM_CAP = 4000

StatusProvider = Callable[[], Awaitable[dict[str, Any]]]


def _approval_keyboard(approval_id: str) -> InlineKeyboardMarkup:
    """Approve / reject buttons for one request.

    callback_data is capped at 64 bytes by Telegram; an action word plus
    a 26-character ULID fits with room to spare - one more dividend from
    short sortable ids.
    """
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Approve", callback_data=f"approve:{approval_id}"),
        InlineKeyboardButton(text="Reject", callback_data=f"reject:{approval_id}"),
    ]])


def _reel_keyboard(episode_id: str, approvable: bool = True) -> InlineKeyboardMarkup:
    """Approve / Notes / Cancel for one reel draft - without Approve when
    the draft is incomplete and there is nothing to render. 'reel:approve:'
    plus a ULID is 39 bytes, inside Telegram's 64."""
    buttons = [
        InlineKeyboardButton(text="Notes", callback_data=f"reel:notes:{episode_id}"),
        InlineKeyboardButton(text="Cancel", callback_data=f"reel:cancel:{episode_id}"),
    ]
    if approvable:
        buttons.insert(0, InlineKeyboardButton(
            text="Approve", callback_data=f"reel:approve:{episode_id}"))
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def _prop_keyboard(episode_id: str, prop_id: str) -> InlineKeyboardMarkup:
    """Approve prop / Notes / Cancel on one new prop's card. 'rp:a:' plus
    a ULID and a prop id of up to 31 characters is 63 bytes, inside 64."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Approve prop", callback_data=f"rp:a:{episode_id}:{prop_id}"),
        InlineKeyboardButton(text="Notes", callback_data=f"rp:n:{episode_id}:{prop_id}"),
        InlineKeyboardButton(text="Cancel", callback_data=f"reel:cancel:{episode_id}"),
    ]])


class _StreamingReply:
    """One growing Telegram message: buffer chunks, flush edits on a
    metronome, finalise with the full text (splitting if oversized)."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._buffer = ""
        self._sent_text = ""
        self._message_id: int | None = None
        self._lock = asyncio.Lock()
        self._flusher: asyncio.Task[None] | None = None
        self._done = asyncio.Event()

    async def on_text(self, chunk: str) -> None:
        async with self._lock:
            self._buffer += chunk
        if self._flusher is None:
            self._flusher = asyncio.create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        while not self._done.is_set():
            await self._flush(partial=True)
            try:
                await asyncio.wait_for(self._done.wait(), timeout=_EDIT_INTERVAL_S)
            except asyncio.TimeoutError:
                pass

    async def _flush(self, partial: bool) -> None:
        async with self._lock:
            text = self._buffer[:_STREAM_CAP] if partial else self._buffer
        if not text or text == self._sent_text:
            return
        try:
            if self._message_id is None:
                sent = await self._bot.send_message(self._chat_id, text)
                self._message_id = sent.message_id
            else:
                await self._bot.edit_message_text(
                    text, chat_id=self._chat_id, message_id=self._message_id
                )
            self._sent_text = text
        except Exception:
            log.warning("stream flush failed", exc_info=True)

    async def finalize(self, full_text: str) -> None:
        self._done.set()
        if self._flusher is not None:
            await self._flusher

        head, rest = full_text[:_STREAM_CAP], full_text[_STREAM_CAP:]
        async with self._lock:
            self._buffer = head
        await self._flush(partial=False)

        while rest:
            piece, rest = rest[:_TG_LIMIT], rest[_TG_LIMIT:]
            await self._bot.send_message(self._chat_id, piece)


class TelegramBridge:
    """Owns the bot, the dispatcher, and the polling task."""

    def __init__(
        self,
        token: str,
        owner_id: int,
        session_mgr: SessionManager,
        sessions_repo: SessionsRepo,
        status_provider: StatusProvider,
        approval_service: ApprovalService | None = None,
        approvals_repo: ApprovalsRepo | None = None,
        policy: "NotificationPolicy | None" = None,
        tz: "ZoneInfo | None" = None,
        transcriber: "Transcriber | None" = None,
        reels: "ReelService | None" = None,
        brief: "BriefService | None" = None,
        schedules: "SchedulesRepo | None" = None,
        life: "LifeService | None" = None,
    ) -> None:
        if owner_id <= 0:
            raise ValueError(
                "telegram bot token configured without an owner id - "
                "refusing to start an open bot"
            )
        self._owner_id = owner_id
        self._mgr = session_mgr
        self._sessions = sessions_repo
        self._status = status_provider
        self._approval_service = approval_service
        self._approvals = approvals_repo
        self._policy = policy
        self._transcriber = transcriber
        self._reels = reels
        self._brief = brief
        self._schedules = schedules
        self._life = life
        # Set by a Notes tap: (episode id, chat id, message id, prop id -
        # empty for the draft itself). The next thing the owner sends
        # becomes the notes for that draft or prop.
        self._awaiting_notes: tuple[str, int, int, str] | None = None
        self._tz = tz or ZoneInfo("UTC")
        self._bot = Bot(token, default=DefaultBotProperties(parse_mode=None))
        self._dp = Dispatcher()
        self._dp.message.register(self._on_message)
        self._dp.callback_query.register(
            self._on_callback, F.data.startswith(("approve:", "reject:"))
        )
        self._dp.callback_query.register(
            self._on_reel_callback, F.data.startswith("reel:")
        )
        self._dp.callback_query.register(
            self._on_prop_callback, F.data.startswith("rp:")
        )

    async def run(self) -> None:
        """Long-poll until cancelled, reconnecting through network trouble.

        A dropped wifi connection, a DNS hiccup, or Telegram having a bad
        minute must NOT take down the daemon. The Core has duties -
        memory, the job queue, the scheduler, the worker link, the web
        client - that owe nothing to Telegram being reachable.

        Same distinction the job queue already draws between transient
        and permanent failure, applied to a resident: retry what
        retrying can fix. A bad token still kills the bridge, because no
        amount of waiting repairs that.
        """
        log.info("telegram bridge polling", extra={"owner_id": self._owner_id})
        backoff = 5.0
        try:
            while True:
                try:
                    await self._dp.start_polling(self._bot, handle_signals=False)
                    return          # clean stop
                except asyncio.CancelledError:
                    raise
                except TelegramUnauthorizedError:
                    log.critical("telegram rejected the bot token")
                    raise
                except Exception as exc:
                    log.warning("telegram polling failed - retrying", extra={
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                        "retry_in_s": backoff,
                    })
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 300.0)
        finally:
            await self._bot.session.close()

    # -- outbound (the notifier's delivery surface) ---------------------------

    async def deliver(
        self,
        text: str,
        file_path: Path | None = None,
        file_name: str | None = None,
        approval_id: str | None = None,
    ) -> None:
        """Push an unprompted message to the owner, with buttons when it
        is a question. Called only by the notifier."""
        keyboard = _approval_keyboard(approval_id) if approval_id else None
        await self._bot.send_message(
            self._owner_id, text[:_TG_LIMIT], reply_markup=keyboard
        )
        if file_path is not None and file_path.exists():
            await self._bot.send_document(
                self._owner_id,
                FSInputFile(file_path, filename=file_name or file_path.name),
            )

    # -- outbound: reels (the reel service's delivery surface) ----------------

    async def send_reel_review(
        self,
        episode_id: str,
        text: str,
        sheet: Path | None,
        voice: Path | None,
        voice_name: str | None,
        approvable: bool = True,
        props: list[PropCard] | None = None,
    ) -> None:
        """A draft for review: any new props first, each on its own card,
        then what it looks like, what it sounds like, then the script with
        the buttons, so the buttons arrive last."""
        for card in props or []:
            caption = f"New prop: {card.id}\n{card.description}"[:1024]
            keyboard = _prop_keyboard(episode_id, card.id)
            if card.preview is not None and card.preview.exists():
                try:
                    await self._bot.send_photo(
                        self._owner_id, FSInputFile(card.preview),
                        caption=caption, reply_markup=keyboard,
                    )
                    continue
                except Exception:
                    log.warning("prop preview would not send as a photo", exc_info=True)
            await self._bot.send_message(
                self._owner_id, caption + "\n(no preview came through)",
                reply_markup=keyboard,
            )
        if sheet is not None and sheet.exists():
            try:
                await self._bot.send_photo(
                    self._owner_id, FSInputFile(sheet), caption="Contact sheet"
                )
            except Exception:
                # Too large or too tall for a photo: a document always works.
                await self._bot.send_document(
                    self._owner_id, FSInputFile(sheet, filename=sheet.name)
                )
        if voice is not None and voice.exists():
            name = voice_name or voice.name
            if name.endswith(".m4a"):
                await self._bot.send_audio(
                    self._owner_id, FSInputFile(voice, filename=name), title="Voice"
                )
            else:
                await self._bot.send_document(
                    self._owner_id, FSInputFile(voice, filename=name)
                )
        await self._bot.send_message(
            self._owner_id, text[:_TG_LIMIT],
            reply_markup=_reel_keyboard(episode_id, approvable),
        )

    async def send_reel_video(self, video: Path, caption: str) -> None:
        await self._bot.send_video(
            self._owner_id, FSInputFile(video, filename=video.name),
            caption=caption[:1024], supports_streaming=True,
        )

    async def send_reel_text(self, text: str) -> None:
        await self._bot.send_message(self._owner_id, text[:_TG_LIMIT])

    # -- inbound: taps --------------------------------------------------------

    async def _on_callback(self, callback: CallbackQuery) -> None:
        """A button was tapped."""
        if callback.from_user.id != self._owner_id:
            await callback.answer("Not for you.", show_alert=True)
            log.warning("ignored non-owner callback", extra={
                "from_id": callback.from_user.id,
            })
            return

        if self._approval_service is None or not callback.data:
            await callback.answer("Approvals are not available.", show_alert=True)
            return

        action, _, approval_id = callback.data.partition(":")
        approve = action == "approve"
        decided = await self._approval_service.decide(approval_id, approve=approve)

        if not decided:
            # Already answered, or expired while the owner was deciding.
            await callback.answer("Already settled, sir.", show_alert=True)
        else:
            await callback.answer("Approved." if approve else "Rejected.")

        # Remove the buttons and record the outcome in the message
        # itself: a live button on a settled question invites a tap that
        # does nothing, and confusion is expensive in a safety interface.
        if callback.message is not None:
            verdict = "APPROVED" if approve else "REJECTED"
            suffix = f"\n\n--- {verdict} ---" if decided else "\n\n--- already settled ---"
            try:
                await self._bot.edit_message_text(
                    (callback.message.text or "")[:_TG_LIMIT - 40] + suffix,
                    chat_id=callback.message.chat.id,
                    message_id=callback.message.message_id,
                    reply_markup=None,
                )
            except Exception:
                log.warning("could not update approval message", exc_info=True)

    async def _on_reel_callback(self, callback: CallbackQuery) -> None:
        """Approve / Notes / Cancel on a reel draft."""
        if callback.from_user.id != self._owner_id:
            await callback.answer("Not for you.", show_alert=True)
            return
        if self._reels is None or not callback.data:
            await callback.answer("Reels are not available.", show_alert=True)
            return

        _, action, episode_id = callback.data.split(":", 2)
        message = callback.message if isinstance(callback.message, Message) else None

        if action == "notes":
            if message is None:
                await callback.answer("Use /reel notes <ref> <notes>.", show_alert=True)
                return
            self._awaiting_notes = (episode_id, message.chat.id, message.message_id, "")
            await callback.answer()
            await self._bot.send_message(
                self._owner_id,
                "Your notes, sir - type them or send a voice note. "
                "/skip to keep chatting instead.",
            )
            return

        try:
            reply, moved = await self._reels.answer(episode_id, ReviewAction(action))
        except ValueError:
            await callback.answer("Unknown action.", show_alert=True)
            return
        await callback.answer(reply[:200], show_alert=not moved)
        if message is not None and moved:
            await self._close_reel_message(message.chat.id, message.message_id,
                                           message.text or "", reply)

    async def _on_prop_callback(self, callback: CallbackQuery) -> None:
        """Approve prop / Notes on a new prop's card. The card keeps its
        buttons: if the Mac could not approve it, Sir can tap again."""
        if callback.from_user.id != self._owner_id:
            await callback.answer("Not for you.", show_alert=True)
            return
        if self._reels is None or not callback.data:
            await callback.answer("Reels are not available.", show_alert=True)
            return
        parts = callback.data.split(":")
        if len(parts) != 4 or parts[1] not in ("a", "n"):
            await callback.answer("Unknown action.", show_alert=True)
            return
        _, verb, episode_id, prop_id = parts
        message = callback.message if isinstance(callback.message, Message) else None

        if verb == "n":
            if message is None:
                await callback.answer("This card is too old to take notes.", show_alert=True)
                return
            self._awaiting_notes = (episode_id, message.chat.id, message.message_id, prop_id)
            await callback.answer()
            await self._bot.send_message(
                self._owner_id,
                f"What should change about the {prop_id}, sir? Type it or send a "
                f"voice note. /skip to keep chatting instead.",
            )
            return

        reply, moved = await self._reels.answer(
            episode_id, ReviewAction.PROP_APPROVE, prop_id=prop_id)
        await callback.answer(reply[:200], show_alert=not moved)

    async def _take_reel_notes(self, message: Message, text: str) -> None:
        """The owner's next message after a Notes tap, on a draft or a prop."""
        assert self._awaiting_notes is not None and self._reels is not None
        episode_id, chat_id, message_id, prop_id = self._awaiting_notes
        self._awaiting_notes = None
        if prop_id:
            reply, moved = await self._reels.answer(
                episode_id, ReviewAction.PROP_NOTES, text, prop_id=prop_id)
        else:
            reply, moved = await self._reels.answer(episode_id, ReviewAction.NOTES, text)
        await message.answer(reply)
        if moved:
            await self._close_reel_message(chat_id, message_id, None, reply)

    async def _close_reel_message(
        self, chat_id: int, message_id: int, original: str | None, verdict: str
    ) -> None:
        """Strip the buttons off a settled draft, like any settled gate."""
        try:
            if original is None:
                await self._bot.edit_message_reply_markup(
                    chat_id=chat_id, message_id=message_id, reply_markup=None
                )
            else:
                await self._bot.edit_message_text(
                    original[:_TG_LIMIT - 120] + f"\n\n--- {verdict[:100]} ---",
                    chat_id=chat_id, message_id=message_id, reply_markup=None,
                )
        except Exception:
            log.warning("could not update reel message", exc_info=True)

    async def _reel_command(self, message: Message, text: str) -> None:
        if self._reels is None:
            await message.answer("Reels are not configured on this Core, sir.")
            return
        args = text.partition(" ")[2].strip()
        sub, _, rest = args.partition(" ")
        sub = sub.lower()
        rest = rest.strip()

        if not args:
            await message.answer(
                "/reel <idea> - brief one and start drafting\n"
                "/reel pitch - three ideas; /reel pick 2 to make one\n"
                "/reel status - the episode log\n"
                "/reel notes <ref> <notes> - notes without the button\n"
                "/reel cancel <ref> - drop one\n"
                "/reel posted <ref> - mark one as live"
            )
        elif sub == "pitch":
            thinking = await message.answer("Thinking of three...")
            try:
                pitches = await self._reels.pitch()
            except Exception as exc:
                log.error("reel pitch failed", exc_info=True)
                await thinking.edit_text(f"No pitches, sir: {exc}")
                return
            if not pitches:
                await thinking.edit_text("Nothing usable came back, sir. Ask again.")
                return
            body = "\n\n".join(
                f"{i}. {p.title} ({EPISODE_TYPE_LABELS[p.episode_type]})\n{p.premise}"
                for i, p in enumerate(pitches, start=1)
            )
            await thinking.edit_text(f"{body}\n\n/reel pick <number> to make one.")
        elif sub == "pick":
            pitch = self._reels.pitch_at(int(rest)) if rest.isdigit() else None
            if pitch is None:
                await message.answer("No such pitch, sir. /reel pitch first.")
                return
            await self._request_reel(message, f"{pitch.title}: {pitch.premise}")
        elif sub == "status":
            await message.answer(await self._reels.status_text())
        elif sub == "posted":
            await message.answer(await self._reels.mark_posted(rest or "last"))
        elif sub == "cancel":
            await message.answer(await self._reels.cancel(rest or "last"))
        elif sub == "notes":
            ref, _, notes = rest.partition(" ")
            found = await self._reels.resolve(ref)
            if isinstance(found, str):
                await message.answer(found)
                return
            await message.answer(
                await self._reels.decide(found.id, ReviewAction.NOTES, notes)
            )
        else:
            await self._request_reel(message, args)

    async def _request_reel(self, message: Message, idea: str) -> None:
        assert self._reels is not None
        status = await message.answer("Briefing...")
        try:
            outcome = await self._reels.request(idea)
        except ReelBusy as busy:
            e = busy.episode
            await status.edit_text(
                f"One reel at a time, sir: \u201c{e.title or e.request[:40]}\u201d is "
                f"{e.status.value.replace('_', ' ')}. /reel cancel {e.ref} to drop it."
            )
            return
        except BriefRejected as exc:
            await status.edit_text(f"I will not brief that one, sir: {exc}")
            return
        except Exception as exc:
            log.error("reel request failed", exc_info=True)
            await status.edit_text(f"The brief failed, sir: {type(exc).__name__}.")
            return

        episode = outcome.episode
        reply = (
            f"Briefed - {EPISODE_TYPE_LABELS[episode.type]}.\n"
            f"{episode.brief.premise}\n\n"
            f"Drafting on the Mac; script, stills and voice follow for your "
            f"review.  (ref {episode.ref})"
        )
        if not outcome.worker_online:
            reply += "\n\nThe Mac is not connected - it starts when it is."
        await status.edit_text(reply)

    # -- inbound: messages ----------------------------------------------------

    async def _on_message(self, message: Message) -> None:
        # Owner lock FIRST. Everyone else does not exist.
        if message.from_user is None or message.from_user.id != self._owner_id:
            log.warning("ignored non-owner message", extra={
                "from_id": message.from_user.id if message.from_user else None,
            })
            return

        # A voice note becomes an ordinary text turn: transcribe it, then
        # hand the words on. Nothing downstream - memory, jobs,
        # subagents - knows or cares that it was spoken.
        if message.voice is not None:
            await self._handle_voice(message)
            return

        text = (message.text or "").strip()
        if not text:
            await message.answer(
                "Text or a voice note, sir - I cannot read the rest yet."
            )
            return

        if self._awaiting_notes is not None and not text.startswith("/"):
            await self._take_reel_notes(message, text)
            return

        if text.startswith("/"):
            await self._handle_command(message, text)
            return

        reply = _StreamingReply(self._bot, message.chat.id)
        try:
            result = await self._mgr.handle_user_message(
                CLIENT_KIND, text, on_text=reply.on_text
            )
        except Exception as exc:
            log.error("turn failed", exc_info=True)
            await reply.finalize(
                f"Something broke while I was thinking, sir: "
                f"{type(exc).__name__}. The details are in my logs."
            )
            return

        if result.interrupted:
            return
        assert result.reply is not None
        await reply.finalize(result.reply.content)

    async def _handle_voice(self, message: Message) -> None:
        """Transcribe a voice note and treat it as a message.

        Telegram sends OGG/Opus, which Whisper accepts directly - no
        conversion, no ffmpeg, nothing in the latency budget.

        The transcript comes back WITH the reply. Without it, a
        misheard question produces an answer to something the owner
        never asked, and he has no way to tell which happened.
        """
        if self._transcriber is None or not self._transcriber.available:
            await message.answer(
                "I cannot transcribe yet, sir - no speech key configured."
            )
            return

        assert message.voice is not None
        if message.voice.duration > 300:
            await message.answer(
                "That is over five minutes, sir. Send it in pieces."
            )
            return

        thinking = await message.answer("Listening...")

        try:
            file = await self._bot.get_file(message.voice.file_id)
            if file.file_path is None:
                raise RuntimeError("telegram returned no file path")
            buffer = await self._bot.download_file(file.file_path)
            audio = buffer.read() if buffer else b""
        except Exception:
            log.error("could not fetch voice note", exc_info=True)
            await thinking.edit_text("I could not fetch that recording, sir.")
            return

        text = await self._transcriber.transcribe(audio, mime="audio/ogg")
        if not text:
            await thinking.edit_text("I did not catch that, sir.")
            return

        # Show what was heard before the answer, so a mishearing is
        # obvious rather than confusing.
        await thinking.edit_text(f"\u201c{text}\u201d")

        if self._awaiting_notes is not None:
            await self._take_reel_notes(message, text)
            return

        reply = _StreamingReply(self._bot, message.chat.id)
        try:
            result = await self._mgr.handle_user_message(
                CLIENT_KIND, text, on_text=reply.on_text
            )
        except Exception as exc:
            log.error("voice turn failed", exc_info=True)
            await reply.finalize(
                f"Something broke while I was thinking, sir: "
                f"{type(exc).__name__}."
            )
            return

        if result.interrupted:
            return
        assert result.reply is not None
        await reply.finalize(result.reply.content)

    async def _handle_command(self, message: Message, text: str) -> None:
        command = text.split()[0].lower()

        if command == "/start":
            await message.answer(
                "Online, sir. Type or send a voice note - either works. "
                "/status for vitals, /approvals for anything awaiting "
                "your say-so, /quiet to silence me, /new for a fresh "
                "thread, /reel for the show, /brief for the morning brief, "
                "/today for today's checklist, /undo to take back the last log."
            )

        elif command == "/new":
            session = await self._sessions.get_or_create_default(CLIENT_KIND)
            await self._sessions.archive(session.id)
            await message.answer("Thread archived. Clean slate, sir.")

        elif command == "/status":
            s = await self._status()
            hours, rem = divmod(s["uptime_s"], 3600)
            minutes = rem // 60
            await message.answer(
                f"Core up {hours}h {minutes}m. "
                f"Today: ${s['cost_today_usd']} across {s['llm_calls']} model calls. "
                f"{s['sessions']} sessions, {s['turns']} turns on record."
            )

        elif command == "/approvals":
            await self._list_approvals(message)

        elif command == "/quiet":
            await self._snooze(message, text)

        elif command == "/reel":
            await self._reel_command(message, text)

        elif command == "/today":
            if self._life is None:
                await message.answer("The life log is not available in this build.")
            else:
                await message.answer(await self._life.status_text())

        elif command == "/undo":
            if self._life is None:
                await message.answer("The life log is not available in this build.")
            else:
                await message.answer(await self._life.undo_last())

        elif command == "/brief":
            await self._brief_command(message, text)

        elif command == "/skip":
            self._awaiting_notes = None
            await message.answer("Back to conversation, sir.")

        else:
            await message.answer(f"No such command, sir: {command}")

    async def _brief_command(self, message: Message, text: str) -> None:
        """/brief - the brief now. /brief off | on - stop or restart the
        daily one. /brief at 8:15 - move it."""
        if self._brief is None:
            await message.answer("The morning brief is not available in this build.")
            return
        parts = text.split()
        arg = parts[1].lower() if len(parts) > 1 else ""

        if not arg or arg == "now":
            await self._bot.send_chat_action(message.chat.id, "typing")
            reply = await self._brief.on_demand(new_ulid())
            for start in range(0, len(reply), _TG_LIMIT):
                await message.answer(reply[start:start + _TG_LIMIT])
            return

        if self._schedules is None:
            await message.answer("Schedules are not available in this build.")
            return
        schedule = await self._schedules.by_name(BRIEF_SCHEDULE)
        if schedule is None or not schedule.cron_expr:
            await message.answer("There is no morning brief schedule to change, sir.")
            return

        if arg in ("off", "on"):
            enabled = arg == "on"
            await self._schedules.set_enabled(
                BRIEF_SCHEDULE, enabled,
                next_cron_time(schedule.cron_expr, self._tz) if enabled else None,
            )
            await message.answer(
                f"The morning brief is back on, sir - {_cron_clock(schedule.cron_expr)} daily."
                if enabled else
                "No more morning briefs until you say /brief on, sir. "
                "/brief still gets you one whenever you like."
            )
            return

        if arg == "at" and len(parts) > 2:
            match = re.fullmatch(r"(\d{1,2})(?:[:.](\d{2}))?", parts[2])
            hour = int(match.group(1)) if match else 99
            minute = int(match.group(2) or 0) if match else 99
            if hour > 23 or minute > 59:
                await message.answer("Usage: /brief at 8:15 (24-hour clock).")
                return
            cron = f"{minute} {hour} * * *"
            await self._schedules.set_cron(
                BRIEF_SCHEDULE, cron, next_cron_time(cron, self._tz))
            note = ""
            if self._policy is not None and self._policy.is_quiet_hour(
                    datetime.now(self._tz).replace(hour=hour, minute=minute)):
                note = (" That is inside quiet hours, so it will wait until "
                        f"{self._policy.quiet_end.strftime('%H:%M')}.")
            state = "" if schedule.enabled else " It is switched off - /brief on to restart it."
            await message.answer(
                f"Morning brief moved to {hour:02d}:{minute:02d}, sir.{note}{state}")
            return

        await message.answer(
            "Usage: /brief for one now, /brief off | on, /brief at 8:15.")

    async def _snooze(self, message: Message, text: str) -> None:
        """Silence everything but urgent for a while.

        Runtime state, not persisted: a snooze that survives a restart
        is a snooze the owner forgets he set, and then wonders why
        JARVIS has gone quiet.
        """
        if self._policy is None:
            await message.answer("Notification policy is not available.")
            return

        parts = text.split()
        duration = parts[1] if len(parts) > 1 else "2h"

        if duration.lower() == "off":
            self._policy.snooze_until = None
            await message.answer("Listening again, sir.")
            return
        match = re.fullmatch(r"(\d+)([hm])", duration.lower())
        if not match:
            await message.answer(
                "Usage: /quiet 2h, or /quiet 30m. /quiet off to cancel."
            )
            return

        amount, unit = int(match.group(1)), match.group(2)
        delta = (
            timedelta(hours=amount) if unit == "h" else timedelta(minutes=amount)
        )
        until = datetime.now(timezone.utc) + delta
        self._policy.snooze_until = until

        local = until.astimezone(self._tz).strftime("%H:%M")
        await message.answer(
            f"Quiet until {local}, sir. Approvals will still reach you - "
            f"work stops without them."
        )

    async def _list_approvals(self, message: Message) -> None:
        """Re-present anything still waiting - useful when a request was
        scrolled past, or its message was tapped and lost."""
        if self._approvals is None:
            await message.answer("Approvals are not available in this build.")
            return

        pending = await self._approvals.all_pending()
        if not pending:
            await message.answer("Nothing awaiting your say-so, sir.")
            return

        for request in pending:
            if request.tool == REEL_REVIEW_TOOL:
                # Reel drafts have their own buttons; a generic approve
                # would only make the Mac ask again.
                await message.answer(
                    f"{request.summary}\n\nAnswer it on the draft message, "
                    f"or /reel status."
                )
                continue
            body = f"{request.summary}\n\n{request.detail}"
            if request.risk_note:
                body += f"\n\nRisk: {request.risk_note}"
            await message.answer(
                body[:_TG_LIMIT], reply_markup=_approval_keyboard(request.id)
            )


def _cron_clock(cron_expr: str) -> str:
    """"35 7 * * *" -> "07:35"; anything fancier is shown as the rule."""
    fields = cron_expr.split()
    if len(fields) == 5 and fields[0].isdigit() and fields[1].isdigit():
        return f"{int(fields[1]):02d}:{int(fields[0]):02d}"
    return cron_expr
