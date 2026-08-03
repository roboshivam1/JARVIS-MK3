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
# Commands: /start, /new, /status, /approvals
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
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from jarvis.common.log import get_logger
from jarvis.core.approvals.service import ApprovalService
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.initiative.policy import NotificationPolicy
from jarvis.core.initiative.policy import NotificationPolicy
from jarvis.core.initiative.policy import NotificationPolicy
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
        self._tz = tz or ZoneInfo("UTC")
        self._bot = Bot(token, default=DefaultBotProperties(parse_mode=None))
        self._dp = Dispatcher()
        self._dp.message.register(self._on_message)
        self._dp.callback_query.register(
            self._on_callback, F.data.startswith(("approve:", "reject:"))
        )

    async def run(self) -> None:
        """Long-poll until cancelled. Runs as one supervised Core task."""
        log.info("telegram bridge polling", extra={"owner_id": self._owner_id})
        try:
            await self._dp.start_polling(self._bot, handle_signals=False)
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

    # -- inbound: messages ----------------------------------------------------

    async def _on_message(self, message: Message) -> None:
        # Owner lock FIRST. Everyone else does not exist.
        if message.from_user is None or message.from_user.id != self._owner_id:
            log.warning("ignored non-owner message", extra={
                "from_id": message.from_user.id if message.from_user else None,
            })
            return

        text = (message.text or "").strip()
        if not text:
            await message.answer("Text only for now, sir.")
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

    async def _handle_command(self, message: Message, text: str) -> None:
        command = text.split()[0].lower()

        if command == "/start":
            await message.answer(
                "Online, sir. Speak freely - or /status for vitals, "
                "/approvals for anything awaiting your say-so, "
                "/new for a fresh thread."
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

        else:
            await message.answer(f"No such command, sir: {command}")

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
            body = f"{request.summary}\n\n{request.detail}"
            if request.risk_note:
                body += f"\n\nRisk: {request.risk_note}"
            await message.answer(
                body[:_TG_LIMIT], reply_markup=_approval_keyboard(request.id)
            )