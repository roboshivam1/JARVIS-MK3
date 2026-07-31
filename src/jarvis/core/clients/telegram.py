# =============================================================================
# src/jarvis/core/clients/telegram.py - the Telegram bridge
# =============================================================================
#
# A thin adapter between Telegram and the session manager. It renders and
# relays; it never thinks. All intelligence stays in the Core.
#
# Mechanics:
#   - Long-polling: the bridge asks Telegram for updates in a loop, so no
#     public address or webhook is needed. Works the same on a laptop and
#     a VPS.
#   - Streamed replies: Telegram cannot append to a message, so we send a
#     placeholder and EDIT it as text arrives. Edits are throttled to one
#     every ~1.2 s per reply (Telegram pushes back around 1/s), driven by
#     a small flusher task; a final edit delivers the complete text.
#   - Owner lock: the FIRST check on every update is the sender's numeric
#     id. Strangers get silence and a log line. A configured bot token
#     without an owner id refuses to start at all.
#
# Commands:
#   /start  - greeting and a liveness hint
#   /new    - archive the current thread; next message opens a fresh one
#   /status - the same snapshot the HTTP /status route serves
#
# Telegram caps messages at 4096 chars. The stream buffer edits the first
# chunk; if a finished reply is longer, the remainder is sent as follow-up
# messages at the end.
# =============================================================================

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import FSInputFile, Message
from jarvis.common.log import get_logger
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.sessionmgr import SessionManager

log = get_logger("core.telegram")

CLIENT_KIND = "telegram"
_EDIT_INTERVAL_S = 1.2
_TG_LIMIT = 4096
_STREAM_CAP = 4000          # headroom under the hard limit while streaming

# Provided by the gateway module: computes the shared status snapshot.
StatusProvider = Callable[[], Awaitable[dict[str, Any]]]


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
        """The callback handed to the session manager; called per chunk."""
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
                pass   # interval elapsed - loop and flush again

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
            # Rendering must never kill the turn; worst case the final
            # flush delivers everything in one go.
            log.warning("stream flush failed", exc_info=True)

    async def finalize(self, full_text: str) -> None:
        """Stop the metronome and deliver the complete reply."""
        self._done.set()
        if self._flusher is not None:
            await self._flusher

        head, rest = full_text[:_STREAM_CAP], full_text[_STREAM_CAP:]
        async with self._lock:
            self._buffer = head
        await self._flush(partial=False)

        # Anything beyond the first message's cap goes out as follow-ups.
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
        self._bot = Bot(token, default=DefaultBotProperties(parse_mode=None))
        self._dp = Dispatcher()
        self._dp.message.register(self._on_message)

    async def deliver(
        self,
        text: str,
        file_path: Path | None = None,
        file_name: str | None = None,
    ) -> None:
        """Deliverer implementation: push an unprompted message (and
        optionally a file) to the owner. Called only by the notifier -
        nothing else may message the owner unprompted.

        Files stream from disk via FSInputFile rather than loading into
        memory, so a large artifact costs the same as a small one.
        """
        await self._bot.send_message(self._owner_id, text[:_TG_LIMIT])
        if file_path is not None and file_path.exists():
            await self._bot.send_document(
                self._owner_id,
                FSInputFile(file_path, filename=file_name or file_path.name),
            )

    async def run(self) -> None:
        """Long-poll until cancelled. Runs as one supervised Core task."""
        log.info("telegram bridge polling", extra={"owner_id": self._owner_id})
        try:
            await self._dp.start_polling(
                self._bot,
                handle_signals=False,   # the Core owns signal handling
            )
        finally:
            await self._bot.session.close()

    # -- update handling ------------------------------------------------------

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

        # A normal conversational turn, streamed back via edits.
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
            # A newer message took over; that turn will speak for itself.
            return
        assert result.reply is not None
        await reply.finalize(result.reply.content)

    async def _handle_command(self, message: Message, text: str) -> None:
        command = text.split()[0].lower()

        if command == "/start":
            await message.answer(
                "Online, sir. Speak freely - or /status for vitals, "
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

        else:
            await message.answer(f"No such command, sir: {command}")