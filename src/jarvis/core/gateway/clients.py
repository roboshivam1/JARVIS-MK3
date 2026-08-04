# =============================================================================
# src/jarvis/core/gateway/clients.py - the client WebSocket endpoint
# =============================================================================
#
# One coroutine per connected client. Authenticates, joins a session,
# relays messages to the session manager, and streams replies back.
#
# SESSIONS ARE SHARED ACROSS SURFACES, by owner decision. The web client
# and Telegram join the SAME conversation, so a thought started at the
# desk continues on the phone with full context. One JARVIS, many
# windows - the alternative (a session per surface) means explaining
# yourself twice.
#
# The connection registry lets the notifier push to live clients, so an
# open browser tab gets a finished brief without polling. A client that
# is not connected misses nothing: the notification stays in the outbox
# and Telegram delivers it, which is the point of having an outbox.
# =============================================================================

from __future__ import annotations

import asyncio
import base64
import secrets
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from jarvis.common.client_protocol import (
    CLIENT_AUDIO_CHUNK,
    CLIENT_AUDIO_END,
    CLIENT_COMMAND,
    CLIENT_HELLO,
    CLIENT_INTERRUPT,
    CLIENT_USER_MESSAGE,
    CORE_ASSISTANT_DELTA,
    CORE_ASSISTANT_DONE,
    CORE_ERROR,
    CORE_NOTIFICATION,
    CORE_READY,
    CORE_STATUS,
    ClientCommand,
    ClientHello,
    ClientUserMessage,
    CoreAssistantDelta,
    CoreAssistantDone,
    CoreError,
    CoreNotification,
    CoreReady,
    CoreStatus,
    CoreTranscript,
    ClientAudioChunk,
)
from jarvis.common.client_protocol import (
    CORE_AUDIO,
    CORE_AUDIO_DONE,
    CORE_TRANSCRIPT,
    CoreAudio,
    CoreAudioDone,
)
from jarvis.common.envelope import Envelope, UnknownKind, make_envelope
from jarvis.common.log import get_logger
from jarvis.core.approvals.service import ApprovalService
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.sessionmgr import SessionManager
from jarvis.llm.speech import SentenceBuffer, SpeechBackend
from jarvis.llm.transcription import Transcriber

log = get_logger("core.gateway.clients")

# The session all interactive clients share. Telegram uses its own
# client_kind for historical reasons; both resolve to one conversation
# through the session manager's default-session lookup.
SHARED_CLIENT_KIND = "telegram"


class ClientRegistry:
    """Live client connections, for pushing unprompted messages.

    Memory only, like the worker registry: who is connected right now is
    a property of open sockets, and a client that reconnects announces
    itself again within seconds.
    """

    def __init__(self) -> None:
        self._clients: dict[str, "ClientConnection"] = {}

    def add(self, connection: "ClientConnection") -> None:
        self._clients[connection.id] = connection

    def remove(self, connection_id: str) -> None:
        self._clients.pop(connection_id, None)

    def connected(self) -> list["ClientConnection"]:
        return list(self._clients.values())

    async def broadcast_notification(
        self, text: str, priority: int = 5,
        approval_id: str | None = None, artifact_id: str | None = None,
    ) -> int:
        """Push to every open client. Returns how many received it."""
        sent = 0
        for client in list(self._clients.values()):
            try:
                await client.send(CORE_NOTIFICATION, CoreNotification(
                    text=text, priority=priority,
                    approval_id=approval_id, artifact_id=artifact_id,
                ))
                sent += 1
            except Exception:
                log.debug("client push failed - it will reconnect")
        return sent

    async def broadcast_status(self, payload: dict[str, Any]) -> None:
        for client in list(self._clients.values()):
            try:
                await client.send(CORE_STATUS, CoreStatus(payload=payload))
            except Exception:
                pass


class ClientConnection:
    """One client's connection, from hello to hangup."""

    def __init__(
        self,
        websocket: WebSocket,
        expected_token: str,
        session_mgr: SessionManager,
        sessions: SessionsRepo,
        registry: ClientRegistry,
        owner_timezone: str,
        approvals: ApprovalService | None = None,
        transcriber: "Transcriber | None" = None,
        speaker: "SpeechBackend | None" = None,
    ) -> None:
        self._ws = websocket
        self._expected_token = expected_token
        self._mgr = session_mgr
        self._sessions = sessions
        self._registry = registry
        self._owner_timezone = owner_timezone
        self._approvals = approvals
        self._transcriber = transcriber
        self._speaker = speaker
        # Whether this client wants to hear replies. Set when it speaks
        # rather than types: someone who typed is reading, and reading
        # while being read to is worse than either alone.
        self._speak_replies = False
        # Audio arrives in pieces while the owner is still speaking, so
        # it accumulates here until he stops. Keyed by sequence so
        # out-of-order frames reassemble correctly.
        self._audio: dict[int, bytes] = {}
        self._audio_mime = "audio/webm"
        self.id = secrets.token_urlsafe(8)
        self.client_kind = "web"
        self.label = ""

    async def run(self) -> None:
        await self._ws.accept()
        try:
            if not await self._handshake():
                return
            self._registry.add(self)
            await self._receive_loop()
        except WebSocketDisconnect:
            pass
        except Exception:
            log.error("client connection failed", exc_info=True,
                      extra={"client_id": self.id})
        finally:
            self._registry.remove(self.id)

    # -- handshake ------------------------------------------------------------

    async def _handshake(self) -> bool:
        try:
            raw = await asyncio.wait_for(self._ws.receive_json(), timeout=10.0)
            envelope = Envelope.model_validate(raw)
            if envelope.kind != CLIENT_HELLO:
                await self.send(CORE_ERROR, CoreError(
                    message="first frame must be client.hello"
                ))
                return False
            hello = envelope.parse_payload()
            assert isinstance(hello, ClientHello)
        except Exception:
            await self._ws.close(code=1008)
            return False

        # Timing-safe, same reasoning as everywhere else: how long a
        # comparison takes must not leak how much of a guess was right.
        if not self._expected_token or not secrets.compare_digest(
            hello.token, self._expected_token
        ):
            log.warning("client presented a bad token")
            await self.send(CORE_ERROR, CoreError(message="invalid token"))
            await self._ws.close(code=1008)
            return False

        self.client_kind = hello.client_kind
        self.label = hello.label

        session = await self._sessions.get_or_create_default(SHARED_CLIENT_KIND)
        await self.send(CORE_READY, CoreReady(
            session_id=session.id,
            owner_timezone=self._owner_timezone,
        ))
        log.info("client connected", extra={
            "client_id": self.id, "kind": self.client_kind,
            "label": self.label,
        })
        return True

    # -- receiving ------------------------------------------------------------

    async def _receive_loop(self) -> None:
        while True:
            raw = await self._ws.receive_json()
            try:
                envelope = Envelope.model_validate(raw)
                payload = envelope.parse_payload()
            except (UnknownKind, ValidationError) as exc:
                await self.send(CORE_ERROR, CoreError(
                    message=f"unparseable frame: {str(exc)[:200]}"
                ))
                continue

            if envelope.kind == CLIENT_USER_MESSAGE:
                assert isinstance(payload, ClientUserMessage)
                await self._handle_message(payload)

            elif envelope.kind == CLIENT_INTERRUPT:
                # Handled implicitly: the session manager cancels an
                # in-flight turn when the next message arrives. An
                # explicit stop-without-replacement is a later refinement.
                log.debug("interrupt received", extra={"client_id": self.id})

            elif envelope.kind == CLIENT_COMMAND:
                assert isinstance(payload, ClientCommand)
                await self._handle_command(payload)

            elif envelope.kind == CLIENT_AUDIO_CHUNK:
                assert isinstance(payload, ClientAudioChunk)
                self._audio[payload.seq] = base64.b64decode(payload.data_b64)
                self._audio_mime = payload.mime

            elif envelope.kind == CLIENT_AUDIO_END:
                await self._handle_audio_end()

    async def _handle_message(self, message: ClientUserMessage) -> None:
        """Relay a message, streaming text and - if the owner spoke -
        speech back.

        Speech is synthesised SENTENCE BY SENTENCE as the text streams.
        Waiting for the whole reply would mean several seconds of
        silence; this way JARVIS starts talking while he is still
        writing the rest of the answer.
        """
        buffer = SentenceBuffer()
        speaking = (
            self._speak_replies
            and self._speaker is not None
            and self._speaker.available
        )
        audio_seq = 0

        async def speak(sentence: str, seq: int) -> None:
            """Synthesise one sentence and send it, tagged with its
            position in the reply.

            THE NUMBER IS ASSIGNED BY THE CALLER, before this runs.
            Synthesis happens concurrently - deliberately, since doing
            it in sequence would stall the text stream behind the audio -
            so a short sentence finishes before a long one that came
            first. Numbering here, after the await, produced audio that
            arrived and played out of order: "Ready when you are" before
            the paragraph it followed.
            """
            assert self._speaker is not None
            audio = await self._speaker.synthesise(sentence)
            if not audio:
                # A lost phrase, not a lost reply - but the client is
                # waiting for this number, so tell it to skip.
                await self.send(CORE_AUDIO, CoreAudio(
                    seq=seq, data_b64="", mime="audio/wav",
                ))
                return
            await self.send(CORE_AUDIO, CoreAudio(
                seq=seq,
                data_b64=base64.b64encode(audio).decode("ascii"),
            ))

        async def on_text(chunk: str) -> None:
            # on_text is now the one that ADVANCES the counter, since
            # numbering moved out of speak() - so the declaration lives
            # here rather than there.
            nonlocal audio_seq

            await self.send(CORE_ASSISTANT_DELTA, CoreAssistantDelta(text=chunk))
            if not speaking:
                return
            for sentence in buffer.feed(chunk):
                # Number it NOW, while the order is known. Synthesis
                # runs concurrently - awaiting it here would stall the
                # text stream behind the audio - so the sequence has to
                # be fixed before the race starts.
                asyncio.create_task(speak(sentence, audio_seq))
                audio_seq += 1

        try:
            result = await self._mgr.handle_user_message(
                SHARED_CLIENT_KIND, message.text,
                on_text=on_text,
                attachments=message.attachments,
            )
        except Exception as exc:
            log.error("turn failed", exc_info=True)
            await self.send(CORE_ERROR, CoreError(
                message=f"{type(exc).__name__}: {exc}"
            ))
            return

        if result.interrupted:
            return
        if speaking:
            trailing = buffer.flush()
            if trailing:
                await speak(trailing, audio_seq)
                audio_seq += 1
            await self.send(CORE_AUDIO_DONE, CoreAudioDone())

        assert result.reply is not None
        await self.send(CORE_ASSISTANT_DONE, CoreAssistantDone(
            turn_id=result.reply.id,
            job_refs=result.reply.job_refs,
        ))

    async def _handle_audio_end(self) -> None:
        """The owner stopped speaking. Transcribe, show him what was
        heard, then treat it as an ordinary message.

        The transcript goes back BEFORE thinking starts: if he was
        misheard he finds out immediately, rather than after a confused
        answer arrives twenty seconds later.
        """
        audio = b"".join(self._audio[seq] for seq in sorted(self._audio))
        self._audio.clear()

        if self._transcriber is None or not self._transcriber.available:
            await self.send(CORE_ERROR, CoreError(
                message="Voice is not configured - no transcription key."
            ))
            return

        text = await self._transcriber.transcribe(audio, self._audio_mime)
        if not text:
            await self.send(CORE_ERROR, CoreError(
                message="I did not catch that, sir."
            ))
            return

        # He spoke, so he expects to be spoken to. Typing later turns
        # this back off - see _handle_message.
        self._speak_replies = True
        await self.send(CORE_TRANSCRIPT, CoreTranscript(text=text))
        await self._handle_message(ClientUserMessage(text=text))

    async def _handle_command(self, command: ClientCommand) -> None:
        if command.name == "new_session":
            session = await self._sessions.get_or_create_default(
                SHARED_CLIENT_KIND
            )
            await self._sessions.archive(session.id)
            await self.send(CORE_NOTIFICATION, CoreNotification(
                text="Thread archived. Clean slate, sir.", priority=5,
            ))

        elif command.name in ("approve", "reject") and self._approvals:
            approval_id = str(command.args.get("approval_id", ""))
            decided = await self._approvals.decide(
                approval_id, approve=command.name == "approve"
            )
            await self.send(CORE_NOTIFICATION, CoreNotification(
                text=(
                    f"{command.name.title()}d."
                    if decided else "Already settled, sir."
                ),
                priority=5,
            ))

    # -- sending --------------------------------------------------------------

    async def send(self, kind: str, payload: Any) -> None:
        envelope = make_envelope(kind, payload)
        await self._ws.send_json(envelope.model_dump(mode="json"))
