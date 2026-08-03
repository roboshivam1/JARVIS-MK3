# =============================================================================
# src/jarvis/common/client_protocol.py - the Core <-> client wire vocabulary
# =============================================================================
#
# Every message between the Core and an interactive client, in both
# directions, wrapped in the same Envelope the worker link uses.
#
# WHY A SOCKET AND NOT HTTP REQUESTS: a client needs the server to PUSH.
# Reply text streams in as it is generated, job progress arrives while
# work runs, and notifications appear with nobody having asked. Over
# HTTP that means polling, which is either slow or wasteful. The socket
# stays open and either side speaks when it has something to say.
#
# FILE UPLOADS ARE NOT HERE. They go over ordinary HTTP multipart: a
# 10MB file base64-encoded through JSON frames is 33% larger and needs
# reassembly logic we would have to write and get right. The socket
# carries conversation; HTTP carries bytes. The client gets an artifact
# id back and mentions it in a message.
#
# Telegram does not use this protocol - it speaks Telegram's, bridged
# inside the Core. That asymmetry is worth naming: Telegram was the
# first client and got a special case, and this is what the general
# version looks like.
# =============================================================================

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.envelope import register_kind

# -- client -> core -----------------------------------------------------------

class ClientHello(BaseModel):
    """First frame after connecting. Authenticates and identifies the
    surface, which decides which session the client joins."""

    model_config = ConfigDict(extra="forbid")

    token: str
    client_kind: str = "web"        # web | voice.mac | voice.rpi
    label: str = ""                 # human name for the device, for /status


class ClientUserMessage(BaseModel):
    """The owner said something."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    # Artifact ids uploaded before this message, attached to the turn.
    attachments: list[str] = Field(default_factory=list)


class ClientAudioChunk(BaseModel):
    """A slice of recorded speech, base64 PCM or webm/opus.

    Audio is one case where base64 over the socket IS right: it arrives
    in small pieces while the owner is still talking, so there is no
    complete file to POST until he stops - by which point the latency
    budget is already spent.
    """

    model_config = ConfigDict(extra="forbid")

    seq: int
    data_b64: str
    mime: str = "audio/webm"


class ClientAudioEnd(BaseModel):
    """The owner stopped talking. Transcribe what has arrived."""

    model_config = ConfigDict(extra="forbid")


class ClientInterrupt(BaseModel):
    """Stop talking - barge-in, or a mind changed mid-reply."""

    model_config = ConfigDict(extra="forbid")


class ClientCommand(BaseModel):
    """Structured commands: new_session, cancel_job, approve, reject."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


# -- core -> client -----------------------------------------------------------

class CoreReady(BaseModel):
    """Authenticated. Carries enough for the client to render itself
    before anything happens."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    owner_timezone: str
    protocol_v: int = 1


class CoreAssistantDelta(BaseModel):
    """A chunk of reply text, as it is generated."""

    model_config = ConfigDict(extra="forbid")

    text: str


class CoreAssistantDone(BaseModel):
    """The reply is complete."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    job_refs: list[str] = Field(default_factory=list)


class CoreTranscript(BaseModel):
    """What the Core heard. Sent before thinking starts, so the owner
    can see whether he was understood before waiting for an answer."""

    model_config = ConfigDict(extra="forbid")

    text: str


class CoreStatus(BaseModel):
    """A snapshot for the sidebar: workers, jobs, spend, approvals.
    Pushed on change rather than polled."""

    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any] = Field(default_factory=dict)


class CoreNotification(BaseModel):
    """An unprompted message that passed the notification policy."""

    model_config = ConfigDict(extra="forbid")

    text: str
    priority: int = 5
    approval_id: str | None = None
    artifact_id: str | None = None


class CoreThinking(BaseModel):
    """What JARVIS is doing right now - retrieving memory, calling a
    subagent. Makes a two-second wait feel purposeful rather than dead,
    and the events already exist; this only surfaces them."""

    model_config = ConfigDict(extra="forbid")

    note: str


class CoreError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str


# -- kind registration --------------------------------------------------------

CLIENT_HELLO = "client.hello"
CLIENT_USER_MESSAGE = "client.user_message"
CLIENT_AUDIO_CHUNK = "client.audio_chunk"
CLIENT_AUDIO_END = "client.audio_end"
CLIENT_INTERRUPT = "client.interrupt"
CLIENT_COMMAND = "client.command"

CORE_READY = "core.ready"
CORE_ASSISTANT_DELTA = "core.assistant_delta"
CORE_ASSISTANT_DONE = "core.assistant_done"
CORE_TRANSCRIPT = "core.transcript"
CORE_STATUS = "core.status"
CORE_NOTIFICATION = "core.notification"
CORE_THINKING = "core.thinking"
CORE_ERROR = "core.error"

register_kind(CLIENT_HELLO, ClientHello)
register_kind(CLIENT_USER_MESSAGE, ClientUserMessage)
register_kind(CLIENT_AUDIO_CHUNK, ClientAudioChunk)
register_kind(CLIENT_AUDIO_END, ClientAudioEnd)
register_kind(CLIENT_INTERRUPT, ClientInterrupt)
register_kind(CLIENT_COMMAND, ClientCommand)

register_kind(CORE_READY, CoreReady)
register_kind(CORE_ASSISTANT_DELTA, CoreAssistantDelta)
register_kind(CORE_ASSISTANT_DONE, CoreAssistantDone)
register_kind(CORE_TRANSCRIPT, CoreTranscript)
register_kind(CORE_STATUS, CoreStatus)
register_kind(CORE_NOTIFICATION, CoreNotification)
register_kind(CORE_THINKING, CoreThinking)
register_kind(CORE_ERROR, CoreError)
