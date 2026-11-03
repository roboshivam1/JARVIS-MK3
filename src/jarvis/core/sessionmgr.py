# =============================================================================
# src/jarvis/core/sessionmgr.py - the front desk of every conversation
# =============================================================================
#
# Every user message, from any client, passes through handle_user_message.
# Its duties, in order:
#
#   1. INTERRUPT: if a reply is still being generated for this session,
#      cancel it cleanly (asyncio task cancellation - the model call stops
#      mid-stream) and record session.interrupted. The newest message wins.
#   2. PERSIST FIRST: write the user's turn to the database BEFORE any
#      thinking starts. A crash mid-reply must never lose the question.
#   3. ASSEMBLE CONTEXT: the model remembers nothing between calls; memory
#      is the trick of replaying the last N stored turns as conversation
#      history. The database remembers so the model does not have to.
#   4. THINK: run the orchestrator inside a tracked, cancellable task.
#   5. PERSIST THE REPLY with the ids of the model calls that produced it,
#      so any answer can be traced to its exact calls and cost.
#
# The merge rule: the provider API demands strict user/assistant
# alternation, but real history can hold two user turns in a row (a reply
# was cancelled between them). Consecutive same-role turns are merged into
# one message before the model sees them.
# =============================================================================

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any

from jarvis.agentloop.loop import LoopResult
from jarvis.common.events import Event, EventKind
from jarvis.common.ids import new_ulid
from jarvis.common.log import get_logger
from jarvis.common.sessions import Session, Turn, TurnRole
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.db.repos.artifacts import ArtifactsRepo
from jarvis.core.memory.profile import ProfileStore
from jarvis.core.memory.service import MemoryService
from jarvis.core.orchestrator.agent import Orchestrator
from jarvis.core.people.service import PeopleService
from jarvis.core.projects.service import ProjectService
from jarvis.core.life.service import LifeService
from jarvis.core.reels.service import ReelService
from jarvis.llm.layer import TextCallback

log = get_logger("core.sessionmgr")

SOURCE = "core.sessionmgr"

# How much verbatim history the model sees per turn. Enough for a natural
# conversation; the rolling summary will cover older ground when the
# memory phase arrives.
CONTEXT_TURNS = 20

# Attached text files are inlined up to this size. Beyond it the file
# is described and its id given instead - a megabyte of CSV in the
# prompt costs more than handing it to DAEDALUS, who can actually
# compute over it.
_MAX_INLINE_BYTES = 100_000

# Images go to the model as pictures, in the turn they arrive and only
# that turn. History replays the text, never the pixels: a photo resent
# with every later message would cost the same again on each one.
# The provider takes 10MB per image once base64-encoded, which is about
# 7.5MB of file; past that the image is described like any other file.
_MAX_IMAGE_BYTES = 7_000_000
_MAX_IMAGES_PER_TURN = 5


@dataclass(frozen=True)
class TurnResult:
    """What handling one user message produced. reply is None when this
    turn was interrupted by a newer message before finishing."""

    session_id: str
    trace_id: str
    reply: Turn | None
    interrupted: bool


class SessionManager:
    """Owns conversation flow: storage order, context, cancellation."""

    def __init__(
        self,
        sessions: SessionsRepo,
        events: EventsRepo,
        orchestrator: Orchestrator,
        memory: MemoryService | None = None,
        profile: ProfileStore | None = None,
        artifacts: "ArtifactsRepo | None" = None,
        people: "PeopleService | None" = None,
        reels: "ReelService | None" = None,
        projects: "ProjectService | None" = None,
        life: "LifeService | None" = None,
    ) -> None:
        self._sessions = sessions
        self._events = events
        self._orchestrator = orchestrator
        # Optional so tests can exercise conversation flow without a
        # memory system; absent simply means no memory in context.
        self._memory = memory
        self._profile = profile
        self._artifacts = artifacts
        self._people = people
        self._reels = reels
        self._projects = projects
        self._life = life
        # session id -> the task currently generating its reply
        self._inflight: dict[str, asyncio.Task[LoopResult]] = {}

    async def handle_user_message(
        self,
        client_kind: str,
        text: str,
        on_text: TextCallback | None = None,
        attachments: list[str] | None = None,
    ) -> TurnResult:
        """The one entry point for user messages, all clients."""
        session = await self._sessions.get_or_create_default(client_kind)
        trace_id = new_ulid()

        # A session with no turns yet is one we just created (or one that
        # never got used) - record its opening once, when first spoken to.
        if not await self._sessions.recent_turns(session.id, limit=1):
            await self._events.append(Event(
                kind=EventKind.SESSION_OPENED,
                source=SOURCE,
                session_id=session.id,
                trace_id=trace_id,
                payload={"client_kind": client_kind},
            ))

        # 1. Interrupt any reply still being generated for this session.
        await self._cancel_inflight(session.id, trace_id)

        # 2. Persist the question before thinking about it.
        user_turn = Turn(
            session_id=session.id, role=TurnRole.USER, content=text,
            attachments=attachments or [],
        )
        await self._sessions.append_turn(user_turn)
        await self._events.append(Event(
            kind=EventKind.SESSION_TURN_USER,
            source=SOURCE,
            session_id=session.id,
            trace_id=trace_id,
            payload={"turn_id": user_turn.id},
        ))

        # 3. Assemble context: stored history, the standing profile, and
        #    facts retrieved with THIS message as the query.
        turns = await self._sessions.recent_turns(session.id, limit=CONTEXT_TURNS)
        messages = _turns_to_messages(turns)

        # Attached files become part of the message the model sees.
        # Without this the artifact exists, is catalogued, and is
        # invisible - JARVIS truthfully reports that nothing came
        # through, which is a confusing thing to hear about a file you
        # just watched upload.
        if attachments and self._artifacts is not None:
            described, images = await self._describe_attachments(attachments)
            text_part = messages[-1]["content"]
            if described:
                text_part += "\n\n" + described
            # Pictures first, then the words about them - the order the
            # provider recommends, and the one a person would read in.
            messages[-1]["content"] = (
                [*images, {"type": "text", "text": text_part}] if images else text_part
            )

        profile_doc = await self._profile.current() if self._profile else ""
        retrieved_memory = ""
        if self._memory is not None:
            hits = await self._memory.search(text, k=6)
            retrieved_memory = "\n".join(f"- {h.fact.text}" for h in hits)
            if hits:
                log.debug("memory retrieved for turn", extra={
                    "session_id": session.id, "hits": len(hits),
                })

        # People this message mentions: who they are, minus anything
        # sensitive - that only comes out when the owner asks by name.
        if self._people is not None:
            people_block = await self._people.mention_context(text)
            if people_block:
                retrieved_memory = "\n\n".join(
                    part for part in (retrieved_memory,
                                      "People mentioned:\n" + people_block) if part)

        # Projects this message names: what they are and where they stand.
        if self._projects is not None:
            try:
                project_block = await self._projects.mention_context(text)
            except Exception:
                log.exception("project context failed - carrying on without it")
                project_block = ""
            if project_block:
                retrieved_memory = "\n\n".join(
                    part for part in (retrieved_memory,
                                      "Projects mentioned:\n" + project_block) if part)

        # His own reels, every turn: the reel service talks to the owner
        # outside this conversation, and "what should the caption be?"
        # needs to know which film just went out.
        if self._reels is not None:
            try:
                reels_block = await self._reels.context_block()
            except Exception:
                log.warning("reel context unavailable", exc_info=True)
                reels_block = ""
            if reels_block:
                retrieved_memory = "\n\n".join(
                    part for part in (retrieved_memory,
                                      "Your reels, newest first:\n" + reels_block) if part)

        # His day so far: the checklist and what is logged. Small, and it
        # is what lets "done with the module" tick the right item.
        if self._life is not None:
            try:
                life_block = await self._life.status_text()
            except Exception:
                log.warning("life context unavailable", exc_info=True)
                life_block = ""
            if life_block:
                retrieved_memory = "\n\n".join(
                    part for part in (retrieved_memory,
                                      "His day (life log):\n" + life_block) if part)

        # 4. Think, inside a tracked and cancellable task.
        gen_task = asyncio.create_task(
            self._orchestrator.respond(
                messages,
                session_id=session.id,
                rolling_summary=session.rolling_summary,
                profile_doc=profile_doc,
                retrieved_memory=retrieved_memory,
                trace_id=trace_id,
                on_text=on_text,
            ),
            name=f"turn-{session.id[-6:]}",
        )
        self._inflight[session.id] = gen_task
        try:
            result = await gen_task
        except asyncio.CancelledError:
            # A newer message cancelled us; the interrupt path has already
            # recorded the event. Report the interruption and step aside.
            return TurnResult(
                session_id=session.id, trace_id=trace_id,
                reply=None, interrupted=True,
            )
        finally:
            if self._inflight.get(session.id) is gen_task:
                del self._inflight[session.id]

        # 5. Persist the reply with its trace linkage.
        reply = Turn(
            session_id=session.id,
            role=TurnRole.ASSISTANT,
            content=result.text,
            llm_call_ids=result.llm_call_ids,
        )
        await self._sessions.append_turn(reply)
        await self._events.append(Event(
            kind=EventKind.SESSION_TURN_ASSISTANT,
            source=SOURCE,
            session_id=session.id,
            trace_id=trace_id,
            payload={
                "turn_id": reply.id,
                "iterations": result.iterations,
                "tool_calls": result.tool_calls_made,
                "hit_iteration_budget": result.hit_iteration_budget,
            },
        ))
        return TurnResult(
            session_id=session.id, trace_id=trace_id,
            reply=reply, interrupted=False,
        )

    async def _cancel_inflight(self, session_id: str, trace_id: str) -> None:
        """Cancel the reply being generated for this session, if any, and
        wait until it has actually stopped before proceeding."""
        task = self._inflight.get(session_id)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task   # ensure the model call is torn down before we go on
        except asyncio.CancelledError:
            pass
        except Exception:
            # It died of its own error in the same moment we cancelled it;
            # either way it is finished, which is all we need here.
            log.warning("inflight turn ended with error during cancel",
                        exc_info=True, extra={"session_id": session_id})
        await self._events.append(Event(
            kind=EventKind.SESSION_INTERRUPTED,
            source=SOURCE,
            session_id=session_id,
            trace_id=trace_id,
            payload={},
        ))
        log.info("interrupted in-flight turn", extra={"session_id": session_id})


    async def _describe_attachments(
        self, artifact_ids: list[str],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Render attached files into the message.

        Returns (text, image blocks). Images the model can see go in as
        IMAGE BLOCKS - he looks at them. Text-shaped files are INLINED,
        because the point of attaching a CSV is usually that JARVIS
        should read it. Anything else gets a line describing it plus its
        id, which is enough for him to hand it to a subagent - DAEDALUS
        can load a spreadsheet the model cannot.
        """
        assert self._artifacts is not None
        parts: list[str] = []
        images: list[dict[str, Any]] = []
        for artifact_id in artifact_ids:
            artifact = await self._artifacts.get(artifact_id)
            if artifact is None:
                continue

            if image_media_type(artifact.mime, artifact.name) is not None:
                try:
                    data = self._artifacts.read_bytes(artifact)
                except Exception:
                    log.warning("attached image unreadable", exc_info=True)
                    data = b""
                # Trust the bytes, not the label: a PNG named .jpg is
                # refused outright by the provider if declared wrongly.
                media_type = sniff_image(data)
                if media_type is None:
                    parts.append(
                        f"[Attached: {artifact.name}, artifact id {artifact.id}. "
                        f"An image in a format you cannot view (only JPEG, PNG, "
                        f"GIF and WebP) - ask him to send it as a JPEG or screenshot.]"
                    )
                elif len(data) > _MAX_IMAGE_BYTES:
                    parts.append(
                        f"[Attached: {artifact.name}, artifact id {artifact.id}. "
                        f"An image too large to view ({len(data) // 1_000_000} MB) - "
                        f"ask him to send a smaller copy.]"
                    )
                elif len(images) >= _MAX_IMAGES_PER_TURN:
                    parts.append(
                        f"[Attached: {artifact.name}, artifact id {artifact.id}. "
                        f"Not shown: only the first {_MAX_IMAGES_PER_TURN} images "
                        f"of a message are.]"
                    )
                else:
                    images.append({"type": "image", "source": {
                        "type": "base64", "media_type": media_type,
                        "data": base64.b64encode(data).decode("ascii"),
                    }})
                    parts.append(
                        f"[Attached image {len(images)}: {artifact.name}, "
                        f"artifact id {artifact.id} - shown above.]"
                    )
                continue

            inlineable = (
                artifact.mime.startswith("text/")
                or artifact.mime in ("application/json", "application/xml")
                or artifact.name.endswith((".md", ".txt", ".csv", ".py", ".json"))
            )

            if inlineable and artifact.size <= _MAX_INLINE_BYTES:
                try:
                    content = self._artifacts.read_bytes(artifact).decode(
                        "utf-8", errors="replace"
                    )
                except Exception:
                    content = "(could not be read)"
                parts.append(
                    f"[Attached: {artifact.name}, artifact id {artifact.id}]\n"
                    f"{content}"
                )
            else:
                parts.append(
                    f"[Attached: {artifact.name} ({artifact.mime}, "
                    f"{artifact.size} bytes), artifact id {artifact.id}. "
                    f"Not readable as text - pass this id to a subagent "
                    f"that can handle it.]"
                )
        return "\n\n".join(parts), images


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif")


def image_media_type(mime: str, name: str) -> str | None:
    """Is this attachment meant as a picture? By label only - the bytes
    are checked separately by sniff_image before anything is sent."""
    if mime.startswith("image/") or name.lower().endswith(_IMAGE_SUFFIXES):
        return mime if mime.startswith("image/") else "image/*"
    return None


def sniff_image(data: bytes) -> str | None:
    """The media type the provider will accept for these bytes, from
    their first few bytes; None for anything else (HEIC included)."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _turns_to_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    """Stored turns -> provider messages, honouring two provider rules:
    strict role alternation (merge consecutive same-role turns) and a
    user message first (drop assistant turns stranded at the window's
    leading edge)."""
    messages: list[dict[str, Any]] = []
    for turn in turns:
        if not turn.content.strip():
            continue
        role = turn.role.value
        content = turn.content
        # Attachments are only ever SHOWN in their own turn; in history,
        # a marker says they were there, so a later "what was in that
        # photo?" is answered from his earlier reply instead of a guess.
        if turn.attachments and turn is not turns[-1]:
            content += f"\n\n[sent with {len(turn.attachments)} attachment(s)]"
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages