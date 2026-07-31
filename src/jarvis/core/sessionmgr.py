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
from dataclasses import dataclass
from typing import Any

from jarvis.agentloop.loop import LoopResult
from jarvis.common.events import Event, EventKind
from jarvis.common.ids import new_ulid
from jarvis.common.log import get_logger
from jarvis.common.sessions import Session, Turn, TurnRole
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.memory.profile import ProfileStore
from jarvis.core.memory.service import MemoryService
from jarvis.core.orchestrator.agent import Orchestrator
from jarvis.llm.layer import TextCallback

log = get_logger("core.sessionmgr")

SOURCE = "core.sessionmgr"

# How much verbatim history the model sees per turn. Enough for a natural
# conversation; the rolling summary will cover older ground when the
# memory phase arrives.
CONTEXT_TURNS = 20


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
    ) -> None:
        self._sessions = sessions
        self._events = events
        self._orchestrator = orchestrator
        # Optional so tests can exercise conversation flow without a
        # memory system; absent simply means no memory in context.
        self._memory = memory
        self._profile = profile
        # session id -> the task currently generating its reply
        self._inflight: dict[str, asyncio.Task[LoopResult]] = {}

    async def handle_user_message(
        self,
        client_kind: str,
        text: str,
        on_text: TextCallback | None = None,
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
        user_turn = Turn(session_id=session.id, role=TurnRole.USER, content=text)
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

        profile_doc = await self._profile.current() if self._profile else ""
        retrieved_memory = ""
        if self._memory is not None:
            hits = await self._memory.search(text, k=6)
            retrieved_memory = "\n".join(f"- {h.fact.text}" for h in hits)
            if hits:
                log.debug("memory retrieved for turn", extra={
                    "session_id": session.id, "hits": len(hits),
                })

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
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + turn.content
        else:
            messages.append({"role": role, "content": turn.content})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages