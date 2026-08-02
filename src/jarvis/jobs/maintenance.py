# =============================================================================
# src/jarvis/jobs/maintenance.py - the memory sleep cycle
# =============================================================================
#
# The nightly job that turns conversations into lasting memory. Four
# stages, each checkpointed:
#
#   extract      - read turns since the last cycle, pull out durable
#                  facts, store them (store-time dedup supersedes
#                  restatements automatically)
#   decay        - facts nobody has touched in a long time lose
#                  confidence; below a floor they expire. Never deleted.
#   profile      - rewrite the standing page about the owner
#   housekeeping - embed anything still missing a vector, or carrying one
#                  from a superseded model
#
# RESUMABLE, not idempotent: this is the first job type that genuinely
# needs checkpoints. A crash during profile writing must not re-run
# extraction and pay for those model calls twice, so each finished stage
# is recorded and skipped on the next attempt.
#
# The checkpoint also carries last_turn_id - the high-water mark of what
# has been read. That is what makes each night's extraction cover only
# what is new.
#
# DELIBERATELY NOT HERE: reconciling genuine CONTRADICTIONS across the
# vault ("uses Postgres" vs "switched to SQLite"). Store-time dedup
# merges restatements, but real contradictions need judgment rather than
# similarity, and that deserves its own pass with its own prompt.
#
# No session_id on this job, on purpose: nothing about routine nightly
# maintenance should buzz the owner's phone at 3:30 in the morning. The
# counts land in the event log, and JARVIS can narrate them if asked.
# =============================================================================

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.common.sessions import Turn
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.facts import FactsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.memory.profile import ProfileStore
from jarvis.core.memory.service import MemoryService
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec
from jarvis.llm.layer import LLMLayer
from jarvis.subagents.archivist import extract_facts, write_profile

log = get_logger("jobs.maintenance")

# How many turns one cycle reads. A night of heavy use is far less than
# this; the cap exists so a first run on an old database cannot send an
# enormous prompt.
_MAX_TURNS_PER_CYCLE = 120

# Per-turn character cap when building the conversation text. Long turns
# (a pasted document) would otherwise dominate the extraction prompt.
_MAX_TURN_CHARS = 1500

# Decay: a fact untouched this long starts losing confidence.
_DECAY_AFTER_DAYS = 45
_DECAY_STEP = 0.15
_CONFIDENCE_FLOOR = 0.25
# Facts this important never decay - they are the owner's defining
# details, and being unmentioned for a month does not make them untrue.
_DECAY_IMPORTANCE_SHIELD = 0.75


class SleepCycleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Empty payload today; a future manual trigger may want to request
    # specific stages.
    reason: str = Field(default="scheduled")


class SleepCycleOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turns_read: int
    facts_extracted: int
    facts_decayed: int
    facts_expired: int
    profile_rewritten: bool
    embeddings_backfilled: int


def register_maintenance_jobs(
    registry: JobTypeRegistry,
    llm: LLMLayer,
    memory: MemoryService,
    facts_repo: FactsRepo,
    sessions: SessionsRepo,
    profile: ProfileStore,
    events: EventsRepo,
) -> None:
    """Register maintenance job types. Called once at boot."""
    # Fail at BOOT if a dependency has not been constructed yet, rather
    # than at 3:30 a.m. when the handler dereferences None and burns
    # three retries discovering it.
    if memory is None or profile is None or sessions is None:
        raise RuntimeError(
            "register_maintenance_jobs called before its dependencies "
            "exist - check construction order in CoreApp.boot()"
        )

    async def handle(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, SleepCycleIn)
        state: dict[str, Any] = dict(ctx.checkpoint or {})
        done: list[str] = list(state.get("stages_done", []))

        counts = {
            "turns_read": int(state.get("turns_read", 0)),
            "facts_extracted": int(state.get("facts_extracted", 0)),
            "facts_decayed": int(state.get("facts_decayed", 0)),
            "facts_expired": int(state.get("facts_expired", 0)),
            "embeddings_backfilled": int(state.get("embeddings_backfilled", 0)),
        }
        profile_rewritten = bool(state.get("profile_rewritten", False))

        async def save(stage: str) -> None:
            done.append(stage)
            await ctx.save_checkpoint({
                **counts,
                "stages_done": done,
                "last_turn_id": state.get("last_turn_id"),
                "profile_rewritten": profile_rewritten,
            })

        # -- stage 1: extract -------------------------------------------------
        if "extract" not in done:
            await ctx.progress("reading recent conversation")
            after = state.get("last_turn_id")
            turns = await sessions.turns_since(after, limit=_MAX_TURNS_PER_CYCLE)
            counts["turns_read"] = len(turns)

            if turns:
                known = [f.text for f in await facts_repo.all_active(limit=40)]
                conversation = _render_turns(turns)
                candidates = await extract_facts(
                    llm, conversation, known, trace_id=ctx.trace_id
                )
                for candidate in candidates:
                    # store() handles near-duplicate supersession.
                    await memory.store(
                        text=candidate.text,
                        category=candidate.category,
                        importance=candidate.importance,
                        source_event_ids=[t.id for t in turns[-5:]],
                    )
                counts["facts_extracted"] = len(candidates)
                state["last_turn_id"] = turns[-1].id
            await save("extract")

        # -- stage 2: decay ---------------------------------------------------
        if "decay" not in done:
            await ctx.progress("decaying unused facts")
            cutoff = utc_now() - timedelta(days=_DECAY_AFTER_DAYS)
            decayed = expired = 0
            for fact in await facts_repo.all_active(limit=1000):
                if fact.importance >= _DECAY_IMPORTANCE_SHIELD:
                    continue
                if fact.last_accessed_ts > cutoff:
                    continue
                new_confidence = round(fact.confidence - _DECAY_STEP, 3)
                if new_confidence < _CONFIDENCE_FLOOR:
                    await facts_repo.expire(fact.id)
                    expired += 1
                else:
                    await facts_repo.set_confidence(fact.id, new_confidence)
                    decayed += 1
            counts["facts_decayed"] = decayed
            counts["facts_expired"] = expired
            await save("decay")

        # -- stage 3: profile -------------------------------------------------
        if "profile" not in done:
            top_facts = await facts_repo.all_active(limit=60)
            if top_facts:
                await ctx.progress("rewriting the profile")
                content = await write_profile(
                    llm, top_facts, await profile.current(), trace_id=ctx.trace_id
                )
                if content:
                    version = await profile.write(
                        content=content,
                        generated_by="archivist",
                        fact_count=len(top_facts),
                    )
                    profile_rewritten = True
                    await events.append(Event(
                        kind=EventKind.MEMORY_PROFILE_UPDATED,
                        source="core.memory",
                        job_id=ctx.job_id,
                        trace_id=ctx.trace_id,
                        payload={
                            "profile_version": version.id,
                            "fact_count": len(top_facts),
                        },
                    ))
            await save("profile")

        # -- stage 4: housekeeping --------------------------------------------
        if "housekeeping" not in done:
            await ctx.progress("backfilling embeddings")
            counts["embeddings_backfilled"] = await memory.backfill_embeddings(
                limit=500
            )
            await save("housekeeping")

        await events.append(Event(
            kind=EventKind.MEMORY_SLEEP_CYCLE_COMPLETED,
            source="core.memory",
            job_id=ctx.job_id,
            trace_id=ctx.trace_id,
            payload={**counts, "profile_rewritten": profile_rewritten},
        ))
        log.info("sleep cycle complete", extra=counts)

        return SleepCycleOut(**counts, profile_rewritten=profile_rewritten)

    registry.register(JobTypeSpec(
        type="memory.sleep_cycle",
        input_model=SleepCycleIn,
        output_model=SleepCycleOut,
        execution="resumable",
        requires=[],
        default_priority=8,      # idle-time work; never ahead of the owner
        timeout_s=900,
        handler=handle,
    ))


def _render_turns(turns: list[Turn]) -> str:
    """Turns as readable dialogue for the extraction prompt."""
    lines: list[str] = []
    for turn in turns:
        speaker = "OWNER" if turn.role.value == "user" else "JARVIS"
        content = turn.content.strip()
        if len(content) > _MAX_TURN_CHARS:
            content = content[:_MAX_TURN_CHARS] + " [...]"
        lines.append(f"{speaker}: {content}")
    return "\n\n".join(lines)