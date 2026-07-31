# =============================================================================
# src/jarvis/subagents/archivist.py - MNEMOSYNE, the archivist
# =============================================================================
#
# The memory curator. Unlike ATHENA, she is not an agentic loop: her two
# jobs are single structured calls, so there is nothing to iterate on.
#
#   extract_facts - read recent conversation, return durable facts
#   write_profile - distil the fact vault into the standing page about
#                   the owner
#
# STRUCTURED OUTPUT VIA TOOL, NOT JSON PARSING. To get a list of facts
# back reliably we hand the model a TOOL whose input schema is the shape
# we want and read the arguments it passes. The provider enforces the
# schema. Asking for JSON in prose and then repairing the result - MK2's
# habit - is deliberately dead here.
#
# Tiers: extraction runs on the cheap utility tier (it is reading and
# summarising, done nightly, possibly over many turns). Profile writing
# runs on the reasoner: that page is injected into EVERY conversation
# forever, so it is worth the better model once a night.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from jarvis.common.facts import Fact, FactCategory
from jarvis.common.log import get_logger
from jarvis.llm.layer import LLMLayer, ToolSpec, user_message
from jarvis.llm.tiers import Tier

log = get_logger("subagent.archivist")

ACTOR = "subagent.archivist"

_CATEGORIES = [c.value for c in FactCategory]

EXTRACTION_PROMPT_V1 = """\
You are MNEMOSYNE, the archivist of a personal AI system. You read
conversations between the system and its owner and record what is worth
remembering permanently.

Record a fact when the conversation reveals something DURABLE:
- what the owner is building, studying, or responsible for
- how he wants things done: tools, style, working preferences
- people, places, and commitments that recur in his life
- decisions he has made that will still matter in six months

Do NOT record:
- passing logistics ("running late today", "will look at it tonight")
- anything the system said unless the owner confirmed it
- transient state: moods, what he is doing right now, weather
- restatements of what you already know, unless the new version
  CORRECTS it
- anything he would be uncomfortable seeing written down about himself

Rules for the text of a fact:
- ONE self-contained sentence that will make sense read alone in a year
- name the owner rather than writing "he"
- concrete over vague: "uses SQLite with WAL for the Core's storage"
  beats "has opinions about databases"

Importance: 0.9 defining and permanent, 0.5 ordinary, 0.2 minor detail.

If nothing in the conversation is worth keeping, record no facts. That
is a normal and frequent outcome - an empty night is better than a
vault full of noise.
"""

PROFILE_PROMPT_V1 = """\
You are MNEMOSYNE, writing the standing profile of the owner of a
personal AI system. This page is placed in front of the system at the
start of EVERY conversation, so it must be short, current, and worth its
space.

Write markdown, under 400 words, in this shape:

## Who
Two or three sentences: name, what he does, where he is.

## Current work
The projects that are actually live, one line each, with their state.

## How he works
Preferences the system should honour without being told each time.

## Context
People, commitments, and standing facts that recur.

Rules:
- Include only what is STABLE. One-off details stay in the vault and
  arrive through search when relevant.
- Prefer the recent version where facts disagree.
- Write plainly, third person, no filler, no headings beyond the four.
- If a section has nothing worth saying, omit the section entirely.
"""

_RECORD_FACTS_TOOL = ToolSpec(
    name="record_facts",
    description="Record the durable facts found in the conversation.",
    input_schema={
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "One self-contained sentence.",
                        },
                        "category": {"type": "string", "enum": _CATEGORIES},
                        "importance": {
                            "type": "number", "minimum": 0.0, "maximum": 1.0,
                        },
                    },
                    "required": ["text", "category", "importance"],
                },
            }
        },
        "required": ["facts"],
    },
)


@dataclass(frozen=True)
class ExtractedFact:
    """One candidate fact, before it enters the vault."""

    text: str
    category: FactCategory
    importance: float


async def extract_facts(
    llm: LLMLayer,
    conversation: str,
    known_facts: list[str],
    *,
    trace_id: str,
) -> list[ExtractedFact]:
    """Read a stretch of conversation and return what is worth keeping.

    known_facts is a sample of what the vault already holds, so the model
    can skip restatements rather than producing near-duplicates that
    store-time dedup then has to merge.
    """
    known_block = (
        "Already known, do not repeat unless correcting:\n"
        + "\n".join(f"- {f}" for f in known_facts[:40])
        if known_facts else "The vault is empty; anything durable is new."
    )
    prompt = f"{known_block}\n\nConversation:\n{conversation}"

    response = await llm.complete(
        Tier.UTILITY,
        EXTRACTION_PROMPT_V1,
        [user_message(prompt)],
        actor=ACTOR,
        trace_id=trace_id,
        tools=[_RECORD_FACTS_TOOL],
        max_tokens=2048,
    )

    extracted: list[ExtractedFact] = []
    for call in response.tool_calls:
        if call.name != "record_facts":
            continue
        for raw in call.args.get("facts", []):
            try:
                extracted.append(ExtractedFact(
                    text=str(raw["text"]).strip(),
                    category=FactCategory(raw.get("category", "other")),
                    importance=float(raw.get("importance", 0.5)),
                ))
            except (KeyError, ValueError):
                # One malformed entry must not discard a good batch.
                log.warning("skipped malformed extracted fact",
                            extra={"raw": str(raw)[:200]})
    return extracted


async def write_profile(
    llm: LLMLayer,
    facts: list[Fact],
    current_profile: str,
    *,
    trace_id: str,
) -> str:
    """Distil the vault into the standing page about the owner."""
    fact_block = "\n".join(
        f"- [{f.category.value}, importance {f.importance:.1f}] {f.text}"
        for f in facts
    )
    previous = (
        f"The current profile, for continuity:\n{current_profile}\n\n"
        if current_profile.strip() else ""
    )
    prompt = (
        f"{previous}Everything known about the owner, most important "
        f"first:\n{fact_block}\n\nWrite the updated profile."
    )

    response = await llm.complete(
        Tier.REASONER,
        PROFILE_PROMPT_V1,
        [user_message(prompt)],
        actor=ACTOR,
        trace_id=trace_id,
        max_tokens=2048,
    )
    return response.text.strip()