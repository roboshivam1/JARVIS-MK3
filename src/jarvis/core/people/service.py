# =============================================================================
# src/jarvis/core/people/service.py - debriefs in, proposals out
# =============================================================================
#
# The owner talks about someone; this turns it into proposals waiting in
# the review queue. It never writes to the real tables - that is the
# repo's approve(), driven by the owner.
#
# HANDLES, NOT IDS. The model sees the roster as R1, R2... and known
# facts as F1, F2... rather than 26-character ULIDs. Copying a ULID back
# correctly is exactly the kind of chore a model occasionally botches,
# and a botched id means a fact silently attached to nobody. Handles are
# short and unambiguous; this file translates them back before staging,
# and drops anything that points at a handle that does not exist.
#
# WHAT GOES TO THE API: the roster (names, aliases, one-line summaries)
# and the non-sensitive facts of the people THIS debrief mentions.
# Sensitive facts never leave the machine during extraction.
#
# Utility tier: extraction is a chore, and every result is reviewed
# anyway. If links or sensitivity calls come back poor, REASONER is a
# one-word change below.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from jarvis.common.log import get_logger
from jarvis.common.people import (
    PROPOSAL_MODELS,
    RESULT_FIELDS,
    AliasProposal,
    CommitmentProposal,
    ExtractionResult,
    FactProposal,
    KeyDateProposal,
    NewPersonProposal,
)
from jarvis.core.db.repos.people import PeopleRepo
from jarvis.llm.layer import LLMLayer, ToolSpec, user_message
from jarvis.llm.tiers import Tier

log = get_logger("core.people")

ACTOR = "core.people"
_TOOL = "record_proposals"
_ROSTER_LIMIT = 200
_TIER = Tier.UTILITY

_SYSTEM = """You turn the owner's spoken debriefs about people into proposals \
for his personal people database. The owner reviews every proposal before \
anything is stored, so be complete but never invent.

Rules:
- Extract only what the debrief states or directly implies.
- One atomic fact per item, a short standalone sentence ("Studies ECE at LNMIIT").
- Someone clearly on the ROSTER: person_ref "existing:<handle>", e.g. existing:R3.
- Anyone else: a new_people entry with tag P1, P2, ... referenced as "new:P1". \
If unsure whether someone is a roster person, make them new and write \
"possibly R3" in their summary - the owner decides.
- Nicknames or other ways the owner refers to someone become aliases.
- A fact that replaces one under KNOWN FACTS (changed job, moved city) sets \
supersedes_fact_id to that fact's handle, e.g. F2.
- sensitivity "sensitive" for health, family trouble, conflicts, romantic \
relationships, money trouble, or anything said in confidence.
- Commitments: what the owner promised (i_owe) or they promised (they_owe). \
Relative dates become absolute YYYY-MM-DD using today's date.
- Dates: birthdays recur (MM-DD); one-off events are YYYY-MM-DD.
- The owner himself is not in this database. Skip facts about him.
- Nothing worth storing: return empty lists.

Today is {today}."""


class ExtractionFailed(RuntimeError):
    """The model call returned nothing usable. Nothing was staged."""


@dataclass(frozen=True)
class DebriefOutcome:
    debrief_id: str
    staged: int
    dropped: int      # invalid or dangling proposals thrown away

    def summary(self) -> str:
        if self.staged == 0:
            return "Nothing in that debrief was worth storing."
        note = f" ({self.dropped} unusable item(s) dropped)" if self.dropped else ""
        return f"{self.staged} proposal(s) waiting for review{note}."


class PeopleService:
    def __init__(self, repo: PeopleRepo, llm: LLMLayer, tz: ZoneInfo) -> None:
        self._repo = repo
        self._llm = llm
        self._tz = tz

    # -- debrief -> staged proposals -----------------------------------------

    async def debrief(self, raw_text: str, *, trace_id: str, source: str = "web") -> DebriefOutcome:
        people_map, fact_map, context = await self._context(raw_text)
        raw = await self._call_model(raw_text, context, trace_id)
        parsed, dropped = self._parse(raw)
        result, dangling = self._translate(parsed, people_map, fact_map)
        debrief_id = await self._repo.stage(raw_text, result, source)
        return DebriefOutcome(debrief_id, result.count, dropped + dangling)

    async def _context(self, raw_text: str) -> tuple[dict[str, str], dict[str, str], str]:
        """The prompt's reference material, plus handle -> id maps."""
        people_map: dict[str, str] = {}
        lines = ["ROSTER (handle | name | aliases | relationship | summary):"]
        for i, p in enumerate(await self._repo.roster(limit=_ROSTER_LIMIT), start=1):
            people_map[f"R{i}"] = p["id"]
            lines.append(
                f"R{i} | {p['name']} | {', '.join(p['aliases']) or '-'} | "
                f"{p['relationship'] or '-'} | {p['summary'] or '-'}")
        if not people_map:
            lines.append("(empty - everyone is new)")

        handle_of = {pid: h for h, pid in people_map.items()}
        fact_map: dict[str, str] = {}
        known: list[str] = []
        for pid in await self._repo.mentioned_in(raw_text):
            prof = await self._repo.profile(pid, include_sensitive=False)
            if prof is None or pid not in handle_of:
                continue
            for f in prof["facts"]:
                handle = f"F{len(fact_map) + 1}"
                fact_map[handle] = f["id"]
                known.append(f"{handle} | {handle_of[pid]} {prof['name']} | {f['text']}")
        if known:
            lines += ["", "KNOWN FACTS (handle | person | fact):", *known]
        return people_map, fact_map, "\n".join(lines)

    async def _call_model(self, raw_text: str, context: str, trace_id: str) -> dict[str, Any]:
        today = datetime.now(self._tz).strftime("%Y-%m-%d (%A)")
        response = await self._llm.complete(
            _TIER,
            _SYSTEM.format(today=today),
            [user_message(f"{context}\n\nDEBRIEF:\n{raw_text}")],
            actor=ACTOR,
            trace_id=trace_id,
            tools=[ToolSpec(
                name=_TOOL,
                description="Record the proposals extracted from the debrief.",
                input_schema=ExtractionResult.model_json_schema(),
            )],
            tool_choice={"type": "tool", "name": _TOOL},
            max_tokens=4096,
        )
        # A tool call cut off by the token limit loses its LAST fields
        # silently - a debrief missing its commitments would look complete.
        if response.stop_reason == "max_tokens":
            raise ExtractionFailed("the debrief was too long to extract in one go - split it")
        if not response.tool_calls:
            raise ExtractionFailed("the model returned no proposals")
        return response.tool_calls[0].args

    @staticmethod
    def _parse(raw: dict[str, Any]) -> tuple[ExtractionResult, int]:
        """Validate item by item: one malformed item must not sink the rest."""
        clean: dict[str, list[Any]] = {}
        dropped = 0
        for kind, field in RESULT_FIELDS.items():
            items = raw.get(field) or []
            clean[field] = []
            for item in items if isinstance(items, list) else []:
                try:
                    clean[field].append(PROPOSAL_MODELS[kind].model_validate(item))
                except ValidationError as exc:
                    dropped += 1
                    log.warning("dropped invalid proposal", extra={
                        "kind": kind.value, "reason": exc.errors()[0]["msg"]})
        return ExtractionResult(**clean), dropped

    @staticmethod
    def _translate(
        result: ExtractionResult, people_map: dict[str, str], fact_map: dict[str, str]
    ) -> tuple[ExtractionResult, int]:
        """Handles -> real ids. Anything pointing at a handle or tag that
        does not exist is dropped here rather than failing at review."""
        dangling = 0
        tags: set[str] = set()
        new_people: list[NewPersonProposal] = []
        for person in result.new_people:
            if person.tag in tags:
                dangling += 1
                continue
            tags.add(person.tag)
            new_people.append(person)

        def fix_ref(ref: str) -> str | None:
            kind, _, value = ref.partition(":")
            if kind == "existing":
                pid = people_map.get(value)
                return f"existing:{pid}" if pid else None
            return ref if value in tags else None

        def about(items: list[Any]) -> list[Any]:
            nonlocal dangling
            kept = []
            for item in items:
                ref = fix_ref(item.person_ref)
                if ref is None:
                    dangling += 1
                    continue
                update: dict[str, Any] = {"person_ref": ref}
                if isinstance(item, FactProposal) and item.supersedes_fact_id:
                    update["supersedes_fact_id"] = fact_map.get(item.supersedes_fact_id)
                kept.append(item.model_copy(update=update))
            return kept

        aliases: list[AliasProposal] = about(result.aliases)
        facts: list[FactProposal] = about(result.facts)
        commitments: list[CommitmentProposal] = about(result.commitments)
        key_dates: list[KeyDateProposal] = about(result.key_dates)
        return ExtractionResult(
            new_people=new_people, aliases=aliases, facts=facts,
            commitments=commitments, key_dates=key_dates,
        ), dangling

    # -- reading people back, for conversation -------------------------------

    async def describe(self, person_id: str, include_sensitive: bool = True) -> str:
        """A person as plain text - for the lookup tool and the per-message
        context block. Sensitive facts are marked when included."""
        p = await self._repo.profile(person_id, include_sensitive=include_sensitive)
        if p is None:
            return ""
        head = p["name"]
        if p["aliases"]:
            head += f" (aka {', '.join(p['aliases'])})"
        if p["relationship"]:
            head += f" - {p['relationship']}"
        lines = [head]
        if p["summary"]:
            lines.append(p["summary"])
        for f in p["facts"]:
            mark = " [sensitive]" if f["sensitivity"] == "sensitive" else ""
            lines.append(f"- {f['text']}{mark}")
        for c in p["commitments"]:
            who = "Owner owes them" if c["direction"] == "i_owe" else "They owe the owner"
            due = f" (due {c['due_date']})" if c["due_date"] else ""
            lines.append(f"- Open: {who}: {c['what']}{due}")
        for d in p["dates"]:
            lines.append(f"- {d['label'].capitalize()}: {d['date']}")
        return "\n".join(lines)

    async def mention_context(self, text: str, limit: int = 4) -> str:
        """Who this message mentions, for the system prompt. Sensitive
        facts are never included here - only when explicitly looked up."""
        ids = await self._repo.mentioned_in(text)
        blocks = [await self.describe(pid, include_sensitive=False) for pid in ids[:limit]]
        return "\n\n".join(b for b in blocks if b)

    # -- answering questions about people -------------------------------------

    async def lookup(self, name: str) -> str:
        """Everything known about whoever the owner means, sensitive facts
        included - he asked about this person by name."""
        ids = [p["id"] for p in await self._repo.find(name)]
        if not ids:
            ids = await self._repo.mentioned_in(name)
        if not ids:
            return f"No one called {name!r} is in the people database."
        blocks = [await self.describe(pid, include_sensitive=True) for pid in ids[:5]]
        intro = "" if len(ids) == 1 else f"{len(ids)} people match {name!r}:\n\n"
        return intro + "\n\n".join(blocks)

    async def upcoming(self, days: int = 30) -> str:
        """Open commitments in both directions, and dates in the next
        `days` days: "who owes me what", "whose birthday is coming up"."""
        today = datetime.now(self._tz).date()
        i_owe: list[str] = []
        owed: list[str] = []
        for c in await self._repo.open_commitments():
            due = f" (due {c['due_date']})" if c["due_date"] else ""
            line = f"- {c['name']}: {c['what']}{due}"
            (i_owe if c["direction"] == "i_owe" else owed).append(line)

        dated: list[tuple[date, str]] = []
        for d in await self._repo.all_dates():
            when = _next_occurrence(d["date"], bool(d["recurring"]), today)
            if when is not None and 0 <= (when - today).days <= days:
                dated.append((when, f"- {when.strftime('%d %b')}: {d['name']}, {d['label']}"))

        lines: list[str] = []
        if i_owe:
            lines += ["The owner owes:", *i_owe]
        if owed:
            lines += ["Owed to the owner:", *owed]
        if dated:
            lines += [f"Coming up in the next {days} days:", *(t for _, t in sorted(dated))]
        return "\n".join(lines) or (
            f"No open commitments, and nothing dated in the next {days} days.")


def _next_occurrence(value: str, recurring: bool, today: date) -> date | None:
    """The next date on or after today that a stored date falls on. A
    29 February birthday lands on the 28th in years without one."""
    try:
        if not recurring:
            return date.fromisoformat(value)
        month, day = (int(part) for part in value.split("-"))
    except ValueError:
        return None
    for year in (today.year, today.year + 1):
        try:
            when = date(year, month, day)
        except ValueError:
            when = date(year, 2, 28)
        if when >= today:
            return when
    return None
