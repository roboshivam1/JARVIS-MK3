"""people/extract.py — debrief text → staged proposals, via the Anthropic API.

The model is forced to answer through a single tool whose input schema is
ExtractionResult, so the output is structured JSON rather than prose.
Invalid items are dropped individually instead of failing the whole debrief.
"""
from __future__ import annotations

from datetime import date

from pydantic import ValidationError

from .models import ExtractionResult, PAYLOAD_MODELS, RESULT_FIELDS
from .review import ReviewQueue
from .store import PeopleStore

SYSTEM = """You turn {owner}'s spoken debriefs about people into proposals for his \
personal people database. Every proposal is reviewed by {owner} before it is stored.

Rules:
- Extract only what the debrief states or directly implies. Never invent details.
- One atomic fact per item, a short standalone sentence ("Studies ECE at LNMIIT").
- If the debrief clearly means someone on the ROSTER, use person_ref "existing:<id>".
- Anyone not on the roster gets a new_people entry (tag P1, P2, ...) and is \
referenced as "new:P1". If unsure whether someone is an existing person, create a \
new person and say "possibly existing:<id>" in their summary — {owner} will decide.
- Nicknames or other ways {owner} refers to someone → aliases.
- If a fact replaces one listed under KNOWN FACTS (changed job, moved city), set \
supersedes_fact_id to that fact's id.
- sensitivity "sensitive" for health, family trouble, conflicts, romantic \
relationships, money problems, or anything said in confidence.
- Commitments: what {owner} promised (i_owe) or they promised (they_owe). Turn \
relative dates into absolute YYYY-MM-DD using today's date.
- key_dates: birthdays are recurring (MM-DD); one-off events are YYYY-MM-DD.
- {owner} himself is not in this database — skip facts about {owner}.
- If there is nothing worth storing, return empty lists.

Today is {today}."""

TOOL_NAME = "record_proposals"


class DebriefExtractor:
    def __init__(self, store: PeopleStore, queue: ReviewQueue, client, model: str,
                 owner: str = "Shivam", max_tokens: int = 4096):
        self.store, self.queue = store, queue
        self.client, self.model = client, model          # client: anthropic.AsyncAnthropic
        self.owner, self.max_tokens = owner, max_tokens

    async def run(self, raw_text: str) -> tuple[int, ExtractionResult]:
        """Extract proposals from a debrief and stage them. Returns (debrief_id, result)."""
        result = await self._extract(raw_text)
        debrief_id = await self.queue.stage(raw_text, result)
        return debrief_id, result

    async def _context(self, raw_text: str) -> str:
        lines = ["ROSTER (id | name | aliases | relationship | summary):"]
        for p in await self.store.roster(limit=200):
            aka = ", ".join(p["aliases"]) or "-"
            lines.append(f"existing:{p['id']} | {p['name']} | {aka} | "
                         f"{p['relationship'] or '-'} | {p['summary'] or '-'}")
        if len(lines) == 1:
            lines.append("(empty — everyone is new)")

        # Only people this debrief mentions, and only non-sensitive facts,
        # so the minimum amount of existing data goes to the API.
        mentioned = await self.store.mentioned_in(raw_text)
        if mentioned:
            lines.append("\nKNOWN FACTS about people mentioned (fact id | person | fact):")
            for pid in mentioned:
                prof = await self.store.profile(pid, include_sensitive=False)
                for f in prof["facts"]:
                    lines.append(f"{f['id']} | existing:{pid} {prof['name']} | {f['text']}")
        return "\n".join(lines)

    async def _extract(self, raw_text: str) -> ExtractionResult:
        system = SYSTEM.format(owner=self.owner, today=date.today().isoformat())
        user = f"{await self._context(raw_text)}\n\nDEBRIEF:\n{raw_text}"

        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[{
                "name": TOOL_NAME,
                "description": "Record proposals extracted from the debrief.",
                "input_schema": ExtractionResult.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": user}],
        )
        # A truncated tool call silently drops its last fields — fail loudly instead.
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("Debrief extraction hit max_tokens — raise max_tokens or split the debrief.")
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        if block is None:
            raise RuntimeError("Model did not return proposals.")
        return self._lenient(block.input)

    @staticmethod
    def _lenient(raw: dict) -> ExtractionResult:
        """Validate item by item — one bad item shouldn't sink the whole debrief."""
        clean = {}
        for kind, field in RESULT_FIELDS.items():
            items = []
            for item in raw.get(field) or []:
                try:
                    items.append(PAYLOAD_MODELS[kind].model_validate(item))
                except ValidationError as e:
                    print(f"[people] dropped invalid {kind}: {e.errors()[0]['msg']} — {item}")
            clean[field] = items
        return ExtractionResult(**clean)
