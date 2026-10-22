# =============================================================================
# src/jarvis/core/projects/service.py - talk about work, get records
# =============================================================================
#
# The owner talks about what he is building; this turns it into
# proposals for the review queue, and reads projects back as text for
# conversation. The people service's design, for projects:
#
# HANDLES, NOT IDS. The model sees projects as J1, J2..., and the known
# records of the projects this debrief mentions as D (decisions), F
# (facts), B (parts - the bill of materials) and T (open tasks), plus
# people from the people database as H. Translated back to ids here;
# anything pointing at a handle that does not exist is dropped.
#
# NOTHING IS WRITTEN WITHOUT HIM. debrief() only stages proposals. The
# repo's approve() is driven from /app/projects.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from jarvis.common.log import get_logger
from jarvis.common.projects import (
    PROPOSAL_MODELS,
    RESULT_FIELDS,
    DecisionProposal,
    ExtractionResult,
    FactProposal,
    NewProjectProposal,
    PartUpdateProposal,
    PersonProposal,
    TaskDoneProposal,
)
from jarvis.core.db.repos.projects import ProjectsRepo
from jarvis.llm.layer import LLMLayer, ToolSpec, user_message
from jarvis.llm.tiers import Tier

log = get_logger("core.projects")

ACTOR = "core.projects"
_TOOL = "record_proposals"
_TIER = Tier.UTILITY
_PEOPLE_LIMIT = 150

_SYSTEM = """You turn the owner's debriefs about what he is building into \
proposals for his projects database. He reviews every proposal before it is \
stored, so be complete but never invent.

Rules:
- Extract only what he states or directly implies.
- A project in PROJECTS: project_ref "existing:<handle>", e.g. existing:J2. \
A project not there: a new_projects entry tagged N1, N2... and referenced as \
"new:N1". If unsure whether it is a listed project, make it new and put \
"possibly J2" in its pitch - he decides.
- Other names he uses for a project become aliases.
- decisions: a choice made, with the reason and the options turned down when \
he gives them. A decision that replaces a KNOWN DECISION sets \
supersedes_decision_id to its handle (D1).
- facts: one atomic fact per item - specs, constraints, current status, \
lessons learned, risks, ideas not yet decided. One replacing a KNOWN FACT \
sets supersedes_fact_id (F1).
- parts: components, materials or paid services to buy or use. A change to \
a KNOWN PART (arrived, ordered, dropped) is a part_update on its handle (B1).
- tasks: things still to do. A KNOWN OPEN TASK he says is finished is a \
tasks_done entry with its handle (T1).
- log: what happened, dated - one entry per event. Use today's date unless \
he gives another; relative dates become absolute.
- people: someone from PEOPLE working on or helping with it, by handle (H3), \
with their role. Anyone not in PEOPLE: leave them out.
- status_changes: only when he says the project is paused, shipped, \
abandoned, back on, or was just an idea.
- links: only URLs he actually gives.
- Nothing worth storing: return empty lists.

Today is {today}."""


class ExtractionFailed(RuntimeError):
    """The model call returned nothing usable. Nothing was staged."""


@dataclass(frozen=True)
class DebriefOutcome:
    debrief_id: str
    staged: int
    dropped: int

    def summary(self) -> str:
        if self.staged == 0:
            return "Nothing in that was worth storing in the projects database."
        note = f" ({self.dropped} unusable item(s) dropped)" if self.dropped else ""
        return f"{self.staged} project proposal(s) waiting for review{note}."


@dataclass
class _Handles:
    projects: dict[str, str]
    decisions: dict[str, str]
    facts: dict[str, str]
    parts: dict[str, str]
    tasks: dict[str, str]
    people: dict[str, str]


class ProjectService:
    def __init__(self, repo: ProjectsRepo, llm: LLMLayer, tz: ZoneInfo) -> None:
        self._repo = repo
        self._llm = llm
        self._tz = tz

    # -- debrief -> staged proposals -----------------------------------------

    async def debrief(self, raw_text: str, *, trace_id: str,
                      source: str = "web") -> DebriefOutcome:
        handles, context = await self._context(raw_text)
        raw = await self._call_model(raw_text, context, trace_id)
        parsed, dropped = self._parse(raw)
        result, dangling = self._translate(parsed, handles)
        debrief_id = await self._repo.stage(raw_text, result, source)
        return DebriefOutcome(debrief_id, result.count, dropped + dangling)

    async def _context(self, raw_text: str) -> tuple[_Handles, str]:
        h = _Handles({}, {}, {}, {}, {}, {})
        lines = ["PROJECTS (handle | name | aliases | status | pitch):"]
        for i, p in enumerate(await self._repo.roster(limit=100), start=1):
            h.projects[f"J{i}"] = p["id"]
            lines.append(f"J{i} | {p['name']} | {', '.join(p['aliases']) or '-'} | "
                         f"{p['status']} | {p['pitch'] or '-'}")
        if not h.projects:
            lines.append("(empty - every project is new)")
        handle_of = {pid: k for k, pid in h.projects.items()}

        known: list[str] = []
        mentioned = await self._repo.mentioned_in(raw_text)
        for pid in mentioned[:4]:
            prof = await self._repo.profile(pid)
            if prof is None or pid not in handle_of:
                continue
            tag = f"{handle_of[pid]} {prof['name']}"
            for d in prof["decisions"]:
                key = f"D{len(h.decisions) + 1}"
                h.decisions[key] = d["id"]
                known.append(f"{key} | {tag} | decision: {d['decision']}")
            for f in prof["facts"]:
                key = f"F{len(h.facts) + 1}"
                h.facts[key] = f["id"]
                known.append(f"{key} | {tag} | {f['category']}: {f['text']}")
            for b in prof["parts"]:
                key = f"B{len(h.parts) + 1}"
                h.parts[key] = b["id"]
                known.append(f"{key} | {tag} | part ({b['status']}): {b['name']}")
            for t in prof["tasks"]:
                key = f"T{len(h.tasks) + 1}"
                h.tasks[key] = t["id"]
                known.append(f"{key} | {tag} | open task: {t['text']}")
        if known:
            lines += ["", "KNOWN RECORDS of the projects mentioned (handle | project | record):",
                      *known]

        people = await self._repo.people_names(_PEOPLE_LIMIT)
        if people:
            lines += ["", "PEOPLE (handle | name):"]
            for i, person in enumerate(people, start=1):
                h.people[f"H{i}"] = person["id"]
                lines.append(f"H{i} | {person['name']}")
        return h, "\n".join(lines)

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
        if response.stop_reason == "max_tokens":
            raise ExtractionFailed("that was too long to extract in one go - split it")
        if not response.tool_calls:
            raise ExtractionFailed("the model returned no proposals")
        return response.tool_calls[0].args

    @staticmethod
    def _parse(raw: dict[str, Any]) -> tuple[ExtractionResult, int]:
        """Item by item: one malformed proposal must not sink the rest."""
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
                    log.warning("dropped invalid project proposal", extra={
                        "kind": kind.value, "reason": exc.errors()[0]["msg"]})
        return ExtractionResult(**clean), dropped

    @staticmethod
    def _translate(result: ExtractionResult, h: _Handles) -> tuple[ExtractionResult, int]:
        """Handles -> ids; anything dangling is dropped, and counted."""
        dangling = 0
        tags: set[str] = set()
        new_projects: list[NewProjectProposal] = []
        for project in result.new_projects:
            if project.tag in tags:
                dangling += 1
                continue
            tags.add(project.tag)
            new_projects.append(project)

        def fix_ref(ref: str) -> str | None:
            kind, _, value = ref.partition(":")
            if kind == "existing":
                pid = h.projects.get(value)
                return f"existing:{pid}" if pid else None
            return ref if value in tags else None

        out: dict[str, list[Any]] = {"new_projects": new_projects}
        for kind, field in RESULT_FIELDS.items():
            if field == "new_projects":
                continue
            kept: list[Any] = []
            for item in getattr(result, field):
                update: dict[str, Any] = {}
                if hasattr(item, "project_ref"):
                    ref = fix_ref(item.project_ref)
                    if ref is None:
                        dangling += 1
                        continue
                    update["project_ref"] = ref
                if isinstance(item, DecisionProposal) and item.supersedes_decision_id:
                    update["supersedes_decision_id"] = h.decisions.get(
                        item.supersedes_decision_id)
                if isinstance(item, FactProposal) and item.supersedes_fact_id:
                    update["supersedes_fact_id"] = h.facts.get(item.supersedes_fact_id)
                if isinstance(item, PartUpdateProposal):
                    part = h.parts.get(item.part_id)
                    if part is None:
                        dangling += 1
                        continue
                    update["part_id"] = part
                if isinstance(item, TaskDoneProposal):
                    task = h.tasks.get(item.task_id)
                    if task is None:
                        dangling += 1
                        continue
                    update["task_id"] = task
                if isinstance(item, PersonProposal):
                    person = h.people.get(item.person_id)
                    if person is None:
                        dangling += 1
                        continue
                    update["person_id"] = person
                kept.append(item.model_copy(update=update))
            out[field] = kept
        return ExtractionResult(**out), dangling

    # -- reading projects back, for conversation -----------------------------

    async def describe(self, project_id: str, full: bool = True) -> str:
        p = await self._repo.profile(project_id)
        if p is None:
            return ""
        head = f"{p['name']} [{p['status']}, {p['kind']}]"
        if p["aliases"]:
            head += f" (aka {', '.join(p['aliases'])})"
        lines = [head]
        for label, key in (("What", "pitch"), ("Goal", "goal"), ("About", "description"),
                           ("Repo", "repo_url"), ("Folder", "folder")):
            if p.get(key):
                lines.append(f"{label}: {p[key]}")
        if p["people"]:
            lines.append("People: " + ", ".join(
                f"{x['name']}" + (f" ({x['role']})" if x["role"] else "") for x in p["people"]))
        decisions = p["decisions"] if full else p["decisions"][-5:]
        for d in decisions:
            why = f" - because {d['why']}" if d["why"] else ""
            alt = f" (instead of {d['alternatives']})" if d["alternatives"] and full else ""
            lines.append(f"- Decided: {d['decision']}{why}{alt}")
        facts = p["facts"] if full else p["facts"][:8]
        for f in facts:
            lines.append(f"- {f['category'].capitalize()}: {f['text']}")
        for t in p["tasks"][: None if full else 5]:
            due = f" (due {t['due_date']})" if t["due_date"] else ""
            lines.append(f"- To do: {t['text']}{due}")
        if full:
            for b in p["parts"]:
                cost = f", Rs {b['cost_inr']:.0f}" if b["cost_inr"] is not None else ""
                spec = f" {b['spec']}" if b["spec"] else ""
                lines.append(f"- Part: {b['quantity']} x {b['name']}{spec} [{b['status']}{cost}]")
            for r in p["relations"]:
                arrow = "->" if r["direction"] == "out" else "<-"
                lines.append(f"- {r['kind'].replace('_', ' ')} {arrow} {r['other']}")
            for link in p["links"]:
                lines.append(f"- Link: {link['label']}: {link['url']}")
        recent = p["log"][: 10 if full else 3]
        if recent:
            lines.append("Recent:")
            lines += [f"- {e['happened_on']}: {e['text']}" for e in recent]
        if full and p["documents"]:
            lines.append(f"Master document: version {p['documents'][0]['version']}.")
        return "\n".join(lines)

    async def mention_context(self, text: str, limit: int = 2) -> str:
        """The projects this message names, briefly, for the system prompt."""
        ids = await self._repo.mentioned_in(text)
        blocks = [await self.describe(pid, full=False) for pid in ids[:limit]]
        return "\n\n".join(b for b in blocks if b)

    async def lookup(self, name: str) -> str:
        ids = [p["id"] for p in await self._repo.find(name)]
        if not ids:
            ids = await self._repo.mentioned_in(name)
        if not ids:
            ids = await self._repo.search(name, limit=3)
            if ids:
                blocks = [await self.describe(pid) for pid in ids]
                return (f"No project is called {name!r}; these mention it:\n\n"
                        + "\n\n".join(blocks))
            return f"No project matching {name!r} is in the projects database."
        return "\n\n".join([await self.describe(pid) for pid in ids[:3]])

    async def overview(self) -> str:
        rows = await self._repo.directory()
        if not rows:
            return "The projects database is empty."
        lines = []
        for r in rows:
            extras = []
            if r["open_tasks"]:
                extras.append(f"{r['open_tasks']} open tasks")
            if r["parts_waiting"]:
                extras.append(f"{r['parts_waiting']} parts to get")
            tail = f" ({', '.join(extras)})" if extras else ""
            pitch = f" - {r['pitch']}" if r["pitch"] else ""
            lines.append(f"- {r['name']} [{r['status']}]{pitch}{tail}")
        pending = await self._repo.pending_count()
        if pending:
            lines.append(f"{pending} proposal(s) waiting for review at /app/projects.")
        return "\n".join(lines)

    async def find(self, name: str) -> list[dict[str, Any]]:
        return await self._repo.find(name)

    async def note(self, name: str, text: str) -> str:
        """A log line the owner dictated himself - no review needed."""
        found = await self._repo.find(name)
        if len(found) != 1:
            return (f"No project called {name!r}." if not found
                    else f"{len(found)} projects match {name!r} - use the exact name.")
        await self._repo.add_log(found[0]["id"], text, source="owner",
                                 happened_on=datetime.now(self._tz).date().isoformat())
        return f"Logged on {found[0]['name']}."
