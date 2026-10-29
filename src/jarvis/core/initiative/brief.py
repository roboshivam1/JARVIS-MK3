# =============================================================================
# src/jarvis/core/initiative/brief.py - the morning brief
# =============================================================================
#
# One message, once a day, that tells the owner what needs him. Not a
# report of everything that happened - a list of the few things that
# want a decision, a reply or a glance, in the order he should care.
#
# GATHERED BY CODE, PHRASED BY A MODEL. Every fact comes from a query
# here; the model's only job is to turn the facts into three or four
# sentences in JARVIS's voice. It is told to add nothing, and if it is
# unavailable - no key, an outage, a refusal - the same facts go out as
# a plain template. The brief never fails for want of a model.
#
# EMPTY SECTIONS ARE LEFT OUT. A brief that says "no approvals, no
# watcher hits, no reels" every day trains the owner to stop reading it.
# A quiet morning gets one line saying so.
#
# DELIVERED AS A NOTIFICATION, not sent directly, so quiet hours and a
# /quiet snooze apply to it like everything else, and it lands in the
# Telegram thread as an assistant turn - JARVIS knows what he told the
# owner this morning when the owner replies to it.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.common.notifications import Notification
from jarvis.common.reels import EpisodeStatus
from jarvis.common.sessions import Turn, TurnRole
from jarvis.core.db.database import Database
from jarvis.core.db.repos.approvals import ApprovalsRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.people import PeopleRepo
from jarvis.core.db.repos.projects import ProjectsRepo
from jarvis.core.db.repos.reels import ReelsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.db.repos.watchers import WatchersRepo
from jarvis.core.people.service import _next_occurrence
from jarvis.llm.layer import user_message
from jarvis.llm.tiers import Tier

if TYPE_CHECKING:
    from jarvis.core.life.service import LifeService
    from jarvis.core.money.service import MoneyService
    from jarvis.llm.layer import LLMLayer

log = get_logger("core.initiative.brief")

BRIEF_JOB_TYPE = "brief.morning"
BRIEF_SCHEDULE = "morning brief"
BRIEF_ACTOR = "core.brief"
BRIEF_CLIENT = "telegram"

# Work that runs every night whether or not anything is wrong. Its
# successes are noise in a brief; its failures are not.
_ROUTINE = frozenset({
    "memory.sleep_cycle", "system.backup", "watch.check", BRIEF_JOB_TYPE,
})
# Reels report under their own heading, not as jobs.
_OWN_SECTION = frozenset({"reel.make"})

_DAYS_AHEAD = 7
_LOOKBACK = timedelta(hours=24)
_BACKUP_GRACE = timedelta(hours=26)
_MAX_ITEMS = 5
_STALE_DAYS = 14

_SYSTEM = """\
You are JARVIS, writing your employer's morning brief. He is "sir". \
Your voice: dry, precise, unhurried British butler, warm under the \
formality. The brief goes to Telegram as plain text.

Rules:
- Use ONLY the facts given. Never add, guess, round up or soften a fact.
- Open with a short greeting that includes the day and date.
- If there is a "Checklist for today" section, give it right after the greeting as "- " lines, every item, worded exactly as given. Then yesterday in one sentence.
- Lead with what needs him: approvals, reviews, things he owes or is \
owed, anything that failed. Then dates coming up. Running costs last.
- One short paragraph per section, or a few "- " lines where a list is \
clearer. No headings, no markdown, no emoji.
- Under 200 words. If the facts are thin, be brief - do not pad.
- Close with one line at most. No questions unless a fact asks for one."""


@dataclass
class BriefFacts:
    """Everything the brief can say, one list of lines per section."""

    today: date
    sections: dict[str, list[str]] = field(default_factory=dict)

    def add(self, name: str, lines: list[str]) -> None:
        if lines:
            self.sections[name] = lines

    @property
    def empty(self) -> bool:
        return not self.sections

    def as_text(self) -> str:
        """The facts as the model sees them, and as the fallback sends them."""
        blocks = [f"{name}:\n" + "\n".join(f"- {line}" for line in lines)
                  for name, lines in self.sections.items()]
        return "\n\n".join(blocks)


class BriefService:
    """Gathers the facts, phrases them, and queues the message."""

    def __init__(
        self,
        db: Database,
        *,
        people: PeopleRepo | None,
        approvals: ApprovalsRepo,
        watchers: WatchersRepo,
        sessions: SessionsRepo,
        notifications: NotificationsRepo,
        llm: "LLMLayer | None",
        tz: ZoneInfo,
        projects: ProjectsRepo | None = None,
        life: "LifeService | None" = None,
        money: "MoneyService | None" = None,
    ) -> None:
        self._db = db
        self._people = people
        self._approvals = approvals
        self._watchers = watchers
        self._reels = ReelsRepo(db)
        self._sessions = sessions
        self._notifications = notifications
        self._llm = llm
        self._tz = tz
        self._projects = projects
        self._life = life
        self._money = money

    # -- the whole thing ------------------------------------------------------

    async def compose(self, trace_id: str, now: datetime | None = None) -> tuple[str, BriefFacts]:
        """The brief's text, and the facts it was built from."""
        facts = await self.gather(now)
        return await self.phrase(facts, trace_id), facts

    async def send(
        self, trace_id: str, *, job_id: str | None = None, now: datetime | None = None,
    ) -> tuple[str, BriefFacts]:
        """Compose and queue it for Telegram, in the default thread. With a
        job id, a retried job cannot queue the same brief twice - the
        notifications table allows one message per job."""
        text, facts = await self.compose(trace_id, now)
        session = await self._sessions.get_or_create_default(BRIEF_CLIENT)
        await self._notifications.create(Notification(
            client_kind=BRIEF_CLIENT,
            text=text,
            session_id=session.id,
            job_id=job_id,
            priority=3,          # deferred by quiet hours and /quiet, never digested
            trace_id=trace_id,
        ))
        log.info("morning brief queued", extra={
            "trace_id": trace_id, "sections": list(facts.sections)})
        return text, facts

    async def on_demand(self, trace_id: str) -> str:
        """For /brief: composed now and answered directly, past quiet
        hours - he asked. Recorded in the thread like the scheduled one."""
        text, _ = await self.compose(trace_id)
        session = await self._sessions.get_or_create_default(BRIEF_CLIENT)
        await self._sessions.append_turn(Turn(
            session_id=session.id, role=TurnRole.ASSISTANT, content=text,
        ))
        return text

    # -- gathering ------------------------------------------------------------

    async def gather(self, now: datetime | None = None) -> BriefFacts:
        # UTC throughout: stored timestamps are UTC ISO strings, compared as text.
        moment = (now or utc_now()).astimezone(timezone.utc)
        local = moment.astimezone(self._tz)
        facts = BriefFacts(today=local.date())
        since = moment - _LOOKBACK

        if self._life is not None:
            try:
                for name, lines in (await self._life.brief_sections(moment)).items():
                    facts.add(name, lines)
            except Exception:
                log.warning("life sections unavailable", exc_info=True)
        facts.add("Waiting on sir", await self._waiting(since))
        facts.add("Commitments", await self._commitments(local.date()))
        facts.add("Coming up", await self._dates(local.date()))
        facts.add("Overnight", await self._overnight(moment, since))
        facts.add("Projects", await self._project_lines(moment, local.date()))
        facts.add("Reels", await self._reel_lines())
        if self._money is not None:
            try:
                facts.add("Money", await self._money.brief_lines(moment))
            except Exception:
                log.warning("money lines unavailable", exc_info=True)
        facts.add("JARVIS running costs", await self._spend(local))
        return facts

    async def _waiting(self, since: datetime) -> list[str]:
        lines: list[str] = []
        pending = await self._approvals.all_pending()
        for request in pending[:_MAX_ITEMS]:
            lines.append(f"approval waiting: {request.summary}")
        if len(pending) > _MAX_ITEMS:
            lines.append(f"and {len(pending) - _MAX_ITEMS} more approvals")
        if self._people is not None:
            debriefs = await self._people.pending_debriefs()
            count = sum(len(d["proposals"]) for d in debriefs)
            if count:
                lines.append(f"{count} people-database proposals to review "
                             f"from {len(debriefs)} debriefs")
        return lines

    async def _commitments(self, today: date) -> list[str]:
        """Overdue or due within the week; undated ones only as a count."""
        if self._people is None:
            return []
        horizon = today + timedelta(days=_DAYS_AHEAD)
        lines: list[str] = []
        undated = 0
        for c in await self._people.open_commitments():
            if not c["due_date"]:
                undated += 1
                continue
            try:
                due = date.fromisoformat(c["due_date"])
            except ValueError:
                continue
            if due > horizon:
                continue
            who = (f"sir owes {c['name']}" if c["direction"] == "i_owe"
                   else f"{c['name']} owes sir")
            when = ("OVERDUE since " if due < today else "due ") + due.strftime("%a %d %b")
            lines.append(f"{who}: {c['what']} ({when})")
        if undated:
            lines.append(f"{undated} other open commitments with no date")
        return lines[: _MAX_ITEMS + 1]

    async def _dates(self, today: date) -> list[str]:
        if self._people is None:
            return []
        dated: list[tuple[date, str]] = []
        for d in await self._people.all_dates():
            when = _next_occurrence(d["date"], bool(d["recurring"]), today)
            if when is None or not 0 <= (when - today).days <= _DAYS_AHEAD:
                continue
            day = "today" if when == today else when.strftime("%a %d %b")
            dated.append((when, f"{d['name']}: {d['label']} ({day})"))
        return [line for _, line in sorted(dated)][:_MAX_ITEMS]

    async def _overnight(self, now: datetime, since: datetime) -> list[str]:
        lines: list[str] = []

        for watcher in await self._watchers.all():
            hit = watcher.last_hit_ts
            if hit is not None and hit >= since:
                about = f" - {watcher.note}" if watcher.note else ""
                lines.append(f"watcher '{watcher.name}' fired{about}")

        rows = await self._db.query(
            "SELECT type, status, result, error, session_id FROM jobs "
            "WHERE finished_ts >= ? AND status IN ('succeeded', 'failed') "
            "ORDER BY finished_ts", (since.isoformat(),))
        finished = 0
        for row in rows:
            if row["type"] in _OWN_SECTION or row["type"] == BRIEF_JOB_TYPE:
                continue
            if row["status"] == "failed":
                reason = (row["error"] or "no reason recorded").splitlines()[0][:120]
                lines.append(f"FAILED: {row['type']} - {reason}")
            elif row["type"] not in _ROUTINE:
                finished += 1
        if finished:
            lines.append(f"{finished} jobs finished in the last day")

        backup = await self._db.query_one(
            "SELECT MAX(finished_ts) AS ts FROM jobs "
            "WHERE type = 'system.backup' AND status = 'succeeded'")
        last = backup["ts"] if backup is not None else None
        if last is None or datetime.fromisoformat(last) < now - _BACKUP_GRACE:
            lines.append("no successful backup in the last day")
        return lines

    async def _project_lines(self, now: datetime, today: date) -> list[str]:
        """Project tasks due this week, and live projects gone quiet."""
        if self._projects is None:
            return []
        lines: list[str] = []
        horizon = (today + timedelta(days=_DAYS_AHEAD)).isoformat()
        for t in (await self._projects.tasks_due(horizon))[:_MAX_ITEMS]:
            due = date.fromisoformat(t["due_date"])
            when = ("OVERDUE since " if due < today else "due ") + due.strftime("%a %d %b")
            lines.append(f"{t['name']}: {t['text']} ({when})")
        for p in (await self._projects.stale(_STALE_DAYS, now.isoformat()))[:3]:
            days = (today - date.fromisoformat(p["last"][:10])).days
            lines.append(f"{p['name']} is active but untouched for {days} days")
        pending = await self._projects.pending_count()
        if pending:
            lines.append(f"{pending} projects-database proposals to review")
        return lines

    async def _reel_lines(self) -> list[str]:
        lines: list[str] = []
        for episode in await self._reels.recent(limit=20):
            name = f"'{episode.title}'" if episode.title else episode.ref
            if episode.status is EpisodeStatus.AWAITING_REVIEW:
                lines.append(f"reel {name} is waiting for his review")
            elif episode.status is EpisodeStatus.DELIVERED:
                lines.append(f"reel {name} is delivered but not marked posted")
            elif episode.status in (EpisodeStatus.DRAFTING, EpisodeStatus.RENDERING):
                lines.append(f"reel {name} is {episode.status.value}")
        return lines[:_MAX_ITEMS]

    async def _spend(self, local: datetime) -> list[str]:
        start_today = local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_yday = start_today - timedelta(days=1)
        start_month = start_today.replace(day=1)

        async def total(start: datetime, end: datetime) -> tuple[float, int]:
            row = await self._db.query_one(
                "SELECT COALESCE(SUM(cost_inr), 0) AS inr, COUNT(*) AS n "
                "FROM llm_calls WHERE ts >= ? AND ts < ?",
                (_utc_iso(start), _utc_iso(end)))
            return (float(row["inr"]), int(row["n"])) if row is not None else (0.0, 0)

        yday, calls = await total(start_yday, start_today)
        month, _ = await total(start_month, start_today)
        if not calls and not month:
            return []
        return [f"yesterday: Rs {yday:.0f} over {calls} model calls; "
                f"this month so far: Rs {month:.0f}"]

    # -- phrasing -------------------------------------------------------------

    async def phrase(self, facts: BriefFacts, trace_id: str) -> str:
        if facts.empty:
            return (f"Good morning, sir. {_day(facts.today)}. Nothing needs you "
                    f"this morning - no approvals, nothing owed, nothing broken.")
        if self._llm is not None and getattr(self._llm, "available", True):
            try:
                response = await self._llm.complete(
                    Tier.UTILITY,
                    _SYSTEM,
                    [user_message(f"Today is {_day(facts.today)}.\n\n{facts.as_text()}")],
                    actor=BRIEF_ACTOR,
                    trace_id=trace_id,
                    max_tokens=600,
                )
                text = response.text.strip()
                if text and response.stop_reason != "max_tokens":
                    return text
            except Exception as exc:     # the brief must not die with the model
                log.warning("brief phrasing failed - sending the plain version",
                            extra={"trace_id": trace_id, "error": str(exc)[:200]})
        return template(facts)


def template(facts: BriefFacts) -> str:
    """The facts, plainly, for when there is no model to phrase them."""
    return f"Good morning, sir. {_day(facts.today)}.\n\n{facts.as_text()}"


def _day(today: date) -> str:
    return today.strftime("%A %d %B").replace(" 0", " ")


def _utc_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def describe_facts(facts: BriefFacts) -> dict[str, Any]:
    """The job result: which sections said something, and how much."""
    return {name: len(lines) for name, lines in facts.sections.items()}

