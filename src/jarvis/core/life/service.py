# =============================================================================
# src/jarvis/core/life/service.py - reports in, a picture of the days out
# =============================================================================
#
# The owner says what happened in his own words; one cheap model call
# turns it into rows; this file stores them and reads them back as the
# checklist, the evening check-in, the brief's lines and the trends.
#
# STRAIGHT IN, UNDOABLE. No review queue - it is his own day. Every row a
# report writes carries the report's id, and undo() takes back exactly
# the last message's worth.
#
# THE CHECKLIST is the spine. Each day gets its fixed targets (gym, Mon to
# Sat) copied in from habits, plus one-off targets he sets ("tomorrow:
# one Coursera module"). Logging a gym session ticks the gym item by
# itself; anything else he ticks by saying it is done. When a day ends,
# whatever is still open becomes "missed" - that is the data he asked for.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from jarvis.common.life import WEEKDAYS, ActivityItem, ExtractedReport, clock
from jarvis.common.log import get_logger
from jarvis.core.db.repos.life import LifeRepo
from jarvis.llm.layer import ToolSpec, user_message
from jarvis.llm.tiers import Tier

if TYPE_CHECKING:
    from jarvis.core.db.repos.projects import ProjectsRepo
    from jarvis.llm.layer import LLMLayer

log = get_logger("core.life")

ACTOR = "core.life"
_TOOL = "record_report"
WAKE_BY = "wake_by"
# A night owl's "I slept at 1:30" belongs to the day before.
_NIGHT_ENDS = 5

_SYSTEM = """You log the owner's day from what he tells you. Fill the record \
tool with exactly what he said happened - never guess or add.

- wake: when he got up. sleep: when he went to bed, dated to the day that \
was ending (sleeping at 01:30 on the 22nd belongs to the 21st).
- activities: kind is gym, study (college coursework), project (building \
one of his PROJECTS - name it in project), learning (self-study: online \
courses, tutorials, reading) or other. action "start" if he is beginning \
it now, "stop" if he has just finished something he started earlier, \
"log" for something already done. Give minutes or start/end times when \
he says them; "for two hours" is 120 minutes.
- meals: breakfast, lunch, dinner or snack; status skipped if he says he \
skipped it; what and where (home, mess, out) when said.
- done: handles from CHECKLIST of items he says are finished (C2). Only \
items he clearly finished.
- new_targets: things he says he wants to do on a given day ("tomorrow I \
want to finish one Coursera module") - dated to that day.
- day_note: only if he describes how his day went in general.
- Relative dates and times become absolute using NOW. "Just now" is NOW.
- Nothing loggable: return everything empty.

NOW: {now}"""


class ReportFailed(RuntimeError):
    """The model returned nothing usable. Nothing was stored."""


@dataclass(frozen=True)
class ReportOutcome:
    report_id: str | None
    lines: list[str]

    def summary(self) -> str:
        if not self.lines:
            return "Nothing in that to log."
        return "Logged: " + "; ".join(self.lines) + "."


def _minutes(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    span = (eh * 60 + em) - (sh * 60 + sm)
    if span <= 0:
        span += 1440
    return span if span < 1440 else None


def _hours(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h}h{m:02d}" if h else f"{m}m"


class LifeService:
    def __init__(self, repo: LifeRepo, llm: "LLMLayer | None", tz: ZoneInfo,
                 projects: "ProjectsRepo | None" = None) -> None:
        self._repo = repo
        self._llm = llm
        self._tz = tz
        self._projects = projects

    # -- the clock ------------------------------------------------------------

    def now(self) -> datetime:
        return datetime.now(self._tz)

    def _local(self, moment: datetime | None) -> datetime:
        return (moment or self.now()).astimezone(self._tz)

    async def prepare(self, moment: datetime | None = None) -> str:
        """Close finished days, lay out today's and tomorrow's checklists.
        Returns today's date. Cheap and idempotent - called before any read."""
        local = self._local(moment)
        today = local.date()
        await self._repo.close_past(today.isoformat())
        await self._repo.ensure_checklist(today.isoformat())
        await self._repo.ensure_checklist((today + timedelta(days=1)).isoformat())
        return today.isoformat()

    # -- reports --------------------------------------------------------------

    async def report(self, text: str, *, trace_id: str, source: str = "conversation",
                     moment: datetime | None = None) -> ReportOutcome:
        local = self._local(moment)
        today = await self.prepare(local)
        tomorrow = (local.date() + timedelta(days=1)).isoformat()
        handles: dict[str, str] = {}
        lines = ["CHECKLIST (handle | day | item | status):"]
        for day in (today, tomorrow):
            for item in await self._repo.checklist(day):
                key = f"C{len(handles) + 1}"
                handles[key] = item["id"]
                label = "today" if day == today else "tomorrow"
                lines.append(f"{key} | {label} | {item['text']} | {item['status']}")
        running = await self._repo.running()
        if running:
            lines += ["", "RUNNING NOW:"] + [
                f"- {r['kind']}{' ' + r['subject'] if r['subject'] else ''} since "
                f"{r['started_at']}" for r in running]
        if self._projects is not None:
            names = [p["name"] for p in await self._projects.roster(limit=40)]
            if names:
                lines += ["", "PROJECTS: " + ", ".join(names)]

        raw = await self._call_model(text, "\n".join(lines), local, trace_id)
        report = self._parse(raw)
        if report.empty:
            return ReportOutcome(None, [])
        report_id = await self._repo.new_report(text, source)
        recorded = await self._apply(report, handles, report_id, local)
        outcome = ReportOutcome(report_id, recorded)
        await self._repo.set_report_summary(report_id, outcome.summary())
        return outcome

    async def _call_model(self, text: str, context: str, local: datetime,
                          trace_id: str) -> dict[str, Any]:
        if self._llm is None:
            raise ReportFailed("no model is available to read that")
        response = await self._llm.complete(
            Tier.UTILITY,
            _SYSTEM.format(now=local.strftime("%A %Y-%m-%d %H:%M")),
            [user_message(f"{context}\n\nHE SAYS:\n{text}")],
            actor=ACTOR, trace_id=trace_id,
            tools=[ToolSpec(name=_TOOL, description="Record what he reported.",
                            input_schema=ExtractedReport.model_json_schema())],
            tool_choice={"type": "tool", "name": _TOOL},
            max_tokens=1500,
        )
        if response.stop_reason == "max_tokens" or not response.tool_calls:
            raise ReportFailed("that did not come back whole - say it again, shorter")
        return response.tool_calls[0].args

    @staticmethod
    def _parse(raw: dict[str, Any]) -> ExtractedReport:
        """Field by field, item by item: one bad item must not lose the rest."""
        clean: dict[str, Any] = {}
        for key in ("wake", "sleep", "day_note"):
            if raw.get(key):
                try:
                    clean[key] = getattr(ExtractedReport.model_validate({key: raw[key]}), key)
                except ValidationError:
                    log.warning("dropped a life item", extra={"field": key})
        for key in ("activities", "meals", "new_targets"):
            items = raw.get(key) or []
            kept = []
            for item in items if isinstance(items, list) else []:
                try:
                    kept.append(getattr(ExtractedReport.model_validate({key: [item]}), key)[0])
                except ValidationError as exc:
                    log.warning("dropped a life item", extra={
                        "field": key, "reason": exc.errors()[0]["msg"]})
            clean[key] = kept
        done = raw.get("done") or []
        clean["done"] = [str(h) for h in done if isinstance(h, str)] if isinstance(done, list) else []
        return ExtractedReport(**clean)

    async def _apply(self, report: ExtractedReport, handles: dict[str, str],
                     report_id: str, local: datetime) -> list[str]:
        now_clock = local.strftime("%H:%M")
        lines: list[str] = []
        if report.wake:
            await self._repo.set_day(report.wake.date, "wake", report.wake.time, report_id)
            lines.append(f"up at {report.wake.time}")
        if report.sleep:
            await self._repo.set_day(report.sleep.date, "sleep", report.sleep.time, report_id)
            lines.append(f"slept at {report.sleep.time}")
        for item in report.activities:
            line = await self._activity(item, report_id, now_clock)
            if line:
                lines.append(line)
        for meal in report.meals:
            await self._repo.add_meal(meal.date, meal.slot, meal.status, meal.what,
                                      meal.place, report_id)
            lines.append(f"{meal.slot} {'skipped' if meal.status == 'skipped' else 'eaten'}")
        for handle in report.done:
            item_id = handles.get(handle)
            if item_id:
                await self._repo.set_item(item_id, "done", report_id)
                lines.append(f"ticked {handle}")
        for target in report.new_targets:
            await self._repo.ensure_checklist(target.date)
            await self._repo.add_target(target.date, target.text, report_id)
            lines.append(f"target for {target.date}: {target.text}")
        if report.day_note:
            await self._repo.set_day(report.day_note.date, "note", report.day_note.text, report_id)
            lines.append("day note")
        return lines

    async def _activity(self, item: ActivityItem, report_id: str, now_clock: str) -> str:
        project_id = None
        if item.project and self._projects is not None:
            found = await self._projects.find(item.project)
            project_id = found[0]["id"] if len(found) == 1 else None
        what = item.subject or item.project or item.kind

        if item.action == "stop":
            running = await self._repo.running(item.kind)
            if running:
                act = running[0]
                ended = item.ended_at or now_clock
                minutes = item.minutes or _minutes(act["started_at"], ended)
                await self._repo.finish_activity(act["id"], ended, minutes)
                return f"{act['kind']} finished ({_hours(minutes) if minutes else 'time unknown'})"
            if not item.minutes and not item.started_at:
                return f"nothing running to stop for {item.kind}"
        started = item.started_at or (now_clock if item.action == "start" else None)
        finish = item.ended_at if item.action != "start" else None
        minutes = item.minutes or _minutes(started, finish)
        await self._repo.add_activity(
            item.date, item.kind, subject=item.subject, project_id=project_id,
            started_at=started, ended_at=finish, minutes=minutes, note=item.note,
            report_id=report_id)
        ticked = await self._repo.tick_habits_of_kind(item.date, item.kind, report_id)
        tail = f" ({_hours(minutes)})" if minutes else ""
        verb = "started" if item.action == "start" else "logged"
        tick = f", ticked {', '.join(ticked)}" if ticked else ""
        return f"{item.kind} {verb}: {what}{tail}{tick}"

    async def undo_last(self) -> str:
        last = await self._repo.last_report()
        if last is None:
            return "There is nothing to undo."
        await self._repo.undo(last["id"])
        return f"Undone: {last['summary'] or last['raw_text']}"

    # -- phone automations ----------------------------------------------------

    async def ping(self, event: str, moment: datetime | None = None) -> str:
        """One tap or trigger from the iPhone: wake, sleep, gym_start, gym_end."""
        local = self._local(moment)
        await self.prepare(local)
        hhmm = local.strftime("%H:%M")
        day = local.date()
        if event == "wake":
            await self._repo.set_day(day.isoformat(), "wake", hhmm, None)
            return f"Up at {hhmm}."
        if event == "sleep":
            if local.hour < _NIGHT_ENDS:
                day -= timedelta(days=1)
            await self._repo.set_day(day.isoformat(), "sleep", hhmm, None)
            return f"Asleep at {hhmm}."
        if event == "gym_start":
            if await self._repo.running("gym"):
                return "A gym session is already running."
            await self._repo.add_activity(day.isoformat(), "gym", subject=None,
                                          project_id=None, started_at=hhmm, ended_at=None,
                                          minutes=None, note=None, report_id=None)
            await self._repo.tick_habits_of_kind(day.isoformat(), "gym", None)
            return f"Gym started at {hhmm}."
        if event == "gym_end":
            running = await self._repo.running("gym")
            if not running:
                return "No gym session was running."
            minutes = _minutes(running[0]["started_at"], hhmm)
            await self._repo.finish_activity(running[0]["id"], hhmm, minutes)
            return f"Gym done, {_hours(minutes) if minutes else 'time unknown'}."
        raise ValueError(f"unknown event {event!r}")

    # -- reading back ---------------------------------------------------------

    async def day_detail(self, day: str) -> dict[str, Any]:
        """One day as stored. Callers run prepare() first."""
        return {
            "day": await self._repo.day(day),
            "checklist": await self._repo.checklist(day),
            "activities": await self._repo.activities(day),
            "meals": await self._repo.meals(day),
            "wake_by": await self._repo.setting(WAKE_BY),
        }

    @staticmethod
    def _checklist_lines(items: list[dict[str, Any]]) -> list[str]:
        mark = {"done": "✓", "missed": "✗", "open": "☐"}
        return [f"{mark[i['status']]} {i['text']}" for i in items]

    async def status_text(self, day: str | None = None) -> str:
        today = await self.prepare()
        day = day or today
        d = await self.day_detail(day)
        label = "Today" if day == today else date.fromisoformat(day).strftime("%a %d %b")
        lines = [f"{label}:"]
        lines += self._checklist_lines(d["checklist"]) or ["(nothing on the checklist)"]
        info = d["day"]
        lines.append(f"Woke: {info['wake_time'] or 'not logged'}"
                     + (f" (target {d['wake_by']})" if d["wake_by"] else ""))
        if info["sleep_time"]:
            lines.append(f"Slept: {info['sleep_time']}")
        totals: dict[str, int] = {}
        for a in d["activities"]:
            totals[a["kind"]] = totals.get(a["kind"], 0) + (a["minutes"] or 0)
        if totals:
            lines.append("Time: " + ", ".join(
                f"{k} {_hours(v) if v else '(running)'}" for k, v in totals.items()))
        eaten = [m["slot"] for m in d["meals"] if m["status"] == "eaten"]
        skipped = [m["slot"] for m in d["meals"] if m["status"] == "skipped"]
        lines.append("Meals: " + (", ".join(eaten) or "none logged")
                     + (f"; skipped {', '.join(skipped)}" if skipped else ""))
        return "\n".join(lines)

    async def checkin_text(self, moment: datetime | None = None) -> str:
        """The 22:00 message: how today went, what is unlogged, tomorrow's list."""
        local = self._local(moment)
        today = await self.prepare(local)
        tomorrow = local.date() + timedelta(days=1)
        d = await self.day_detail(today)
        parts = ["Evening check-in, sir.", "",
                 "Today:", *(self._checklist_lines(d["checklist"]) or ["(no checklist)"])]

        gaps: list[str] = []
        if not d["day"]["wake_time"]:
            gaps.append("when you woke")
        logged_slots = {m["slot"] for m in d["meals"]}
        for slot in ("breakfast", "lunch", "dinner"):
            if slot not in logged_slots:
                gaps.append(slot)
        if not d["activities"]:
            gaps.append("what you worked on")
        if gaps:
            parts += ["", "Not logged yet: " + ", ".join(gaps) + "."]

        upcoming = await self._repo.checklist(tomorrow.isoformat())
        parts += ["", f"Tomorrow ({tomorrow.strftime('%a %d %b')}):",
                  *([f"- {i['text']}" for i in upcoming] or ["- nothing yet"]),
                  "", "Tell me what got done, and anything to add for tomorrow."]
        return "\n".join(parts)

    async def brief_sections(self, moment: datetime | None = None) -> dict[str, list[str]]:
        """For the morning brief: today's checklist and yesterday in one line."""
        local = self._local(moment)
        today = await self.prepare(local)
        sections: dict[str, list[str]] = {}
        items = await self._repo.checklist(today)
        if items:
            sections["Checklist for today"] = [i["text"] for i in items]
        yesterday = (local.date() - timedelta(days=1)).isoformat()
        y = await self.day_detail(yesterday)
        bits: list[str] = []
        if y["day"]["wake_time"]:
            bits.append(f"woke {y['day']['wake_time']}")
        if y["checklist"]:
            done = sum(1 for i in y["checklist"] if i["status"] == "done")
            missed = [i["text"] for i in y["checklist"] if i["status"] != "done"]
            bits.append(f"{done} of {len(y['checklist'])} checklist items done"
                        + (f" (missed: {', '.join(missed)})" if missed else ""))
        minutes = sum(a["minutes"] or 0 for a in y["activities"])
        if minutes:
            bits.append(f"{_hours(minutes)} logged on activities")
        if bits:
            sections["Yesterday"] = ["; ".join(bits)]
        return sections

    async def stats(self, days: int = 28, moment: datetime | None = None) -> dict[str, Any]:
        """Per-day numbers for the trends page, oldest first."""
        local = self._local(moment)
        await self.prepare(local)
        end = local.date()
        start = end - timedelta(days=days - 1)
        rows = await self._repo.range_rows(start.isoformat(), end.isoformat())
        by_day: dict[str, dict[str, Any]] = {}
        for i in range(days):
            d = (start + timedelta(days=i)).isoformat()
            by_day[d] = {"date": d, "weekday": WEEKDAYS[(start + timedelta(days=i)).weekday()],
                         "wake": None, "sleep": None,
                         "minutes": {"gym": 0, "study": 0, "project": 0, "learning": 0, "other": 0},
                         "meals": 0, "skipped": 0, "done": 0, "total": 0, "gym_planned": False,
                         "gym_done": False}
        for r in rows["days"]:
            by_day[r["date"]].update(wake=r["wake_time"], sleep=r["sleep_time"])
        projects: dict[str, int] = {}
        for a in rows["activities"]:
            day = by_day[a["date"]]
            day["minutes"][a["kind"]] += a["minutes"] or 0
            if a["kind"] == "gym":
                day["gym_done"] = True
            if a["project"]:
                projects[a["project"]] = projects.get(a["project"], 0) + (a["minutes"] or 0)
        for m in rows["meals"]:
            by_day[m["date"]]["meals" if m["status"] == "eaten" else "skipped"] += 1
        for c in rows["checklist"]:
            day = by_day[c["date"]]
            day["total"] += 1
            day["done"] += c["status"] == "done"
            if c["habit_kind"] == "gym":
                day["gym_planned"] = True
                day["gym_done"] = day["gym_done"] or c["status"] == "done"
        return {"days": list(by_day.values()), "projects": projects,
                "wake_by": await self._repo.setting(WAKE_BY)}

    async def week_text(self) -> str:
        """The last seven days, for when he asks how his week went."""
        data = await self.stats(7)
        days = data["days"]
        wakes = [d["wake"] for d in days if d["wake"]]
        gym_planned = sum(d["gym_planned"] for d in days)
        gym_done = sum(d["gym_done"] for d in days if d["gym_planned"])
        totals = {k: sum(d["minutes"][k] for d in days)
                  for k in ("study", "project", "learning")}
        done = sum(d["done"] for d in days)
        total = sum(d["total"] for d in days)
        lines = [f"Last 7 days: {done} of {total} checklist items done."]
        if gym_planned:
            lines.append(f"Gym: {gym_done} of {gym_planned} planned days.")
        if wakes:
            avg = sum(int(w[:2]) * 60 + int(w[3:]) for w in wakes) // len(wakes)
            target = f" (target {data['wake_by']})" if data["wake_by"] else ""
            lines.append(f"Average wake time {avg // 60:02d}:{avg % 60:02d} over "
                         f"{len(wakes)} logged days{target}.")
        lines.append("Time logged: " + ", ".join(f"{k} {_hours(v)}" for k, v in totals.items()) + ".")
        if data["projects"]:
            lines.append("Projects: " + ", ".join(
                f"{k} {_hours(v)}" for k, v in sorted(data["projects"].items(),
                                                       key=lambda kv: -kv[1])) + ".")
        meals = sum(d["meals"] for d in days)
        lines.append(f"Meals logged: {meals} eaten, {sum(d['skipped'] for d in days)} skipped.")
        return "\n".join(lines)

    async def set_wake_by(self, value: str | None) -> str | None:
        clean = clock(value) if value else None
        await self._repo.set_setting(WAKE_BY, clean)
        return clean
