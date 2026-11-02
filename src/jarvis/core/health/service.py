# =============================================================================
# src/jarvis/core/health/service.py - HYGIEIA's working parts
# =============================================================================
#
# Meals are logged in the life log, as before. The moment one is stored,
# this estimates what was in it, keeps the numbers beside it, and answers
# with where the day now stands: "protein 18 of 115 g, sodium at 52% of
# the limit".
#
# ESTIMATES, MADE CONSISTENT. One cheap model call per report turns
# "1 veg cheese Maggi" into items and nutrients - Indian dishes on IFCT
# 2017 values, packaged food on its label. Every item is remembered in
# health_foods, and the next estimate is handed those remembered values
# so the same food gets the same numbers. A mess meal described only as
# "lunch at the mess" is estimated from that day's menu, marked low
# confidence. Expect +/-20-30% on any one meal; the weeks are what count.
#
# WHAT HE CAN GET. Suggestions come only from food he can reach: tonight's
# mess menu, and foods he has actually had from campus or outside.
# =============================================================================

from __future__ import annotations

import re
import statistics
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jarvis.common.log import get_logger
from jarvis.core.db.repos.health import NUTRIENTS, HealthRepo, food_key
from jarvis.core.health.targets import Targets, compute
from jarvis.llm.layer import ToolSpec, user_message
from jarvis.llm.tiers import Tier

if TYPE_CHECKING:
    from jarvis.core.db.repos.money import MoneyRepo
    from jarvis.llm.layer import LLMLayer

log = get_logger("core.health")

ACTOR = "core.health"
_TOOL = "record_nutrition"
_MESS_CHUNK = 40
SLOTS = ("breakfast", "lunch", "snack", "dinner")
PROFILE_KEYS = {"weight_kg", "height_cm", "age", "goal", "training_days", "diet",
                "target_kcal", "target_protein_g", "target_fibre_g", "target_sodium_mg",
                "target_sugar_g", "target_junk_per_week"}

_SYSTEM = """You estimate the nutrition of meals the owner ate, for his own \
health tracking. He is in India, a college student eating mostly at his \
hostel mess, campus shops and outside.

For each meal, list the items he ate with ONE serving's values and how many \
servings he had:
- Indian dishes: values for a typical home/mess recipe, as in IFCT 2017 \
(Indian Food Composition Tables). A mess katori of dal is about 150 ml; a \
roti about 35 g of atta.
- Packaged food (Maggi, chips, biscuits, soft drinks): the label values.
- KNOWN FOODS are values already used for him. When an item is the same \
food, use those numbers exactly, so his records stay consistent.
- A meal with no detail ("lunch at the mess") means a typical plate from that \
day's MESS MENU: about 3 roti, a katori each of the dal and the main sabzi, \
a small portion of rice. Mark confidence low.
- sugar_g is ADDED sugar only (sweets, sugary drinks, sweetened tea), not \
the natural sugar in milk, fruit or vegetables.
- junk is true for deep-fried or ultra-processed items: instant noodles, \
chips, samosa, pakora, puri, bhature, sweets, sugary drinks, fast food.
- confidence: high when he gave quantities or it is a packaged item, medium \
for a normal description, low when you had to assume the plate.
- Never invent food he did not mention."""


class _Per(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kcal: float = Field(ge=0, le=3000)
    protein_g: float = Field(ge=0, le=200)
    carbs_g: float = Field(ge=0, le=500)
    fat_g: float = Field(ge=0, le=300)
    fibre_g: float = Field(ge=0, le=100)
    sodium_mg: float = Field(ge=0, le=10000)
    sugar_g: float = Field(ge=0, le=300)


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1, max_length=80)
    serving: str = Field(description="What one serving means, e.g. '1 pack (70 g)', "
                                     "'1 katori (150 ml)', '1 roti'.")
    servings: float = Field(1, gt=0, le=20)
    per_serving: _Per
    junk: bool = False


class _Meal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    meal: str = Field(description="The meal's handle, e.g. M1.")
    items: list[_Item]
    confidence: str = Field("medium", pattern="^(low|medium|high)$")


class _Result(BaseModel):
    meals: list[_Meal]


class _MenuFoods(BaseModel):
    foods: list[_Item]


def _fmt(n: float | None) -> str:
    return "-" if n is None else f"{n:,.0f}"


def _hours(minutes: float) -> str:
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h{m:02d}"


def menu_names(items: list[str]) -> list[str]:
    """'Omelette (limited) or Idli Sambhar' -> ['Omelette', 'Idli Sambhar']."""
    names: list[str] = []
    for item in items:
        for part in re.split(r"\s+or\s+", item):
            name = re.sub(r"\((?:limited|alternate[^)]*)\)", "", part, flags=re.I).strip()
            if name and name not in names:
                names.append(name)
    return names


def sleep_minutes(slept: str | None, woke_next: str | None) -> float | None:
    """Minutes asleep, from a bedtime (dated to the day that ended - 01:30
    belongs to the evening before) and the next morning's wake time."""
    if not slept or not woke_next:
        return None
    sh, sm = (int(x) for x in slept.split(":")[:2])
    wh, wm = (int(x) for x in woke_next.split(":")[:2])
    bed = sh * 60 + sm
    wake = wh * 60 + wm
    minutes = (1440 - bed + wake) if sh >= 12 else (wake - bed)
    return float(minutes) if 180 <= minutes <= 840 else None


class HealthService:
    def __init__(self, repo: HealthRepo, llm: "LLMLayer | None", tz: ZoneInfo, *,
                 money: "MoneyRepo | None" = None) -> None:
        self._repo = repo
        self._llm = llm
        self._tz = tz
        self._money = money

    @property
    def repo(self) -> HealthRepo:
        return self._repo

    def today(self, moment: datetime | None = None) -> date:
        return (moment or datetime.now(self._tz)).astimezone(self._tz).date()

    async def targets(self) -> Targets:
        return compute(await self._repo.profile())

    # -- estimating meals ---------------------------------------------------------

    async def on_meals(self, meal_ids: list[str], trace_id: str) -> str:
        """The life log's hook: estimate new meals, answer with the day's
        standing. Never raises - a failed estimate must not lose the meal."""
        try:
            meals = [m for m in [await self._repo.meal(i) for i in meal_ids]
                     if m is not None and m["status"] == "eaten"]
            if not meals:
                return ""
            await self.estimate(meals, trace_id)
            return await self.standing_line(meals[-1]["date"])
        except Exception:
            log.exception("nutrition estimate failed")
            return "(Nutrition could not be estimated just now - it will show as unknown.)"

    async def estimate(self, meals: list[dict[str, Any]], trace_id: str,
                       correction: str | None = None) -> None:
        if self._llm is None:
            return
        lines: list[str] = []
        names: list[str] = []
        for i, meal in enumerate(meals, start=1):
            day = date.fromisoformat(meal["date"])
            desc = meal["what"] or "(no detail given)"
            lines.append(f"M{i} | {meal['date']} {day.strftime('%A')} | {meal['slot']} | "
                         f"{desc} | place: {meal['place'] or 'unknown'}")
            names.append(desc)
            if self._at_mess(meal):
                menu = await self._repo.menu(day.weekday(), meal["slot"])
                if menu:
                    lines.append(f"   MESS MENU that {meal['slot']}: {', '.join(menu)}")
                    names += menu
        known = await self._repo.foods_like(names)
        if known:
            lines += ["", "KNOWN FOODS (name | one serving | kcal, protein, carbs, fat, "
                          "fibre g, sodium mg, added sugar g | junk):"]
            lines += [f"- {f['name']} | {f['serving']} | {f['kcal']:.0f}, {f['protein_g']:.1f}, "
                      f"{f['carbs_g']:.0f}, {f['fat_g']:.1f}, {f['fibre_g']:.1f}, "
                      f"{f['sodium_mg']:.0f}, {f['sugar_g']:.1f} | {bool(f['junk'])}"
                      for f in known]
        if correction:
            lines += ["", f"HIS CORRECTION to this meal: {correction}"]

        response = await self._llm.complete(
            Tier.UTILITY, _SYSTEM, [user_message("MEALS:\n" + "\n".join(lines))],
            actor=ACTOR, trace_id=trace_id,
            tools=[ToolSpec(name=_TOOL, description="Record the estimated nutrition.",
                            input_schema=_Result.model_json_schema())],
            tool_choice={"type": "tool", "name": _TOOL}, max_tokens=3000,
        )
        if response.stop_reason == "max_tokens" or not response.tool_calls:
            raise RuntimeError("the estimate did not come back whole")
        try:
            result = _Result.model_validate(response.tool_calls[0].args)
        except ValidationError as exc:
            raise RuntimeError(f"the estimate was malformed: {exc.errors()[0]['msg']}") from exc

        by_handle = {f"M{i}": m for i, m in enumerate(meals, start=1)}
        for estimate in result.meals:
            logged = by_handle.get(estimate.meal)
            if logged is None or not estimate.items:
                continue
            items = []
            for item in estimate.items:
                per = item.per_serving.model_dump()
                totals = {n: round(per[n] * item.servings, 1) for n in NUTRIENTS}
                items.append({"name": item.name, "serving": item.serving,
                              "servings": item.servings, "junk": item.junk, **totals})
                place = "mess" if self._at_mess(logged) else (logged["place"] or None)
                await self._repo.remember_food(
                    {"name": item.name, "serving": item.serving, "junk": item.junk,
                     "per_serving": per}, place, eaten_on=logged["date"])
            await self._repo.save_nutrition(logged["id"], items, estimate.confidence)

    @staticmethod
    def _at_mess(meal: dict[str, Any]) -> bool:
        text = f"{meal.get('place') or ''} {meal.get('what') or ''}".lower()
        return "mess" in text

    async def fix_meal(self, meal_id: str, correction: str, trace_id: str) -> str:
        meal = await self._repo.meal(meal_id)
        if meal is None:
            return "No such meal."
        await self.estimate([meal], trace_id, correction=correction)
        return await self.standing_line(meal["date"])

    async def learn_menu(self, trace_id: str) -> int:
        """Estimate every mess menu item not yet known, so suggestions can
        draw on tonight's menu. Run lazily; costs one call per 40 items."""
        if self._llm is None:
            return 0
        unknown: list[str] = []
        for row in await self._repo.menu_rows():
            for name in menu_names(row["items"].split(",")):
                if await self._repo.food(food_key(name)) is None and name not in unknown:
                    unknown.append(name)
        learned = 0
        for start in range(0, len(unknown), _MESS_CHUNK):
            chunk = unknown[start:start + _MESS_CHUNK]
            response = await self._llm.complete(
                Tier.UTILITY, _SYSTEM,
                [user_message("Give ONE mess serving's values for each of these hostel mess "
                              "items, as foods (servings 1). Keep each name EXACTLY as "
                              "written:\n" + "\n".join(chunk))],
                actor=ACTOR, trace_id=trace_id,
                tools=[ToolSpec(name="record_foods", description="Record the foods.",
                                input_schema=_MenuFoods.model_json_schema())],
                tool_choice={"type": "tool", "name": "record_foods"}, max_tokens=4000,
            )
            if not response.tool_calls:
                continue
            try:
                foods = _MenuFoods.model_validate(response.tool_calls[0].args).foods
            except ValidationError:
                continue
            for food in foods:
                await self._repo.remember_food(
                    {"name": food.name, "serving": food.serving, "junk": food.junk,
                     "per_serving": food.per_serving.model_dump()}, "mess", source="menu")
                learned += 1
        return learned

    # -- the day --------------------------------------------------------------------

    async def day_totals(self, day: str) -> dict[str, Any]:
        meals = await self._repo.meals_between(day, day)
        eaten = [m for m in meals if m["status"] == "eaten"]
        known = [m for m in eaten if m["kcal"] is not None]
        totals = {n: round(sum(float(m[n]) for m in known), 1) for n in NUTRIENTS}
        totals["junk"] = sum(int(m["junk"] or 0) for m in known)
        return {"date": day, "meals": meals, "logged": len(eaten),
                "unknown": len(eaten) - len(known), "totals": totals}

    async def standing_line(self, day: str) -> str:
        t = await self.targets()
        d = await self.day_totals(day)
        x = d["totals"]
        label = "Today" if day == self.today().isoformat() else f"On {day}"
        parts = [f"protein {_fmt(x['protein_g'])}"
                 + (f" of {_fmt(t.protein_g)} g" if t.protein_g else " g")]
        if t.kcal:
            parts.append(f"{_fmt(x['kcal'])} of {_fmt(t.kcal)} kcal")
        parts.append(f"sodium at {x['sodium_mg'] / t.sodium_mg:.0%} of the limit")
        if x["sugar_g"] >= 5:
            parts.append(f"added sugar {_fmt(x['sugar_g'])} of {_fmt(t.sugar_g)} g")
        parts.append(f"fibre {_fmt(x['fibre_g'])} of {_fmt(t.fibre_g)} g")
        return f"{label} so far: " + ", ".join(parts) + "."

    async def checkin_text(self, moment: datetime | None = None, trace_id: str = "") -> str:
        """The 5 pm check-in: what is left, and what he can actually eat."""
        now = (moment or datetime.now(self._tz)).astimezone(self._tz)
        today = now.date()
        t = await self.targets()
        d = await self.day_totals(today.isoformat())
        if not d["logged"]:
            return ("Five o'clock, sir, and no meals logged today. Tell me what you've eaten "
                    "so far and I'll tell you what's left.")
        if trace_id:
            try:
                await self.learn_menu(trace_id)
            except Exception:
                log.warning("menu foods could not be learned", exc_info=True)
        x = d["totals"]
        lines = ["Five o'clock check, sir. " + (await self.standing_line(today.isoformat()))]
        protein_left = (t.protein_g - x["protein_g"]) if t.protein_g else None
        fibre_left = t.fibre_g - x["fibre_g"]
        sodium_used = x["sodium_mg"] / t.sodium_mg

        todo: list[str] = []
        if protein_left and protein_left > 5:
            todo.append(f"about {protein_left:.0f} g of protein")
        if fibre_left > 5:
            todo.append(f"{fibre_left:.0f} g of fibre")
        if t.kcal and t.kcal - x["kcal"] > 300:
            todo.append(f"about {t.kcal - x['kcal']:,.0f} kcal")
        if todo:
            lines.append("Still to go: " + ", ".join(todo) + ".")

        dinner = await self._menu_foods(today.weekday(), "dinner")
        if protein_left and protein_left > 5 and dinner:
            best = sorted((f for f in dinner if f["protein_g"] >= 4 and not f["junk"]),
                          key=lambda f: -f["protein_g"])[:3]
            if best:
                lines.append("Tonight at the mess, the protein is in " + ", ".join(
                    f"{f['name']} (~{f['protein_g']:.0f} g a serving)" for f in best)
                    + " - a second helping of the first goes a long way.")
            elsewhere = [f for f in await self._repo.foods()
                         if f["place"] not in (None, "mess") and f["protein_g"] >= 8
                         and not f["junk"] and f["times_eaten"] >= 1][:2]
            if elsewhere:
                lines.append("Or from what you usually get: " + ", ".join(
                    f"{f['name']} (~{f['protein_g']:.0f} g)" for f in elsewhere) + ".")
        if sodium_used >= 0.7:
            salty = [f["name"] for f in dinner if f["sodium_mg"] >= 400][:3]
            lines.append(f"Sodium is already at {sodium_used:.0%}" + (
                f" - go easy on {', '.join(salty)} tonight." if salty
                else " - go easy on pickle, papad and anything fried tonight."))
        if x["sugar_g"] >= t.sugar_g * 0.8:
            lines.append(f"Added sugar is at {x['sugar_g']:.0f} of {t.sugar_g:.0f} g - "
                         f"skip the sweet tonight.")
        if d["unknown"]:
            lines.append(f"({d['unknown']} meal(s) today have no estimate yet.)")
        return "\n".join(lines)

    async def _menu_foods(self, weekday: int, slot: str) -> list[dict[str, Any]]:
        foods = []
        for name in menu_names(await self._repo.menu(weekday, slot)):
            food = await self._repo.food(food_key(name))
            if food is not None:
                foods.append(food)
        return foods

    # -- sleep ------------------------------------------------------------------------

    async def sleep(self, start: date, end: date) -> list[dict[str, Any]]:
        """Nights from start to end: bedtime on each date, wake the next day."""
        rows = {r["date"]: r for r in await self._repo.days(
            start.isoformat(), (end + timedelta(days=1)).isoformat())}
        nights = []
        day = start
        while day <= end:
            bed = rows.get(day.isoformat(), {}).get("sleep_time")
            woke = rows.get((day + timedelta(days=1)).isoformat(), {}).get("wake_time")
            nights.append({"night_of": day.isoformat(), "slept": bed, "woke": woke,
                           "minutes": sleep_minutes(bed, woke)})
            day += timedelta(days=1)
        return nights

    # -- reports ----------------------------------------------------------------------

    async def brief_lines(self, moment: datetime | None = None) -> list[str]:
        today = self.today(moment)
        y = (today - timedelta(days=1)).isoformat()
        t = await self.targets()
        d = await self.day_totals(y)
        lines: list[str] = []
        if d["logged"]:
            x = d["totals"]
            bits = [f"protein {_fmt(x['protein_g'])}"
                    + (f"/{_fmt(t.protein_g)} g" if t.protein_g else " g")]
            if x["sodium_mg"] > t.sodium_mg:
                bits.append(f"sodium over the limit ({x['sodium_mg'] / 1000:.1f} g)")
            if x["sugar_g"] > t.sugar_g:
                bits.append(f"added sugar over ({x['sugar_g']:.0f} g)")
            if x["junk"]:
                bits.append(f"{x['junk']} junk item(s)")
            lines.append("yesterday's food: " + ", ".join(bits))
        [night] = await self.sleep(today - timedelta(days=1), today - timedelta(days=1))
        if night["minutes"]:
            lines.append(f"slept {_hours(night['minutes'])} last night")
        if not t.protein_g:
            lines.append("protein target not set - tell HYGIEIA your weight")
        return lines

    async def week(self, end: date) -> dict[str, Any]:
        """The seven days ending on `end`, as numbers."""
        start = end - timedelta(days=6)
        t = await self.targets()
        meals = await self._repo.meals_between(start.isoformat(), end.isoformat())
        days: dict[str, dict[str, Any]] = {}
        for i in range(7):
            key = (start + timedelta(days=i)).isoformat()
            days[key] = {"date": key, "logged": 0, **{n: 0.0 for n in NUTRIENTS}, "junk": 0}
        junk_items: dict[str, int] = {}
        late = 0
        for m in meals:
            if m["status"] != "eaten":
                continue
            day = days[m["date"]]
            day["logged"] += 1
            if m["kcal"] is None:
                continue
            for n in NUTRIENTS:
                day[n] += float(m[n])
            day["junk"] += int(m["junk"] or 0)
            for item in m["items"]:
                if item.get("junk"):
                    junk_items[item["name"]] = junk_items.get(item["name"], 0) + 1
            stamp = datetime.fromisoformat(m["created_ts"]).astimezone(self._tz)
            if m["slot"] in ("snack", "dinner") and (stamp.hour >= 23 or stamp.hour < 4):
                late += 1
        logged = [d for d in days.values() if d["logged"]]
        nights = [n["minutes"] for n in await self.sleep(start - timedelta(days=1),
                                                         end - timedelta(days=1))
                  if n["minutes"]]
        food_spend = late_spend = 0
        if self._money is not None:
            for txn in await self._money.between(start.isoformat(), end.isoformat()):
                if txn["status"] == "success" and txn["category"] == "food":
                    food_spend += txn["amount_paise"]
                    hour = int(txn["occurred_at"][11:13])
                    if hour >= 22 or hour < 4:
                        late_spend += txn["amount_paise"]
        return {
            "start": start.isoformat(), "end": end.isoformat(), "targets": t.as_dict(),
            "days": list(days.values()), "days_logged": len(logged),
            "protein_avg": statistics.mean(d["protein_g"] for d in logged) if logged else 0,
            "protein_days": sum(1 for d in logged if t.protein_g and d["protein_g"]
                                >= t.protein_g * 0.95),
            "sodium_over": sum(1 for d in logged if d["sodium_mg"] > t.sodium_mg),
            "sugar_over": sum(1 for d in logged if d["sugar_g"] > t.sugar_g),
            "fibre_avg": statistics.mean(d["fibre_g"] for d in logged) if logged else 0,
            "junk": sum(d["junk"] for d in logged),
            "junk_items": sorted(junk_items.items(), key=lambda kv: -kv[1])[:4],
            "late_meals": late,
            "sleep_avg": statistics.mean(nights) if nights else None,
            "sleep_short": sum(1 for n in nights if n < 420),
            "sleep_nights": len(nights),
            "food_spend_paise": food_spend, "late_food_paise": late_spend,
        }

    async def weekly_review(self, moment: datetime | None = None) -> str:
        today = self.today(moment)
        w = await self.week(today - timedelta(days=1))
        t = w["targets"]
        span = (f"{date.fromisoformat(w['start']).strftime('%d %b')} - "
                f"{date.fromisoformat(w['end']).strftime('%d %b')}")
        if not w["days_logged"]:
            return (f"Health this week ({span}): no meals logged, sir - nothing to report. "
                    f"Even rough logging gives me something to work with.")
        lines = [f"Health this week ({span}), from {w['days_logged']} logged day(s):", ""]
        if t["protein_g"]:
            lines.append(f"- Protein: averaged {w['protein_avg']:.0f} of {t['protein_g']:.0f} g; "
                         f"target hit on {w['protein_days']} day(s).")
        else:
            lines.append(f"- Protein: averaged {w['protein_avg']:.0f} g (no target until "
                         f"your weight is set).")
        lines.append(f"- Sodium over the limit on {w['sodium_over']} day(s); added sugar over "
                     f"on {w['sugar_over']}.")
        lines.append(f"- Fibre: averaged {w['fibre_avg']:.0f} of {t['fibre_g']:.0f} g.")
        junk = f"- Junk: {w['junk']} item(s) against a limit of {t['junk_per_week']}"
        if w["junk_items"]:
            junk += " - mostly " + ", ".join(f"{name} ({n}x)" for name, n in w["junk_items"])
        lines.append(junk + ".")
        if w["late_meals"]:
            lines.append(f"- {w['late_meals']} meal(s) logged after 11 pm.")
        if w["food_spend_paise"]:
            spend = f"- Food spending: Rs {w['food_spend_paise'] // 100:,}"
            if w["late_food_paise"]:
                spend += f", of which Rs {w['late_food_paise'] // 100:,} after 10 pm"
            lines.append(spend + ".")
        if w["sleep_avg"]:
            lines.append(f"- Sleep: averaged {_hours(w['sleep_avg'])} over {w['sleep_nights']} "
                         f"night(s); {w['sleep_short']} under 7 hours.")
        focus = self._focus(w)
        if focus:
            lines += ["", f"One thing for next week: {focus}"]
        return "\n".join(lines)

    @staticmethod
    def _focus(w: dict[str, Any]) -> str:
        t = w["targets"]
        if t["protein_g"] and w["protein_days"] < max(1, w["days_logged"]) // 2:
            gap = t["protein_g"] - w["protein_avg"]
            return (f"protein - you're about {gap:.0f} g short a day. Eggs at breakfast and a "
                    f"second katori of dal or paneer at dinner covers most of it.")
        if w["junk"] > t["junk_per_week"]:
            return "fewer fried and instant meals - pick the one you have most and swap it."
        if w["sodium_over"] >= 3:
            return "salt - pickle, papad and instant noodles are where most of it comes from."
        if w["sleep_avg"] and w["sleep_avg"] < 420:
            return "sleep - aim for 7 hours; late nights are where the junk meals come from."
        return ""

    # -- the profile ------------------------------------------------------------------

    async def set_profile(self, key: str, value: str | None) -> str:
        if key not in PROFILE_KEYS:
            return f"Unknown setting {key!r}. Settable: {', '.join(sorted(PROFILE_KEYS))}."
        if key == "goal" and value not in (None, "gain_muscle", "maintain", "lose_fat"):
            return "Goal is gain_muscle, maintain or lose_fat."
        if key not in ("goal", "diet") and value not in (None, ""):
            try:
                if float(value) <= 0:
                    raise ValueError
            except ValueError:
                return f"{key} must be a positive number."
        if key == "target_kcal" and value and float(value) < 1600:
            return "I won't set a calorie target below 1,600 - that's under what you need."
        await self._repo.set_profile(key, value)
        t = await self.targets()
        return (f"Set {key}. Targets now: protein {_fmt(t.protein_g)} g, "
                f"calories {_fmt(t.kcal)}, fibre {_fmt(t.fibre_g)} g; limits: sodium "
                f"{_fmt(t.sodium_mg)} mg, added sugar {_fmt(t.sugar_g)} g, junk "
                f"{t.junk_per_week}/week.")

    async def today_text(self, day: str | None = None) -> str:
        day = day or self.today().isoformat()
        d = await self.day_totals(day)
        if not d["logged"]:
            return f"No meals logged for {day}."
        lines = [await self.standing_line(day)]
        for m in d["meals"]:
            if m["status"] != "eaten":
                lines.append(f"[{m['id'][-6:]}] {m['slot']}: skipped")
                continue
            if m["kcal"] is None:
                lines.append(f"[{m['id'][-6:]}] {m['slot']}: {m['what'] or '?'} (no estimate)")
                continue
            items = ", ".join(f"{i['name']} x{i['servings']:g}" for i in m["items"])
            lines.append(f"[{m['id'][-6:]}] {m['slot']}: {items} - {m['kcal']:.0f} kcal, "
                         f"{m['protein_g']:.0f} g protein, {m['sodium_mg']:.0f} mg sodium "
                         f"({m['confidence']} confidence)")
        return "\n".join(lines)

    async def find_meal(self, short: str, day: str | None = None) -> dict[str, Any] | None:
        day = day or self.today().isoformat()
        start = (date.fromisoformat(day) - timedelta(days=7)).isoformat()
        rows = [m for m in await self._repo.meals_between(start, day)
                if m["id"].upper().endswith(short.upper())]
        return rows[0] if len(rows) == 1 else None
