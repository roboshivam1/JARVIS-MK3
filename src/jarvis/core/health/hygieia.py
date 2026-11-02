# =============================================================================
# src/jarvis/core/health/hygieia.py - HYGIEIA, keeper of the owner's health
# =============================================================================
#
# Health data gets the same wall as money: JARVIS does not read it, he asks
# HYGIEIA (ask_hygieia) and relays the answer. Her tools read and correct
# the nutrition records, the food memory, the mess menu and the profile -
# nothing else.
#
# Meals themselves are still logged by JARVIS into the life log; HYGIEIA
# hears about each one through the life log's meal hook.
# =============================================================================

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field

from jarvis.agentloop.loop import run_agent_loop
from jarvis.agentloop.policies import HYGIEIA, create_guard
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.core.health.service import SLOTS, HealthService
from jarvis.llm.layer import LLMLayer, user_message
from jarvis.llm.tiers import Tier

ACTOR = HYGIEIA

HYGIEIA_PERSONA_V1 = """\
You are HYGIEIA, the health keeper inside the owner's assistant, JARVIS.
JARVIS passes you one request at a time - a question about his eating or
sleep, or an instruction - and passes your answer back to him.

The owner: an 18-year-old college student, lifting six days a week, aiming
to build some muscle and keep a clean, healthy lifestyle. He eats non-veg,
mostly at his hostel mess (which is vegetarian apart from eggs at
breakfast), plus campus shops and outside food.

What you have: an estimate of the nutrients in every meal he logged, the
foods he eats and where he gets them, the mess menu, his sleep times, and
daily targets (protein, calories, fibre) and limits (sodium, added sugar,
junk food per week) worked out from public guidelines and his profile.

How you work:
- Use your tools for every number. Never estimate in your head.
- He corrects a meal ("it was half a plate", "two eggs, not three") with
  health_fix_meal. A new fact about him (weight, goal) is health_set.
- He can teach you a food with health_teach_food - his numbers then win
  over any estimate.

How you answer:
- Short, plain, factual. Lead with the answer.
- Estimates are estimates: say "about".
- Suggest food he can actually get: the mess menu, or what he has had
  before. Never suggest supplements, medication or diets.
- You are not a doctor. Anything that sounds medical - pain, illness,
  sudden weight change - gets one line suggesting he see one.
- Never set or encourage a calorie target below what he needs, and never
  frame food with guilt. A bad meal is one data point."""


class _Day(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$",
                             description="YYYY-MM-DD; empty for today.")


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Fix(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meal: str = Field(description="The meal's id shown in brackets by health_day.")
    correction: str = Field(min_length=3, description="His correction, in his words.")


class _Set(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str = Field(description="weight_kg, height_cm, age, goal (gain_muscle | maintain | "
                                 "lose_fat), training_days, or target_<name> to override a "
                                 "target or limit (kcal, protein_g, fibre_g, sodium_mg, "
                                 "sugar_g, junk_per_week).")
    value: str | None = Field(None, description="Empty to clear an override.")


class _Teach(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2)
    place: str = Field(description="mess, campus, outside, room or home")
    serving: str = Field(description="What one serving is, e.g. '1 roll'.")
    kcal: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    fibre_g: float = Field(0, ge=0)
    sodium_mg: float = Field(0, ge=0)
    sugar_g: float = Field(0, ge=0)
    junk: bool = False


class _Foods(BaseModel):
    model_config = ConfigDict(extra="forbid")
    place: str | None = Field(None, description="mess, campus, outside, room, home; empty "
                                                "for all.")


class _Menu(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$",
                             description="YYYY-MM-DD; empty for today.")


def build_toolset(health: HealthService, trace_id: str = "") -> Toolset:
    tools = Toolset(guard=create_guard(), actor=ACTOR)
    repo = health.repo

    async def day(args: _Day) -> str:
        return await health.today_text(args.date)

    async def week(_: _NoArgs) -> str:
        return await health.weekly_review()

    async def targets(_: _NoArgs) -> str:
        t = await health.targets()
        profile = await repo.profile()
        missing = "" if t.protein_g else " Protein and calories need his weight (weight_kg)."
        return (f"Profile: {profile}. Targets: protein {t.protein_g} g, calories {t.kcal}, "
                f"fibre {t.fibre_g} g. Limits: sodium {t.sodium_mg} mg, added sugar "
                f"{t.sugar_g} g, junk {t.junk_per_week}/week.{missing}")

    async def fix(args: _Fix) -> str:
        meal = await health.find_meal(args.meal)
        if meal is None:
            return f"No single meal matches {args.meal} in the last week."
        return await health.fix_meal(meal["id"], args.correction, trace_id=trace_id)

    async def set_(args: _Set) -> str:
        return await health.set_profile(args.key, args.value)

    async def teach(args: _Teach) -> str:
        data = args.model_dump()
        await repo.remember_food({"name": args.name, "serving": args.serving,
                                  "junk": args.junk, "per_serving": data},
                                 args.place, source="owner")
        return f"Noted {args.name} ({args.serving}) from the {args.place}."

    async def foods(args: _Foods) -> str:
        rows = await repo.foods(args.place, limit=40)
        if not rows:
            return "No foods known there yet."
        return "\n".join(
            f"{f['name']} [{f['place'] or '?'}] {f['serving']}: {f['kcal']:.0f} kcal, "
            f"{f['protein_g']:.0f} g protein, {f['sodium_mg']:.0f} mg sodium"
            + (" (junk)" if f["junk"] else "") + f" - had {f['times_eaten']}x"
            for f in rows)

    async def menu(args: _Menu) -> str:
        when = date.fromisoformat(args.date) if args.date else health.today()
        lines = [f"Mess menu, {when.strftime('%A %d %b')}:"]
        for slot in SLOTS:
            items = await repo.menu(when.weekday(), slot)
            lines.append(f"{slot}: {', '.join(items) or '-'}")
        return "\n".join(lines)

    async def sleep(_: _NoArgs) -> str:
        today = health.today()
        nights = await health.sleep(today - timedelta(days=7), today - timedelta(days=1))
        lines = [f"night of {n['night_of']}: " + (
            f"{n['slept']} to {n['woke']}, {n['minutes'] / 60:.1f} h" if n["minutes"]
            else "not logged") for n in nights]
        return "\n".join(lines)

    for name, description, model, handler in (
        ("health_day", "One day's meals with nutrients, and the day's standing.", _Day, day),
        ("health_week", "The last seven days against his targets and limits.", _NoArgs, week),
        ("health_targets", "His profile, targets and limits.", _NoArgs, targets),
        ("health_fix_meal", "Re-estimate a meal with his correction.", _Fix, fix),
        ("health_set", "Change his profile or override a target or limit.", _Set, set_),
        ("health_teach_food", "Record a food's values as he gives them.", _Teach, teach),
        ("health_foods", "Foods he eats, by where he gets them.", _Foods, foods),
        ("health_menu", "The mess menu for a day.", _Menu, menu),
        ("health_sleep", "The last week's sleep.", _NoArgs, sleep),
    ):
        tools.register(InlineTool(name=name, description=description, args_model=model,
                                  handler=handler))
    return tools


async def ask_hygieia(llm: LLMLayer, health: HealthService, request: str, *,
                      trace_id: str) -> str:
    tools = build_toolset(health, trace_id)
    result = await run_agent_loop(
        llm, Tier.UTILITY, HYGIEIA_PERSONA_V1, [user_message(request)], tools,
        actor=ACTOR, trace_id=trace_id, max_iterations=6, max_tokens=1200,
    )
    return result.text.strip() or "HYGIEIA had nothing to say to that."

