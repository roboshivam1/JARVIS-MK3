# =============================================================================
# tests/integration/test_health.py - HYGIEIA
# =============================================================================
#
# The promises: every meal logged in the life log gets nutrition, and the
# reply says where the day stands; the same food keeps the same numbers;
# mess meals are estimated from that day's menu; his own numbers beat any
# estimate; the 5 pm check-in suggests food he can actually get; targets
# come from his profile and never go below a sensible floor; sleep and
# food spending show up in the week.
# =============================================================================

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import httpx
import pytest

from jarvis.agentloop.policies import HYGIEIA, ORCHESTRATOR, create_guard
from jarvis.common.ids import new_ulid
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.health import HealthRepo
from jarvis.core.db.repos.life import LifeRepo
from jarvis.core.db.repos.money import MoneyRepo
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.health.hygieia import build_toolset
from jarvis.core.health.service import HealthService, sleep_minutes
from jarvis.core.health.targets import compute
from jarvis.core.life.service import LifeService
from jarvis.core.money.service import MoneyService
from jarvis.core.observability.traces import TracesRepo
from jarvis.llm.layer import ToolCall
from tests.integration.test_money import alert

JAIPUR = ZoneInfo("Asia/Kolkata")
AUTH = {"Authorization": "Bearer test-token"}
MONDAY = datetime(2026, 11, 2, 13, 30, tzinfo=JAIPUR)

MAGGI = {"name": "Veg cheese Maggi", "serving": "1 plate (1 pack + cheese)", "servings": 1,
         "per_serving": {"kcal": 420, "protein_g": 11, "carbs_g": 52, "fat_g": 18,
                         "fibre_g": 3, "sodium_mg": 1150, "sugar_g": 2}, "junk": True}


def item(name: str, protein: float, sodium: float = 300, kcal: float = 200,
         junk: bool = False, servings: float = 1) -> dict[str, Any]:
    return {"name": name, "serving": "1 katori", "servings": servings, "junk": junk,
            "per_serving": {"kcal": kcal, "protein_g": protein, "carbs_g": 20, "fat_g": 6,
                            "fibre_g": 4, "sodium_mg": sodium, "sugar_g": 0}}


class ScriptedLLM:
    """Answers each tool by name from its own queue."""

    def __init__(self) -> None:
        self.queues: dict[str, list[dict[str, Any]]] = {}
        self.prompts: dict[str, list[str]] = {}

    def on(self, tool: str, *replies: dict[str, Any]) -> None:
        self.queues.setdefault(tool, []).extend(replies)

    async def complete(self, tier: Any, system: str, messages: list[dict[str, Any]],
                       **kwargs: Any) -> Any:
        tool = kwargs["tool_choice"]["name"]
        self.prompts.setdefault(tool, []).append(messages[0]["content"])
        return SimpleNamespace(stop_reason="tool_use", tool_calls=[
            ToolCall(id="t1", name=tool, args=self.queues[tool].pop(0))])


@pytest.fixture
def world(db: Database) -> SimpleNamespace:
    llm = ScriptedLLM()
    money = MoneyService(MoneyRepo(db), JAIPUR)
    health = HealthService(HealthRepo(db), llm, JAIPUR, money=money.repo)  # type: ignore[arg-type]
    life = LifeService(LifeRepo(db), llm, JAIPUR)  # type: ignore[arg-type]
    life.set_meal_hook(health.on_meals)
    return SimpleNamespace(db=db, llm=llm, health=health, life=life, money=money,
                           repo=health.repo)


async def eat(world: SimpleNamespace, meals: list[dict[str, Any]],
              estimates: list[dict[str, Any]], at: datetime = MONDAY) -> str:
    world.llm.on("record_report", {"meals": meals})
    world.llm.on("record_nutrition", {"meals": estimates})
    outcome = await world.life.report("...", trace_id=new_ulid(), moment=at)
    return str(outcome.summary())


def test_targets_follow_the_profile_and_guidelines() -> None:
    base = {"age": "18", "sex": "male", "height_cm": "178", "goal": "gain_muscle",
            "training_days": "6"}
    no_weight = compute(base)
    assert (no_weight.protein_g, no_weight.kcal) == (None, None)
    assert (no_weight.fibre_g, no_weight.sodium_mg, no_weight.sugar_g) == (30, 2000, 50)
    t = compute({**base, "weight_kg": "65"})
    assert t.protein_g == 104 and t.kcal == 2950
    assert compute({**base, "weight_kg": "65", "target_protein_g": "120"}).protein_g == 120


async def test_a_logged_meal_gets_nutrition_and_the_day_standing(world: SimpleNamespace) -> None:
    await world.health.set_profile("weight_kg", "65")
    summary = await eat(world,
                        [{"date": "2026-11-02", "slot": "lunch", "what": "1 veg cheese Maggi",
                          "place": "campus"}],
                        [{"meal": "M1", "items": [MAGGI], "confidence": "high"}])
    assert summary.startswith("Logged: lunch eaten.")
    assert "so far: protein 11 of 104 g" in summary
    assert "sodium at 57% of the limit" in summary
    day = await world.health.day_totals("2026-11-02")
    assert day["totals"]["junk"] == 1 and day["meals"][0]["confidence"] == "high"
    food = await world.repo.food("veg cheese maggi")
    assert food["place"] == "campus" and food["times_eaten"] == 1

    # The same food next time: its remembered values are handed to the model.
    await eat(world, [{"date": "2026-11-02", "slot": "snack", "what": "maggi again"}],
              [{"meal": "M1", "items": [MAGGI], "confidence": "high"}])
    assert "Veg cheese Maggi | 1 plate (1 pack + cheese) | 420, 11.0" in \
        world.llm.prompts["record_nutrition"][-1]

    # Undoing the report takes the nutrition with the meal.
    await world.life.undo_last()
    assert len((await world.health.day_totals("2026-11-02"))["meals"]) == 1


async def test_a_mess_meal_is_estimated_from_that_days_menu(world: SimpleNamespace) -> None:
    await eat(world, [{"date": "2026-11-02", "slot": "lunch", "place": "mess"}],
              [{"meal": "M1", "items": [item("Matar Paneer", 9), item("Dal Arhar", 7)],
                "confidence": "low"}])
    prompt = world.llm.prompts["record_nutrition"][-1]
    assert "M1 | 2026-11-02 Monday | lunch | (no detail given) | place: mess" in prompt
    assert "MESS MENU that lunch: Matar Paneer" in prompt and "Matar Paneer, Dal Arhar" in prompt
    assert (await world.repo.food("dal arhar"))["place"] == "mess"


async def test_corrections_and_his_own_numbers_win(world: SimpleNamespace) -> None:
    await eat(world, [{"date": "2026-11-02", "slot": "breakfast", "what": "3 eggs"}],
              [{"meal": "M1", "items": [item("Boiled egg", 6, servings=3)],
                "confidence": "high"}])
    [meal] = (await world.health.day_totals("2026-11-02"))["meals"]
    world.llm.on("record_nutrition", {"meals": [{"meal": "M1", "items": [
        item("Boiled egg", 6, servings=2)], "confidence": "high"}]})
    await world.health.fix_meal(meal["id"], "it was 2 eggs, not 3", new_ulid())
    assert "HIS CORRECTION to this meal: it was 2 eggs, not 3" in \
        world.llm.prompts["record_nutrition"][-1]
    assert (await world.health.day_totals("2026-11-02"))["totals"]["protein_g"] == 12

    tools = build_toolset(world.health)
    await tools.execute("health_teach_food", {
        "name": "Campus P paneer roll", "place": "campus", "serving": "1 roll",
        "kcal": 380, "protein_g": 16, "carbs_g": 40, "fat_g": 16})
    await world.repo.remember_food({"name": "Campus P paneer roll", "serving": "x",
                                    "per_serving": {"kcal": 999, "protein_g": 1}},
                                   "campus", eaten_on="2026-11-02")
    roll = await world.repo.food("campus p paneer roll")
    assert (roll["kcal"], roll["source"], roll["times_eaten"]) == (380, "owner", 1)


async def test_the_five_pm_check_in(world: SimpleNamespace) -> None:
    five = datetime(2026, 11, 2, 17, 0, tzinfo=JAIPUR)
    assert "no meals logged today" in await world.health.checkin_text(five)

    await world.health.set_profile("weight_kg", "65")
    await eat(world, [{"date": "2026-11-02", "slot": "lunch", "what": "maggi",
                       "place": "campus"}],
              [{"meal": "M1", "items": [MAGGI, MAGGI], "confidence": "high"}])
    world.llm.on("record_foods", {"foods": [
        item("Fried Aloo Pyaz Sabji", 3, sodium=500, junk=True), item("Dal Moong Tadka", 8),
        item("Plain Rice", 3), item("Mixed Pickle", 0, sodium=900), item("Ice Cream", 2),
        item("Plain/Butter Roti", 3, sodium=150), item("Salad", 1), item("Onion", 0),
        item("Roasted Papad", 1, sodium=450)]})
    text = await world.health.checkin_text(five, trace_id=new_ulid())
    assert text.startswith("Five o'clock check, sir. Today so far: protein 22 of 104 g")
    assert "Still to go: about 82 g of protein" in text
    assert "Tonight at the mess, the protein is in Dal Moong Tadka (~8 g a serving)" in text
    assert "Sodium is already at 115% - go easy on Fried Aloo Pyaz Sabji, Mixed Pickle, Roasted Papad tonight." in text
    assert "Fried Aloo Pyaz Sabji" not in text.split("protein is in")[1].split("\n")[0]


def test_sleep_counts_across_midnight() -> None:
    assert sleep_minutes("01:30", "08:00") == 390
    assert sleep_minutes("23:00", "07:00") == 480
    assert sleep_minutes("23:00", None) is None
    assert sleep_minutes("10:00", "10:30") is None                 # a nap, not a night


async def test_the_week_with_sleep_and_spending(world: SimpleNamespace) -> None:
    await world.health.set_profile("weight_kg", "65")
    life = world.life._repo
    await life.set_day("2026-10-26", "sleep", "01:30", None)
    await life.set_day("2026-10-27", "wake", "07:30", None)
    for day in ("2026-10-27", "2026-10-28"):
        await eat(world, [{"date": day, "slot": "dinner", "what": "maggi"}],
                  [{"meal": "M1", "items": [MAGGI, MAGGI], "confidence": "high"}])
    await world.money.ingest_text(alert("Swiggy", "349.00", "1", "28-10-2026 23:40:00"))
    await world.money.categorise_payee("Swiggy", "food")
    review = await world.health.weekly_review(datetime(2026, 11, 1, 10, 30, tzinfo=JAIPUR))
    assert review.startswith("Health this week (25 Oct - 31 Oct), from 2 logged day(s):")
    assert "- Protein: averaged 22 of 104 g; target hit on 0 day(s)." in review
    assert "- Sodium over the limit on 2 day(s)" in review
    assert "- Junk: 4 item(s) against a limit of 4 - mostly Veg cheese Maggi (4x)." in review
    assert "- Food spending: Rs 349, of which Rs 349 after 10 pm." in review
    assert "- Sleep: averaged 6h00 over 1 night(s); 1 under 7 hours." in review
    assert "One thing for next week: protein - you're about 82 g short a day." in review


async def test_profile_guards(world: SimpleNamespace) -> None:
    assert "below 1,600" in await world.health.set_profile("target_kcal", "1200")
    assert "Goal is" in await world.health.set_profile("goal", "bulk")
    assert "Unknown setting" in await world.health.set_profile("password", "x")
    assert (await world.health.set_profile("weight_kg", "68")).startswith("Set weight_kg.")


async def test_hygieia_touches_health_and_nothing_else(world: SimpleNamespace) -> None:
    names = {spec.name for spec in build_toolset(world.health).specs()}
    assert names and all(n.startswith("health_") for n in names)
    guard = create_guard()
    assert guard.check(HYGIEIA, "health_day", {}).verdict.value == "allow"
    assert guard.check(HYGIEIA, "money_summary", {}).verdict.value != "allow"
    assert guard.check(ORCHESTRATOR, "ask_hygieia", {}).verdict.value == "allow"
    assert guard.check(ORCHESTRATOR, "health_day", {}).verdict.value != "allow"


@pytest.fixture
async def client(settings: CoreSettings, world: SimpleNamespace) -> AsyncIterator[httpx.AsyncClient]:
    deps = GatewayDeps(settings=settings, db=world.db, traces=TracesRepo(world.db),
                       started_monotonic=time.monotonic(), health=world.health)
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_the_page_and_its_api(client: httpx.AsyncClient) -> None:
    page = await client.get("/app/health")
    assert page.status_code == 200 and "Targets and limits" in page.text
    assert (await client.get("/health-data/day")).status_code == 401
    assert (await client.get("/health")).json() == {"status": "alive"}   # liveness untouched

    set_weight = await client.post("/health-data/profile", headers=AUTH,
                                   json={"key": "weight_kg", "value": "65"})
    assert set_weight.status_code == 200
    assert (await client.post("/health-data/profile", headers=AUTH,
                              json={"key": "goal", "value": "bulk"})).status_code == 422
    day = (await client.get("/health-data/day", headers=AUTH)).json()
    assert day["targets"]["protein_g"] == 104 and day["logged"] == 0

    menu = await client.put("/health-data/menu", headers=AUTH,
                            json={"weekday": 0, "slot": "dinner", "items": "Egg curry, Rice"})
    assert menu.status_code == 200
    rows = (await client.get("/health-data/menu", headers=AUTH)).json()["menu"]
    assert any(r["weekday"] == 0 and r["slot"] == "dinner" and r["items"] == "Egg curry, Rice"
               for r in rows)
    taught = await client.post("/health-data/foods", headers=AUTH, json={
        "name": "Boiled egg", "place": "outside", "serving": "1 egg", "kcal": 78,
        "protein_g": 6.3})
    assert taught.status_code == 200
    [egg] = (await client.get("/health-data/foods?place=outside", headers=AUTH)).json()["foods"]
    assert egg["source"] == "owner"
    assert (await client.get("/health-data/trend?days=14", headers=AUTH)).status_code == 200
    assert (await client.get("/health-data/summary", headers=AUTH)).json()["logged"] == 0
