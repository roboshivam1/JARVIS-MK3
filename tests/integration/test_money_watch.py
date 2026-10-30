# =============================================================================
# tests/integration/test_money_watch.py - PLUTUS keeping watch
# =============================================================================
#
# The promises: a failed payment is flagged once and stays until cleared;
# "unusual" is judged against his own history and never fires on small
# amounts; subscriptions are found from real cycles, warned before they
# renew, and their price changes and misses are pointed out once; the
# Sunday review and the brief read it all back.
# =============================================================================

from __future__ import annotations

import time
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import httpx
import pytest

from jarvis.common.ids import new_ulid
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.money import MoneyRepo
from jarvis.core.db.repos.money_watch import MoneyWatchRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.sessions import SessionsRepo
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.money.plutus import build_toolset
from jarvis.core.money.service import MoneyService
from jarvis.core.money.watchdog import URGENT, Watchdog, add_cycle
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.queue.registry import JobContext, JobTypeRegistry
from jarvis.jobs.money import register_money_jobs
from tests.integration.test_money import alert

JAIPUR = ZoneInfo("Asia/Kolkata")
AUTH = {"Authorization": "Bearer test-token"}


def at(day: str, hour: int = 9) -> datetime:
    return datetime.fromisoformat(f"{day}T{hour:02d}:00:00").replace(tzinfo=JAIPUR)


@pytest.fixture
def world(db: Database) -> SimpleNamespace:
    said: list[tuple[str, int]] = []

    async def notify(text: str, priority: int) -> None:
        said.append((text, priority))

    ledger = MoneyRepo(db)
    dog = Watchdog(ledger, MoneyWatchRepo(db), JAIPUR, notify=notify)
    money = MoneyService(ledger, JAIPUR, watchdog=dog)
    n = iter(range(1, 10_000))

    async def pay(payee: str, rupees: str, day: str, status: str = "Success") -> Any:
        d = date.fromisoformat(day)
        return await money.ingest_text(alert(payee, rupees, str(next(n)),
                                             d.strftime("%d-%m-%Y") + " 10:30:00", status))

    return SimpleNamespace(db=db, money=money, dog=dog, said=said, pay=pay)


def test_cycles_land_on_the_right_day() -> None:
    assert add_cycle(date(2026, 1, 31), "monthly") == date(2026, 2, 28)
    assert add_cycle(date(2026, 12, 15), "monthly") == date(2027, 1, 15)
    assert add_cycle(date(2028, 2, 29), "yearly") == date(2029, 2, 28)


async def test_a_failed_payment_waits_until_cleared(world: SimpleNamespace) -> None:
    txn = await world.pay("Zomato", "250.00", "2026-10-29", status="Failed")
    await world.money.ingest_text(alert("Zomato", "250.00", "1"))   # same ref: ignored
    [(text, priority)] = world.said
    assert "Rs 250 to Zomato" in text and "marked failed" in text and priority == 3
    [item] = await world.dog.repo.open_items()
    assert item["txn_id"] == txn["id"]

    tools = build_toolset(world.money)
    listed, _ = await tools.execute("money_watchlist", {})
    short = listed[1:7]
    cleared, _ = await tools.execute("money_clear_watch", {"item": short,
                                                           "note": "it came back"})
    assert cleared.startswith("Cleared: failed payment of Rs 250 to Zomato")
    assert await world.dog.repo.open_items() == []


async def test_unusual_is_judged_against_his_own_history(world: SimpleNamespace) -> None:
    for day in ("2026-10-01", "2026-10-05", "2026-10-09"):
        await world.pay("Robu.in", "150.00", day)
    await world.pay("Robu.in", "400.00", "2026-10-10")          # odd, but under the floor
    assert world.said == []
    await world.pay("Robu.in", "1,600.00", "2026-10-12")
    [(text, priority)] = world.said
    assert priority == URGENT
    assert "Rs 1,600 to Robu.in, where you usually pay about Rs 150" in text
    assert "SMS BLOCK 9572 to 8879365472" in text

    for i in range(10):
        await world.pay(f"Shop {i}", "100.00", "2026-10-13")
    await world.pay("Unknown Traders", "2,000.00", "2026-10-14")
    assert "a payee you have never paid before" in world.said[-1][0]


async def test_subscriptions_are_found_warned_and_watched(world: SimpleNamespace) -> None:
    for day in ("2026-07-03", "2026-08-03", "2026-09-03"):
        await world.pay("Spotify", "119.00", day)
    for day in ("2026-08-01", "2026-08-02", "2026-08-29"):      # irregular: not a cycle
        await world.pay("Campus P", "65.00", day)

    message = await world.dog.daily(at("2026-09-10"))
    assert "New regular payment(s) spotted: Spotify (Rs 119 monthly)" in message
    [sub] = await world.dog.repo.subscriptions()
    assert sub["next_due"] == "2026-10-03"

    assert "Spotify renews in" not in message
    warn = await world.dog.daily(at("2026-10-01"))
    assert "Spotify renews Sat 03 Oct - about Rs 119." in warn
    assert await world.dog.daily(at("2026-10-02")) == ""           # once per renewal

    await world.pay("Spotify", "139.00", "2026-10-03")
    assert world.said[-1] == ("Spotify charged Rs 139 this time, sir - it used to be Rs 119.", 4)
    [sub] = await world.dog.repo.subscriptions()
    assert (sub["next_due"], sub["typical_paise"]) == ("2026-11-03", 13900)

    missed = await world.dog.daily(at("2026-11-10"))
    assert "Spotify was due on 03 Nov and has not been paid" in missed
    assert "has not been paid" not in await world.dog.daily(at("2026-11-11"))

    tools = build_toolset(world.money)
    out, _ = await tools.execute("money_set_subscription", {"payee": "Spotify", "cadence": "none"})
    assert out == "Dismissed."
    await world.dog.daily(at("2026-11-12"))
    assert await world.dog.repo.subscriptions() == []             # stays dismissed
    out, _ = await tools.execute("money_set_subscription", {"payee": "campus p",
                                                            "cadence": "monthly"})
    assert out == "Marked as a monthly subscription; next expected 29 Sep."


async def test_the_sunday_review_and_the_brief(world: SimpleNamespace) -> None:
    await world.money.categorise_payee("Zomato", "food")
    await world.pay("Zomato", "200.00", "2026-10-19")
    await world.pay("Zomato", "300.00", "2026-10-21")
    await world.pay("Robu.in", "1,249.00", "2026-10-22")
    await world.pay("Uber", "150.00", "2026-10-27", status="Failed")
    await world.dog.set_subscription("robu.in", "monthly")

    review = await world.money.weekly_review(at("2026-10-25", 10))
    assert review.startswith("Money this week (18 Oct - 24 Oct): Rs 1,749 over 3 payments.")
    assert "- to sort: Rs 1,249" in review and "- food: Rs 500" in review
    assert "Biggest: Rs 1,249 to Robu.in, Rs 300 to Zomato, Rs 200 to Zomato." in review
    assert "Waiting on you: 1 payee(s) to sort" in review

    lines = await world.money.brief_lines(at("2026-11-21", 7))
    assert any(line.startswith("1 payment(s) on the watch list: failed payment of Rs 150 "
                               "to Uber") for line in lines)
    assert "Robu.in renews tomorrow (about Rs 1,249)" in lines


async def test_the_jobs_send_once(world: SimpleNamespace) -> None:
    registry = JobTypeRegistry()
    notes = NotificationsRepo(world.db)
    register_money_jobs(registry, world.money, SessionsRepo(world.db), notes)

    async def nothing(*_: Any) -> None: ...

    async def no_file(*_: Any) -> str:
        return new_ulid()

    ctx = JobContext(job_id=new_ulid(), trace_id=new_ulid(), checkpoint=None,
                     save_checkpoint=nothing, progress=nothing, write_artifact=no_file)
    review = registry.get("money.review")
    daily = registry.get("money.daily")
    assert review is not None and review.handler is not None
    assert daily is not None and daily.handler is not None
    payload = review.input_model.model_validate({})
    await review.handler(payload, ctx)
    await review.handler(payload, ctx)
    [note] = await notes.pending()
    assert note.text.startswith("Money this week")
    out = await daily.handler(daily.input_model.model_validate({}), ctx)
    assert out.sent is False                                     # nothing to say: silent


@pytest.fixture
async def client(settings: CoreSettings, world: SimpleNamespace) -> AsyncIterator[httpx.AsyncClient]:
    deps = GatewayDeps(settings=settings, db=world.db, traces=TracesRepo(world.db),
                       started_monotonic=time.monotonic(), money=world.money)
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_the_page_api(client: httpx.AsyncClient, world: SimpleNamespace) -> None:
    await world.pay("Zomato", "250.00", "2026-10-29", status="Failed")
    [item] = (await client.get("/money/watch", headers=AUTH)).json()["items"]
    cleared = await client.post(f"/money/watch/{item['id']}/clear", headers=AUTH,
                                json={"note": "refunded"})
    assert cleared.json()["summary"].startswith("Cleared")
    assert (await client.get("/money/watch", headers=AUTH)).json()["items"] == []

    await world.pay("Netflix", "199.00", "2026-10-20")
    marked = await client.post("/money/subscriptions", headers=AUTH,
                               json={"payee": "Netflix", "cadence": "monthly"})
    assert "next expected 20 Nov" in marked.json()["summary"]
    [sub] = (await client.get("/money/subscriptions", headers=AUTH)).json()["subscriptions"]
    assert sub["payee"] == "Netflix" and sub["source"] == "owner"
    assert (await client.post("/money/subscriptions", headers=AUTH,
                              json={"payee": "x", "cadence": "weekly"})).status_code == 422
