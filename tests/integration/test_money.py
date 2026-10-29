# =============================================================================
# tests/integration/test_money.py - PLUTUS and the ledger
# =============================================================================
#
# The promises: the real Union Bank alert parses exactly; one payment is
# one row however many alerts describe it; the mailbox reads only new
# mail and never takes the daemon down; a payee is categorised once and
# stays that way; transfers and failures are never counted as spending;
# PLUTUS can touch the ledger and nothing else.
# =============================================================================

from __future__ import annotations

import time
from datetime import datetime
from email.message import EmailMessage
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import httpx
import pytest

from jarvis.agentloop.policies import ORCHESTRATOR, PLUTUS, create_guard
from jarvis.common.settings import CoreSettings
from jarvis.core.db.database import Database
from jarvis.core.db.repos.money import MoneyRepo
from jarvis.core.gateway.http import GatewayDeps, create_app
from jarvis.core.money.mailbox import Fetched
from jarvis.core.money.parsers import message_text, parse_alert
from jarvis.core.money.plutus import build_toolset
from jarvis.core.money.service import MoneyService
from jarvis.core.observability.traces import TracesRepo

JAIPUR = ZoneInfo("Asia/Kolkata")
AUTH = {"Authorization": "Bearer test-token"}

# Exactly as the email reads, minus the greeting's name.
REAL = """Greetings!
Your fund transfer request through UPI has been processed successfully.
Transaction Details
1. Payee Name : Campus P
2. Amount : Rs. 65.00
3. Channel : UPI
4. Transaction ID/RRN : 216155719815
5. Transaction Status : Success
6. Transaction Date and Time : 29-10-2026 11:19:38
7. Debit Account Number : *9572
If you have not initiated this transaction, please report it immediately"""


def alert(payee: str, amount: str, ref: str, when: str = "29-10-2026 13:39:21",
          status: str = "Success") -> str:
    return (REAL.replace("Campus P", payee).replace("65.00", amount)
            .replace("216155719815", ref).replace("29-10-2026 11:19:38", when)
            .replace("Status : Success", f"Status : {status}"))


def email_of(text: str, subject: str = "DEBIT TRANSACTION ALERT") -> EmailMessage:
    """The alert as UBI sends it: an HTML email, one <li> per field."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "noreplyubi-txn@ubi.bank.in"
    items = "".join(f"<li>{line[3:]}</li>" if line[:1].isdigit() else f"<p>{line}</p>"
                    for line in text.splitlines())
    msg.set_content("plain fallback")
    msg.add_alternative(f"<html><head><style>p{{color:red}}</style></head><body>"
                        f"<div>Dear <b>SHIVAM KAPOOR</b>,</div><ol>{items}</ol></body></html>",
                        subtype="html")
    return msg


class FakeMailbox:
    def __init__(self) -> None:
        self.inbox: list[Fetched] = []
        self.fail = False

    async def fetch_since(self, last_uid: int | None, validity: str | None
                          ) -> tuple[list[Fetched], int, str]:
        if self.fail:
            raise OSError("imap.gmail.com unreachable")
        top = max((m.uid for m in self.inbox), default=0)
        if last_uid is None:
            return [], top, "v1"
        new = [m for m in self.inbox if m.uid > last_uid]
        return new, max([last_uid, *(m.uid for m in new)]), "v1"


@pytest.fixture
def world(db: Database) -> Any:
    from types import SimpleNamespace
    box = FakeMailbox()
    money = MoneyService(MoneyRepo(db), JAIPUR, mailbox=box)
    return SimpleNamespace(db=db, box=box, money=money, repo=money.repo)


def test_the_real_alert_parses_exactly() -> None:
    parsed = parse_alert(REAL)
    assert parsed is not None
    assert (parsed.payee, parsed.amount_paise, parsed.ref, parsed.status,
            parsed.account_last4, parsed.channel) == (
        "Campus P", 6500, "216155719815", "success", "9572", "UPI")
    assert parsed.occurred_at == datetime(2026, 10, 29, 11, 19, 38)
    assert parse_alert(alert("SHIVAM K.", "1,500.5", "1")).amount_paise == 150050  # type: ignore[union-attr]
    assert parse_alert(alert("X", "10.00", "2", status="Failed")).status == "failed"  # type: ignore[union-attr]
    assert parse_alert("Your OTP is 123456") is None
    assert parse_alert(REAL.replace("Debit Account", "Credit Account")) is None


def test_the_html_email_reads_back_as_the_same_text() -> None:
    text = message_text(email_of(REAL))
    assert "1. Payee" not in text and "Payee Name : Campus P" in text
    assert "color" not in text                         # styles dropped
    assert parse_alert(text) is not None


async def test_one_payment_is_one_row(world: Any) -> None:
    first = await world.money.ingest_text(REAL)
    again = await world.money.ingest_text(REAL)
    assert first is not None and again is None


async def test_the_mailbox_reads_only_new_mail_and_survives_failure(world: Any) -> None:
    world.box.inbox.append(Fetched(1, email_of(alert("Old", "99.00", "100"))))
    assert (await world.money.poll())["added"] == 0             # first run: no history
    world.box.inbox += [Fetched(2, email_of(alert("Campus P", "65.00", "201"))),
                        Fetched(3, email_of("Your OTP is 4411", subject="OTP"))]
    assert await world.money.poll() == {"configured": True, "added": 1, "skipped": 1}
    assert await world.money.poll() == {"configured": True, "added": 0, "skipped": 0}

    world.box.fail = True
    result = await world.money.poll()
    assert "unreachable" in result["error"]
    status = await world.money.status()
    assert "unreachable" in status["last_error"]
    lines = await world.money.brief_lines(datetime(2026, 10, 30, 7, 35, tzinfo=JAIPUR))
    assert any("mailbox is failing" in line for line in lines)


async def test_a_payee_is_sorted_once(world: Any) -> None:
    await world.money.ingest_text(alert("Campus P", "65.00", "1", "28-10-2026 10:00:00"))
    owner_set = await world.money.ingest_text(alert("Campus P", "300.00", "2"))
    await world.money.edit(owner_set["id"], category="gifts")
    assert [p["payee"] for p in await world.repo.to_sort()] == ["Campus P"]

    said = await world.money.categorise_payee("campus p.", "food", label="Campus canteen")
    assert "1 payment(s) updated" in said                       # the owner's choice stands
    later = await world.money.ingest_text(alert("Campus P", "40.00", "3"))
    assert later["category"] == "food"
    assert await world.repo.to_sort() == []


async def test_spending_leaves_out_transfers_and_failures(world: Any) -> None:
    await world.money.ingest_text(alert("Campus P", "65.00", "1"))
    await world.money.ingest_text(alert("SHIVAM K.", "500.00", "2"))
    await world.money.ingest_text(alert("Zomato", "250.00", "3", status="Failed"))
    await world.money.categorise_payee("SHIVAM K.", "transfers")
    await world.money.add_cash("1,20", "chai", category="food", day="2026-10-29")
    s = await world.money.summary("2026-10-01", "2026-10-31")
    assert s["total_paise"] == 6500 + 12000
    assert s["transfers_paise"] == 50000 and len(s["failed"]) == 1
    assert s["by_category"] == {"food": 12000, "to sort": 6500}
    text = await world.money.summary_text("2026-10")
    assert text.startswith("Spent Rs 185 over 2 payments in October 2026.")
    assert "Transfers (not counted): Rs 500." in text and "1 failed" in text


async def test_plutus_works_the_books_and_nothing_else(world: Any) -> None:
    tools = build_toolset(world.money)
    names = {spec.name for spec in tools.specs()}
    assert names and all(n.startswith("money_") for n in names)

    await world.money.ingest_text(alert("Robu.in", "1,249.00", "9"))
    out, _ = await tools.execute("money_set_payee", {"payee": "Robu.in", "category": "project parts",
                                                     "scope": "basement_supply"})
    assert "project parts (Basement Supply); 1 payment(s) updated" in out
    out, _ = await tools.execute("money_add_cash", {"amount": "200", "what": "chai",
                                                    "category": "food", "date": "2026-10-29"})
    assert out.startswith("Recorded [") and "Rs 200 cash to chai on 2026-10-29 - food" in out
    out, _ = await tools.execute("money_transactions", {"period": "2026-10"})
    assert "Rs 1,249 to Robu.in - project parts (Basement Supply)" in out
    out, _ = await tools.execute("money_set_payee", {"payee": "x", "category": "crypto"})
    assert out.startswith("error: No category 'crypto'")

    guard = create_guard()
    assert guard.check(PLUTUS, "money_summary", {}).verdict.value == "allow"
    assert guard.check(PLUTUS, "memory_search", {}).verdict.value != "allow"
    assert guard.check(ORCHESTRATOR, "money_summary", {}).verdict.value != "allow"
    assert guard.check(ORCHESTRATOR, "ask_plutus", {}).verdict.value == "allow"


@pytest.fixture
async def client(settings: CoreSettings, world: Any) -> AsyncIterator[httpx.AsyncClient]:
    deps = GatewayDeps(settings=settings, db=world.db, traces=TracesRepo(world.db),
                       started_monotonic=time.monotonic(), money=world.money)
    transport = httpx.ASGITransport(app=create_app(deps))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_the_page_and_its_api(client: httpx.AsyncClient) -> None:
    page = await client.get("/app/money")
    assert page.status_code == 200 and "To sort" in page.text
    assert (await client.get("/money/period")).status_code == 401

    pasted = await client.post("/money/alert", headers=AUTH, json={"text": REAL})
    assert pasted.json() == {"recorded": True}
    assert (await client.post("/money/alert", headers=AUTH,
                              json={"text": "this is not a bank alert at all"})).status_code == 422
    [payee] = (await client.get("/money/to-sort", headers=AUTH)).json()["payees"]
    ruled = await client.post("/money/rules", headers=AUTH,
                              json={"payee": payee["payee"], "category": "food"})
    assert ruled.status_code == 200
    october = (await client.get("/money/period?period=2026-10", headers=AUTH)).json()
    assert october["total_paise"] == 6500 and october["by_category"] == {"food": 6500}
    txn = october["transactions"][0]["id"]
    assert (await client.patch(f"/money/transactions/{txn}", headers=AUTH,
                               json={"category": "nonsense"})).status_code == 422
    assert (await client.post("/money/cash", headers=AUTH,
                              json={"amount": "abc", "what": "chai"})).status_code == 422
    summary = (await client.get("/money/summary", headers=AUTH)).json()
    assert summary["configured"] is True and summary["to_sort"] == 0
