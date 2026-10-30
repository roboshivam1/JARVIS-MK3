# =============================================================================
# src/jarvis/core/money/watchdog.py - PLUTUS keeping watch
# =============================================================================
#
# Three things, all judged against the owner's OWN history rather than
# thresholds he would have to pick:
#
#   FAILED PAYMENTS   a payment the bank marks failed goes on the watch
#                     list with a note to check the money was not taken.
#                     It stays there until he clears it.
#
#   UNUSUAL PAYMENTS  far above what he usually pays that payee, or a
#                     large first payment to someone new. Sent at urgent
#                     priority - quiet hours do not hold a possible fraud
#                     until morning - with the bank's own block number.
#
#   SUBSCRIPTIONS     payees paid on a monthly or yearly cycle, detected
#                     from the ledger after a few cycles (or marked by
#                     hand). Warned two days before each renewal; a price
#                     change is pointed out; a payment that never came
#                     is mentioned once.
#
# Everything here only READS the ledger and SENDS messages. Nothing can
# stop, block or move a payment - that stays with the owner and his bank.
# =============================================================================

from __future__ import annotations

import calendar
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from jarvis.common.log import get_logger
from jarvis.core.db.repos.money import MoneyRepo
from jarvis.core.db.repos.money_watch import MoneyWatchRepo

log = get_logger("core.money.watchdog")

Notify = Callable[[str, int], Awaitable[None]]

URGENT = 2         # delivered even in quiet hours
NORMAL = 3
QUIET = 4

_HISTORY_DAYS = 400
_RECENT_DAYS = 60
_WARN_DAYS = 2
_MISSED_GRACE_DAYS = 5
_PRICE_TOLERANCE = 0.05
_SAME_AMOUNT = 0.20

# From the bank's own alert email: how to block the account if a payment
# was not his.
_BLOCK_HINT = ("If this wasn't you: SMS BLOCK {last4} to 8879365472 from your "
               "registered number, or call Union Bank on 1800 8333.")


def rs(paise: int) -> str:
    whole, frac = divmod(int(paise), 100)
    return f"Rs {whole:,}" + (f".{frac:02d}" if frac else "")


def add_cycle(day: date, cadence: str) -> date:
    """The same day next month (or year), clamped to the month's end."""
    if cadence == "yearly":
        year, month = day.year + 1, day.month
    else:
        year, month = (day.year + (day.month == 12)), (day.month % 12) + 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last))


class Watchdog:
    def __init__(self, ledger: MoneyRepo, repo: MoneyWatchRepo, tz: ZoneInfo, *,
                 notify: Notify | None = None, unusual_floor_paise: int = 50_000) -> None:
        self._ledger = ledger
        self._repo = repo
        self._tz = tz
        self._notify = notify
        self._floor = unusual_floor_paise

    @property
    def repo(self) -> MoneyWatchRepo:
        return self._repo

    def set_notify(self, notify: Notify) -> None:
        self._notify = notify

    def today(self, moment: datetime | None = None) -> date:
        return (moment or datetime.now(self._tz)).astimezone(self._tz).date()

    async def _say(self, text: str, priority: int) -> None:
        if self._notify is not None:
            await self._notify(text, priority)

    async def _name(self, txn: dict[str, Any]) -> str:
        rule = await self._ledger.rule(txn["payee_key"])
        return str(rule["label"]) if rule and rule["label"] else str(txn["payee_raw"])

    # -- every new payment ------------------------------------------------------

    async def on_recorded(self, txn: dict[str, Any]) -> None:
        """Look at a payment as it arrives. Never raises: the watchdog
        failing must not lose the payment itself."""
        try:
            if txn["status"] != "success":
                await self._failed(txn)
                return
            await self._unusual(txn)
            await self._subscription_paid(txn)
        except Exception:
            log.exception("watchdog check failed", extra={"txn": txn.get("id")})

    async def _failed(self, txn: dict[str, Any]) -> None:
        name = await self._name(txn)
        reason = f"{txn['status']} payment of {rs(txn['amount_paise'])} to {name}"
        if await self._repo.flag("failed", txn["id"], reason):
            await self._say(
                f"A payment of {rs(txn['amount_paise'])} to {name} at "
                f"{txn['occurred_at'][11:16]} is marked {txn['status']}, sir. If money left "
                f"the account anyway, the bank usually reverses it within a few working "
                f"days. It stays on the watch list at /app/money until you clear it.",
                NORMAL)

    async def _unusual(self, txn: dict[str, Any]) -> None:
        amount = int(txn["amount_paise"])
        if amount < self._floor:
            return
        name = await self._name(txn)
        before = await self._repo.amounts_of_payee(txn["payee_key"], txn["id"])
        reason = None
        if len(before) >= 3:
            usual = statistics.median(before)
            if amount > 3 * usual:
                reason = f"{rs(amount)} to {name}, where you usually pay about {rs(int(usual))}"
        elif not before:
            since = (date.fromisoformat(txn["date"]) - timedelta(days=_RECENT_DAYS)).isoformat()
            recent = await self._repo.recent_amounts(since, txn["id"])
            if len(recent) >= 10 and amount >= max(2 * self._floor, 5 * statistics.median(recent)):
                reason = f"{rs(amount)} to {name}, a payee you have never paid before"
        if reason and await self._repo.flag("unusual", txn["id"], reason):
            hint = _BLOCK_HINT.format(last4=txn["account_last4"] or "<last 4 digits>")
            await self._say(f"Unusual payment, sir: {reason}. {hint}", URGENT)

    async def _subscription_paid(self, txn: dict[str, Any]) -> None:
        sub = await self._repo.subscription(txn["payee_key"])
        if sub is None or sub["status"] != "active" or txn["date"] <= sub["last_date"]:
            return
        amount = int(txn["amount_paise"])
        paid = date.fromisoformat(txn["date"])
        typical = int(sub["typical_paise"])
        changed = abs(amount - typical) > typical * _PRICE_TOLERANCE
        await self._repo.save_subscription(
            txn["payee_key"], cadence=sub["cadence"], typical=amount if changed else typical,
            last=amount, last_date=paid.isoformat(),
            next_due=add_cycle(paid, sub["cadence"]).isoformat(), source=sub["source"])
        if changed:
            name = await self._name(txn)
            await self._say(
                f"{name} charged {rs(amount)} this time, sir - it used to be {rs(typical)}.",
                QUIET)

    # -- once a day -------------------------------------------------------------

    async def detect(self, moment: datetime | None = None) -> list[str]:
        """Find payees paid on a monthly or yearly cycle. Returns new ones."""
        today = self.today(moment)
        since = (today - timedelta(days=_HISTORY_DAYS)).isoformat()
        found: list[str] = []
        for key in await self._repo.recurring_candidates(since):
            existing = await self._repo.subscription(key)
            if existing is not None and (existing["status"] == "dismissed"
                                         or existing["source"] == "owner"):
                continue
            pays = await self._repo.payments_of(key, since)
            cycle = self._cycle(pays)
            if cycle is None:
                continue
            cadence, typical = cycle
            last = pays[-1]
            last_day = date.fromisoformat(last["date"])
            await self._repo.save_subscription(
                key, cadence=cadence, typical=typical, last=int(last["amount_paise"]),
                last_date=last_day.isoformat(),
                next_due=add_cycle(last_day, cadence).isoformat(), source="detected")
            if existing is None:
                found.append(key)
        return found

    @staticmethod
    def _cycle(pays: list[dict[str, Any]]) -> tuple[str, int] | None:
        """(cadence, typical paise) if these payments repeat on a cycle."""
        amounts = [int(p["amount_paise"]) for p in pays]
        usual = statistics.median(amounts)
        similar = [p for p in pays if abs(int(p["amount_paise"]) - usual) <= usual * _SAME_AMOUNT]
        days = [date.fromisoformat(p["date"]) for p in similar]
        gaps = [(b - a).days for a, b in zip(days, days[1:])]
        if len(gaps) >= 2 and all(25 <= g <= 35 for g in gaps[-2:]):
            return "monthly", int(usual)
        if gaps and 350 <= gaps[-1] <= 380:
            return "yearly", int(usual)
        return None

    async def daily(self, moment: datetime | None = None) -> str:
        """New subscriptions, renewals due soon, and payments that never
        came - as one message, or empty when there is nothing to say."""
        today = self.today(moment)
        new = await self.detect(moment)
        lines: list[str] = []
        subs = {s["payee_key"]: s for s in await self._repo.subscriptions()}
        if new:
            lines.append("New regular payment(s) spotted: " + ", ".join(
                f"{subs[k]['payee']} ({rs(subs[k]['typical_paise'])} {subs[k]['cadence']})"
                for k in new if k in subs) + ". Dismiss any that are not subscriptions at "
                "/app/money.")
        for s in subs.values():
            due = date.fromisoformat(s["next_due"])
            if 0 <= (due - today).days <= _WARN_DAYS and s["warned_for"] != s["next_due"]:
                when = "today" if due == today else ("tomorrow" if (due - today).days == 1
                                                     else due.strftime("%a %d %b"))
                lines.append(f"{s['payee']} renews {when} - about {rs(s['typical_paise'])}.")
                await self._repo.mark(s["payee_key"], "warned_for", s["next_due"])
            elif (today - due).days > _MISSED_GRACE_DAYS and s["missed_for"] != s["next_due"]:
                lines.append(f"{s['payee']} was due on {due.strftime('%d %b')} and has not "
                             f"been paid - cancelled, or did a payment fail?")
                await self._repo.mark(s["payee_key"], "missed_for", s["next_due"])
        return "\n".join(lines)

    # -- reading back -----------------------------------------------------------

    async def watch_lines(self) -> list[str]:
        items = await self._repo.open_items()
        return [f"[{i['id'][-6:]}] {i['kind']}: {i['reason']} ({i['date']})" for i in items]

    async def clear(self, item: str, note: str | None = None) -> str:
        found = await self._repo.find(item)
        if found is None:
            return f"No open watch item {item}."
        await self._repo.clear(found["id"], note)
        return f"Cleared: {found['reason']}."

    async def set_subscription(self, key: str, cadence: str | None) -> str:
        """Mark a payee as a subscription by hand, or stop treating it as one."""
        if cadence is None:
            changed = await self._repo.set_subscription_status(key, "dismissed")
            return "Dismissed." if changed else "That payee was not a subscription."
        since = (self.today() - timedelta(days=_HISTORY_DAYS)).isoformat()
        pays = await self._repo.payments_of(key, since)
        if not pays:
            return "No payments to that payee yet - nothing to base it on."
        last = pays[-1]
        last_day = date.fromisoformat(last["date"])
        await self._repo.save_subscription(
            key, cadence=cadence, typical=int(last["amount_paise"]),
            last=int(last["amount_paise"]), last_date=last_day.isoformat(),
            next_due=add_cycle(last_day, cadence).isoformat(), source="owner")
        return (f"Marked as a {cadence} subscription; next expected "
                f"{add_cycle(last_day, cadence).strftime('%d %b')}.")

    async def subscription_lines(self) -> list[str]:
        return [f"{s['payee']}: {rs(s['typical_paise'])} {s['cadence']}, next "
                f"{date.fromisoformat(s['next_due']).strftime('%d %b')}"
                for s in await self._repo.subscriptions()]
