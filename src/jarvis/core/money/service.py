# =============================================================================
# src/jarvis/core/money/service.py - the ledger's working parts
# =============================================================================
#
# Alerts in (from the mailbox, or pasted), transactions out, categorised
# by the owner's own rules. Summaries for PLUTUS, the page and the brief.
#
# CATEGORIES ARE HIS. A new payee arrives uncategorised and waits in "to
# sort"; the first time he says what it is, a rule is made and every
# payment to that payee - past and future - follows it. No model guesses
# a category: "Campus P" could be the canteen or the print shop, and only
# he knows.
#
# Spending means: successful debits in a category that counts as spend.
# Transfers (to himself, to family) are recorded but never counted.
# =============================================================================

from __future__ import annotations

import asyncio
import calendar
import re
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from jarvis.common.ids import utc_now
from jarvis.common.log import get_logger
from jarvis.core.db.repos.money import MoneyError, MoneyRepo, payee_key
from jarvis.core.money.mailbox import MailSource
from jarvis.core.money.parsers import ParsedDebit, message_text, parse_alert

if TYPE_CHECKING:
    from jarvis.core.db.repos.projects import ProjectsRepo

log = get_logger("core.money")


def rupees(paise: int) -> str:
    """12345 -> 'Rs 123.45'; whole rupees shown without paise."""
    whole, frac = divmod(int(paise), 100)
    text = f"{whole:,}"
    return f"Rs {text}" + (f".{frac:02d}" if frac else "")


def to_paise(amount: float | str) -> int:
    value = str(amount).replace(",", "").replace("₹", "").replace("Rs", "").strip()
    if not re.fullmatch(r"\d+(\.\d{1,2})?", value):
        raise MoneyError(f"Not an amount: {amount!r}")
    whole, _, frac = value.partition(".")
    paise = int(whole) * 100 + int((frac + "00")[:2])
    if paise <= 0:
        raise MoneyError("An amount must be more than zero.")
    return paise


class MoneyService:
    def __init__(self, repo: MoneyRepo, tz: ZoneInfo, *, mailbox: MailSource | None = None,
                 projects: "ProjectsRepo | None" = None) -> None:
        self._repo = repo
        self._tz = tz
        self._mailbox = mailbox
        self._projects = projects

    @property
    def repo(self) -> MoneyRepo:
        return self._repo

    def today(self, moment: datetime | None = None) -> date:
        return (moment or datetime.now(self._tz)).astimezone(self._tz).date()

    # -- getting transactions in -------------------------------------------------

    async def record(self, parsed: ParsedDebit, source: str) -> dict[str, Any] | None:
        """Store one parsed alert, applying the payee's rule if there is one.
        None if it was already recorded."""
        key = payee_key(parsed.payee)
        rule = await self._repo.rule(key)
        txn_id = await self._repo.insert({
            "ref": parsed.ref,
            "occurred_at": parsed.occurred_at.strftime("%Y-%m-%d %H:%M:%S"),
            "date": parsed.occurred_at.date().isoformat(),
            "amount_paise": parsed.amount_paise,
            "payee_raw": parsed.payee,
            "payee_key": key,
            "category": rule["category"] if rule else None,
            "categorized_by": "rule" if rule else None,
            "scope": rule["scope"] if rule else "personal",
            "project_id": rule["project_id"] if rule else None,
            "channel": parsed.channel,
            "status": parsed.status,
            "account_last4": parsed.account_last4,
            "source": source,
        })
        return await self._repo.get(txn_id) if txn_id else None

    async def ingest_text(self, text: str, source: str = "web") -> dict[str, Any] | None:
        """A bank alert pasted or forwarded as text."""
        parsed = parse_alert(text)
        if parsed is None:
            raise MoneyError("That does not look like a bank debit alert I can read.")
        return await self.record(parsed, source)

    async def poll(self) -> dict[str, Any]:
        """Read new mail. Never raises: a failure is recorded for the page
        and the brief, and the next poll tries again."""
        if self._mailbox is None:
            return {"configured": False}
        now = utc_now().isoformat()
        try:
            last = await self._repo.state("mail_last_uid")
            validity = await self._repo.state("mail_validity")
            fetched, mark, new_validity = await self._mailbox.fetch_since(
                int(last) if last else None, validity)
            added = skipped = 0
            for item in fetched:
                parsed = parse_alert(message_text(item.message))
                if parsed is None:
                    skipped += 1
                    continue
                added += await self.record(parsed, "email") is not None
            await self._repo.set_state("mail_last_uid", str(mark))
            await self._repo.set_state("mail_validity", new_validity)
            await self._repo.set_state("mail_last_ok", now)
            await self._repo.set_state("mail_last_error", "")
            if added:
                log.info("money alerts recorded", extra={"added": added, "skipped": skipped})
            return {"configured": True, "added": added, "skipped": skipped}
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:300]
            await self._repo.set_state("mail_last_error", message)
            await self._repo.set_state("mail_last_error_at", now)
            log.warning("mailbox poll failed", extra={"error": message})
            return {"configured": True, "error": message}

    async def run(self, interval_s: float) -> None:
        """The resident loop: poll the mailbox, sleep, repeat. A resident
        rather than a scheduled job so that a check every five minutes
        does not leave 288 job rows a day behind it."""
        if self._mailbox is None:
            log.info("no money mailbox configured - bank alerts are not read")
            await asyncio.Event().wait()
        while True:
            await self.poll()
            await asyncio.sleep(interval_s)

    async def add_cash(self, amount: float | str, what: str, *, category: str | None = None,
                       day: str | None = None, scope: str = "personal",
                       note: str | None = None, source: str = "chat") -> dict[str, Any]:
        paise = to_paise(amount)
        when = date.fromisoformat(day) if day else self.today()
        if category:
            category = await self._repo.check_category(category)
        key = payee_key(what)
        rule = await self._repo.rule(key)
        txn_id = await self._repo.insert({
            "occurred_at": f"{when.isoformat()} 12:00:00", "date": when.isoformat(),
            "amount_paise": paise, "payee_raw": what.strip()[:120], "payee_key": key,
            "category": category or (rule["category"] if rule else None),
            "categorized_by": "owner" if category else ("rule" if rule else None),
            "scope": scope, "channel": "cash", "status": "success",
            "source": source, "note": note,
        })
        assert txn_id is not None
        txn = await self._repo.get(txn_id)
        assert txn is not None
        return txn

    # -- sorting ----------------------------------------------------------------

    async def categorise_payee(self, payee: str, category: str, *, scope: str = "personal",
                               label: str | None = None, project: str | None = None) -> str:
        cat = await self._repo.check_category(category)
        key = payee_key(payee)
        project_id = await self._project_id(project)
        changed = await self._repo.set_rule(key, cat, scope=scope, label=label,
                                            project_id=project_id)
        where = " (Basement Supply)" if scope == "basement_supply" else ""
        return (f"{label or payee} is now {cat}{where}; {changed} payment(s) updated, "
                f"and future ones will follow.")

    async def edit(self, txn_id: str, *, category: str | None = None, scope: str | None = None,
                   project: str | None = None, note: str | None = None) -> dict[str, Any]:
        txn = await self._repo.get(txn_id) or await self._repo.find_short(txn_id)
        if txn is None:
            raise MoneyError(f"No transaction {txn_id}.")
        fields: dict[str, Any] = {}
        if category is not None:
            fields["category"] = await self._repo.check_category(category)
            fields["categorized_by"] = "owner"
        if scope is not None:
            fields["scope"] = scope
        if project is not None:
            fields["project_id"] = await self._project_id(project)
        if note is not None:
            fields["note"] = note
        await self._repo.update(txn["id"], **fields)
        updated = await self._repo.get(txn["id"])
        assert updated is not None
        return updated

    async def _project_id(self, project: str | None) -> str | None:
        if not project:
            return None
        if self._projects is None:
            raise MoneyError("The projects database is not available.")
        found = await self._projects.find(project)
        if len(found) != 1:
            raise MoneyError(f"No single project called {project!r}.")
        return str(found[0]["id"])

    # -- reading back -----------------------------------------------------------

    def period(self, name: str, moment: datetime | None = None) -> tuple[str, str, str]:
        """(start, end, label) for today | yesterday | week | month |
        last_month | YYYY-MM | YYYY-MM-DD..YYYY-MM-DD."""
        today = self.today(moment)
        name = (name or "month").strip().lower()
        if name == "today":
            return today.isoformat(), today.isoformat(), "today"
        if name == "yesterday":
            y = today - timedelta(days=1)
            return y.isoformat(), y.isoformat(), "yesterday"
        if name == "week":
            start = today - timedelta(days=6)
            return start.isoformat(), today.isoformat(), "the last 7 days"
        if name in ("month", "this_month"):
            return today.replace(day=1).isoformat(), today.isoformat(), today.strftime("%B so far")
        if name == "last_month":
            end = today.replace(day=1) - timedelta(days=1)
            return end.replace(day=1).isoformat(), end.isoformat(), end.strftime("%B %Y")
        if re.fullmatch(r"\d{4}-\d{2}", name):
            year, month = int(name[:4]), int(name[5:])
            last = calendar.monthrange(year, month)[1]
            start = date(year, month, 1)
            return start.isoformat(), date(year, month, last).isoformat(), start.strftime("%B %Y")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}", name):
            a, b = name.split("..")
            return a, b, f"{a} to {b}"
        raise MoneyError("Period is today, yesterday, week, month, last_month, YYYY-MM, "
                         "or YYYY-MM-DD..YYYY-MM-DD.")

    async def summary(self, start: str, end: str) -> dict[str, Any]:
        rows = await self._repo.between(start, end)
        spend = [r for r in rows if r["status"] == "success" and r["is_spend"]]
        by_category: dict[str, int] = {}
        by_day: dict[str, int] = {}
        by_payee: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        for r in spend:
            cat = r["category"] or "to sort"
            by_category[cat] = by_category.get(cat, 0) + r["amount_paise"]
            by_day[r["date"]] = by_day.get(r["date"], 0) + r["amount_paise"]
            by_payee[r["payee"]] = by_payee.get(r["payee"], 0) + r["amount_paise"]
            by_scope[r["scope"]] = by_scope.get(r["scope"], 0) + r["amount_paise"]
        return {
            "start": start, "end": end,
            "total_paise": sum(r["amount_paise"] for r in spend),
            "count": len(spend),
            "by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
            "by_day": by_day,
            "top_payees": sorted(by_payee.items(), key=lambda kv: -kv[1])[:8],
            "by_scope": by_scope,
            "failed": [r for r in rows if r["status"] != "success"],
            "transfers_paise": sum(r["amount_paise"] for r in rows
                                   if r["status"] == "success" and not r["is_spend"]),
            "transactions": rows,
        }

    async def summary_text(self, period: str = "month") -> str:
        start, end, label = self.period(period)
        s = await self.summary(start, end)
        if not s["count"]:
            return f"No spending recorded for {label}."
        lines = [f"Spent {rupees(s['total_paise'])} over {s['count']} payments in {label}."]
        lines.append("By category: " + ", ".join(
            f"{k} {rupees(v)}" for k, v in s["by_category"].items()) + ".")
        lines.append("Top payees: " + ", ".join(
            f"{k} {rupees(v)}" for k, v in s["top_payees"][:5]) + ".")
        if s["by_scope"].get("basement_supply"):
            lines.append(f"Of which Basement Supply: {rupees(s['by_scope']['basement_supply'])}.")
        if s["transfers_paise"]:
            lines.append(f"Transfers (not counted): {rupees(s['transfers_paise'])}.")
        if s["failed"]:
            lines.append(f"{len(s['failed'])} failed or pending payment(s) - check they "
                         f"were not debited.")
        return " ".join(lines)

    async def brief_lines(self, moment: datetime | None = None) -> list[str]:
        today = self.today(moment)
        yesterday = (today - timedelta(days=1)).isoformat()
        lines: list[str] = []
        y = await self.summary(yesterday, yesterday)
        m = await self.summary(today.replace(day=1).isoformat(), today.isoformat())
        if y["count"]:
            lines.append(f"yesterday: {rupees(y['total_paise'])} over {y['count']} payments")
        if m["count"]:
            lines.append(f"{today.strftime('%B')} so far: {rupees(m['total_paise'])}")
        sort = await self._repo.to_sort()
        if sort:
            lines.append(f"{len(sort)} payee(s) to sort at /app/money")
        error = await self._repo.state("mail_last_error")
        if error:
            lines.append(f"the bank-alert mailbox is failing: {error[:120]}")
        return lines

    async def status(self) -> dict[str, Any]:
        return {
            "configured": self._mailbox is not None,
            "last_ok": await self._repo.state("mail_last_ok"),
            "last_error": await self._repo.state("mail_last_error") or None,
            "last_error_at": await self._repo.state("mail_last_error_at"),
        }
