# =============================================================================
# src/jarvis/core/money/plutus.py - PLUTUS, keeper of the books
# =============================================================================
#
# The owner's money has its own agent so that it has its own wall. JARVIS
# never reads the ledger: he asks PLUTUS (ask_plutus), and gets back an
# answer in words. Raw transactions stay inside this loop.
#
# PLUTUS's tools read and write the ledger and nothing else - no web, no
# memory, no other database, and nothing that could move money. Its
# allowlist in agentloop/policies.py says exactly that.
# =============================================================================

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jarvis.agentloop.loop import run_agent_loop
from jarvis.agentloop.policies import PLUTUS, create_guard
from jarvis.agentloop.toolset import InlineTool, Toolset
from jarvis.core.db.repos.money import MoneyError, payee_key
from jarvis.core.money.service import MoneyService, rupees
from jarvis.llm.layer import LLMLayer, user_message
from jarvis.llm.tiers import Tier

PLUTUS_PERSONA_V1 = """\
You are PLUTUS, keeper of the owner's personal accounts inside his
assistant, JARVIS. JARVIS passes you one request at a time - a question
about his spending, or an instruction to correct the books - and passes
your answer back to the owner.

What you track: payments out of his Union Bank account (read from the
bank's alert emails) and cash he mentions. Money coming in is NOT
tracked, by his choice. Transfers to himself or family go in the
"transfers" category, which is never counted as spending.

How you work:
- Use your tools for every number. Never estimate, never round in his
  favour, never invent a payment.
- Categories belong to him. When he says what a payee is ("Campus P is
  the canteen, food"), set a rule with money_set_payee - it applies to
  past and future payments. To change one payment only, use
  money_edit_transaction.
- Spending for Basement Supply, his business, is scope basement_supply.
  Parts for one of his projects can name the project.
- Cash he mentions ("200 cash on chai") is money_add_cash.
- The watch list holds failed and unusual payments. When he says one is
  sorted ("the Zomato one came back"), clear it with money_clear_watch.
- Subscriptions are detected from the ledger; he can add one by hand
  ("Spotify is monthly") or dismiss one ("that's not a subscription")
  with money_set_subscription.

How you answer:
- Short and exact. Amounts as "Rs 1,240". Lead with the answer.
- Facts, not advice. You do not recommend investments or financial
  products. You may point out plain facts in his data ("food is up
  Rs 900 on last month").
- If a request needs something you cannot do, say so in one line."""

ACTOR = PLUTUS


class _Period(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period: str = Field("month", description="today, yesterday, week, month, last_month, "
                                             "YYYY-MM, or YYYY-MM-DD..YYYY-MM-DD")


class _List(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period: str = Field("month", description="As for money_summary.")
    category: str | None = Field(None, description="Only this category; 'to sort' for "
                                                   "uncategorised.")
    payee: str | None = Field(None, description="Only payees containing this text.")
    limit: int = Field(30, ge=1, le=100)


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _SetPayee(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payee: str = Field(min_length=1, description="The payee as the bank wrote it.")
    category: str
    scope: Literal["personal", "basement_supply"] = "personal"
    label: str | None = Field(None, description="A clearer name, e.g. 'Campus canteen'.")
    project: str | None = Field(None, description="A project from his projects database.")


class _Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction: str = Field(description="The id shown in brackets, e.g. 7Q2KX9.")
    category: str | None = None
    scope: Literal["personal", "basement_supply"] | None = None
    project: str | None = None
    note: str | None = None


class _Cash(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: str = Field(description="Rupees, e.g. '200' or '1,249.50'.")
    what: str = Field(min_length=1, description="What or who it was paid to.")
    category: str | None = None
    date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$",
                             description="YYYY-MM-DD; empty for today.")
    scope: Literal["personal", "basement_supply"] = "personal"
    note: str | None = None


class _Clear(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: str = Field(description="The watch item's id shown in brackets.")
    note: str | None = Field(None, description="How it was resolved, in his words.")


class _Subscription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payee: str = Field(min_length=1, description="The payee as the bank wrote it.")
    cadence: Literal["monthly", "yearly", "none"] = Field(
        description="'none' stops treating it as a subscription.")


def build_toolset(money: MoneyService) -> Toolset:
    tools = Toolset(guard=create_guard(), actor=ACTOR)
    repo = money.repo

    async def summary(args: _Period) -> str:
        try:
            return await money.summary_text(args.period)
        except MoneyError as exc:
            return f"error: {exc}"

    async def transactions(args: _List) -> str:
        try:
            start, end, label = money.period(args.period)
        except MoneyError as exc:
            return f"error: {exc}"
        rows = await repo.between(start, end)
        if args.category:
            wanted = args.category.lower()
            rows = [r for r in rows if (r["category"] or "to sort") == wanted]
        if args.payee:
            rows = [r for r in rows if args.payee.lower() in r["payee"].lower()]
        if not rows:
            return f"No matching payments in {label}."
        lines = [
            f"[{r['id'][-6:]}] {r['occurred_at'][:16]} {rupees(r['amount_paise'])} "
            f"to {r['payee']} - {r['category'] or 'to sort'}"
            + (" (Basement Supply)" if r["scope"] == "basement_supply" else "")
            + (f" project {r['project']}" if r["project"] else "")
            + ("" if r["status"] == "success" else f" [{r['status'].upper()}]")
            + (f" note: {r['note']}" if r["note"] else "")
            for r in rows[: args.limit]]
        more = f"\n...and {len(rows) - args.limit} more." if len(rows) > args.limit else ""
        return "\n".join(lines) + more

    async def to_sort(_: _NoArgs) -> str:
        rows = await repo.to_sort()
        if not rows:
            return "Nothing to sort - every payee has a category."
        return "\n".join(f"{r['payee']}: {r['n']} payment(s), {rupees(r['paise'])}, "
                         f"last {r['last']}" for r in rows)

    async def categories(_: _NoArgs) -> str:
        cats = await repo.categories()
        return ", ".join(c["name"] + ("" if c["is_spend"] else " (not spending)") for c in cats)

    async def set_payee(args: _SetPayee) -> str:
        try:
            return await money.categorise_payee(args.payee, args.category, scope=args.scope,
                                                 label=args.label, project=args.project)
        except MoneyError as exc:
            return f"error: {exc}"

    async def edit(args: _Edit) -> str:
        try:
            txn = await money.edit(args.transaction, category=args.category, scope=args.scope,
                                   project=args.project, note=args.note)
        except MoneyError as exc:
            return f"error: {exc}"
        return (f"Updated [{txn['id'][-6:]}]: {rupees(txn['amount_paise'])} to "
                f"{txn['payee_raw']} is {txn['category'] or 'to sort'}.")

    async def cash(args: _Cash) -> str:
        try:
            txn = await money.add_cash(args.amount, args.what, category=args.category,
                                       day=args.date, scope=args.scope, note=args.note)
        except MoneyError as exc:
            return f"error: {exc}"
        return (f"Recorded [{txn['id'][-6:]}] {rupees(txn['amount_paise'])} cash to "
                f"{txn['payee_raw']} on {txn['date']} - {txn['category'] or 'to sort'}.")

    watch = money.watchdog

    async def watchlist(_: _NoArgs) -> str:
        if watch is None:
            return "The watchdog is not running."
        lines = await watch.watch_lines()
        return "\n".join(lines) or "Nothing on the watch list."

    async def clear(args: _Clear) -> str:
        if watch is None:
            return "The watchdog is not running."
        return await watch.clear(args.item, args.note)

    async def subscriptions(_: _NoArgs) -> str:
        if watch is None:
            return "The watchdog is not running."
        lines = await watch.subscription_lines()
        return "\n".join(lines) or "No subscriptions known yet."

    async def set_subscription(args: _Subscription) -> str:
        if watch is None:
            return "The watchdog is not running."
        return await watch.set_subscription(
            payee_key(args.payee), None if args.cadence == "none" else args.cadence)

    for name, description, model, handler in (
        ("money_watchlist", "Failed and unusual payments still waiting for him.", _NoArgs,
         watchlist),
        ("money_clear_watch", "Clear a watch item once he says it is resolved.", _Clear, clear),
        ("money_subscriptions", "Known subscriptions: amount, cadence, next renewal.",
         _NoArgs, subscriptions),
        ("money_set_subscription", "Mark a payee as a monthly or yearly subscription, or "
                                   "stop treating it as one.", _Subscription,
         set_subscription),
        ("money_summary", "Totals for a period: spent, by category, top payees.", _Period, summary),
        ("money_transactions", "Individual payments, newest first, with their ids.", _List,
         transactions),
        ("money_to_sort", "Payees whose payments have no category yet.", _NoArgs, to_sort),
        ("money_categories", "The categories in use.", _NoArgs, categories),
        ("money_set_payee", "Set what a payee is; applies to all its payments, past and "
                            "future, except ones he set by hand.", _SetPayee, set_payee),
        ("money_edit_transaction", "Change one payment's category, scope, project or note.",
         _Edit, edit),
        ("money_add_cash", "Record a cash payment he mentions.", _Cash, cash),
    ):
        tools.register(InlineTool(name=name, description=description, args_model=model,
                                  handler=handler))
    return tools


async def ask_plutus(llm: LLMLayer, money: MoneyService, request: str, *,
                     trace_id: str) -> str:
    result = await run_agent_loop(
        llm, Tier.UTILITY, PLUTUS_PERSONA_V1, [user_message(request)],
        build_toolset(money), actor=ACTOR, trace_id=trace_id,
        max_iterations=6, max_tokens=1200,
    )
    return result.text.strip() or "PLUTUS had nothing to say to that."
