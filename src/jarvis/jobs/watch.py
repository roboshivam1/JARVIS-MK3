# =============================================================================
# src/jarvis/jobs/watch.py - the watch.check job type
# =============================================================================
#
# One scheduled job runs ALL enabled watchers. Not one job per watcher:
# a dozen watchers would mean a dozen schedules to keep in step, and a
# single tick that checks everything is simpler and cheaper.
#
# EACH CHECK IS ISOLATED. One watcher pointing at a dead URL must not
# stop the others - a failed check logs and moves on, the same
# discipline the notifier needed after one bad row wedged it.
#
# A HIT BECOMES A NOTIFICATION, not a direct message: watchers go
# through the same policy choke point as everything else, so a page
# changing at 3am waits until morning unless the owner marked it
# urgent.
# =============================================================================

from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.events import Event, EventKind
from jarvis.common.ids import new_ulid, utc_now
from jarvis.common.log import get_logger
from jarvis.common.notifications import Notification
from jarvis.common.watchers import Watcher, WatcherKind
from jarvis.core.db.repos.events import EventsRepo
from jarvis.core.db.repos.notifications import NotificationsRepo
from jarvis.core.db.repos.watchers import WatchersRepo
from jarvis.core.observability.traces import TracesRepo
from jarvis.core.queue.registry import JobContext, JobTypeRegistry, JobTypeSpec

log = get_logger("jobs.watch")

_FETCH_TIMEOUT_S = 30
_MAX_PAGE_CHARS = 200_000


class WatchCheckIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="scheduled")


class WatchCheckOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checked: int
    hits: int
    errors: int


def register_watch_jobs(
    registry: JobTypeRegistry,
    watchers: WatchersRepo,
    notifications: NotificationsRepo,
    events: EventsRepo,
    traces: TracesRepo,
    default_client_kind: str = "telegram",
) -> None:
    """Register the watcher tick. Runs on the Core - checks are HTTP and
    database reads, nothing a worker provides."""

    async def check_web_page(watcher: Watcher) -> tuple[bool, str, dict[str, Any]]:
        """Has this page changed since last time?

        Compares a HASH of the visible text, not the raw HTML: pages
        carry session tokens, timestamps, and rotating ad markup that
        change on every load and mean nothing. Hashing stripped text
        is the difference between a useful watcher and one that fires
        every fifteen minutes forever.
        """
        url = str(watcher.config.get("url", ""))
        if not url:
            raise ValueError("web_page watcher has no url")

        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_S, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            body = response.text[:_MAX_PAGE_CHARS]

        text = re.sub(r"<script.*?</script>", " ", body, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        previous = watcher.state.get("digest")
        new_state = {"digest": digest, "length": len(text)}

        if previous is None:
            # First run establishes the baseline. Reporting a "change"
            # here would mean every new watcher fires immediately.
            return False, "baseline recorded", new_state

        if previous == digest:
            return False, "unchanged", new_state

        old_length = int(watcher.state.get("length", 0))
        delta = len(text) - old_length
        return True, (
            f"The page changed ({delta:+d} characters).\n{url}"
        ), new_state

    async def check_spend(watcher: Watcher) -> tuple[bool, str, dict[str, Any]]:
        """Has today's spend crossed a threshold?

        Fires ONCE per day per crossing: without that, a threshold
        passed at noon would notify every tick until midnight.
        """
        threshold = float(watcher.config.get("threshold_inr", 200.0))
        rows = await traces.daily_costs(days=1)
        today_inr = rows[0].cost_inr if rows else 0.0
        today = utc_now().strftime("%Y-%m-%d")

        new_state = {"last_alerted_day": watcher.state.get("last_alerted_day")}
        if today_inr < threshold:
            return False, f"spend {today_inr:.2f} under {threshold:.2f}", new_state
        if watcher.state.get("last_alerted_day") == today:
            return False, "already alerted today", new_state

        new_state["last_alerted_day"] = today
        return True, (
            f"Today's API spend has passed the threshold, sir: "
            f"Rs {today_inr:.2f} against a limit of Rs {threshold:.2f}."
        ), new_state

    async def check_job_health(watcher: Watcher) -> tuple[bool, str, dict[str, Any]]:
        """Is a job type failing repeatedly?"""
        job_type = str(watcher.config.get("job_type", ""))
        threshold = int(watcher.config.get("consecutive_failures", 2))
        if not job_type:
            raise ValueError("job_health watcher has no job_type")

        recent = await events.recent(kind=EventKind.JOB_FAILED, limit=20)
        failures = [
            e for e in recent
            if e.payload.get("terminal") and job_type in str(e.payload)
        ][:threshold]

        last_seen = watcher.state.get("last_failure_event")
        newest = failures[0].id if failures else None
        new_state = {"last_failure_event": newest}

        if len(failures) < threshold or newest == last_seen:
            return False, "healthy or already reported", new_state
        return True, (
            f"{job_type} has failed {len(failures)} times in a row, sir."
        ), new_state

    async def check_idle(watcher: Watcher) -> tuple[bool, str, dict[str, Any]]:
        """Has something NOT happened for too long?

        Absence is harder to notice than presence, which is exactly why
        it is worth watching.
        """
        event_kind = str(watcher.config.get("event_kind", ""))
        max_hours = float(watcher.config.get("max_idle_hours", 72))
        if not event_kind:
            raise ValueError("idle watcher has no event_kind")

        recent = await events.recent(kind=event_kind, limit=1)
        now = utc_now()
        if not recent:
            return True, (
                f"Nothing of kind {event_kind} has ever been recorded, sir."
            ), {"last_alerted": now.isoformat()}

        idle_for = now - recent[0].ts
        if idle_for < timedelta(hours=max_hours):
            return False, f"last seen {idle_for.total_seconds() / 3600:.1f}h ago", {}

        # Do not repeat more than daily once idle.
        last_alerted = watcher.state.get("last_alerted")
        if last_alerted and (now - utc_now().fromisoformat(last_alerted)) < timedelta(days=1):
            return False, "already alerted today", watcher.state

        return True, (
            f"No {event_kind} for {idle_for.days} days, sir - "
            f"you asked to be told after {max_hours:.0f} hours."
        ), {"last_alerted": now.isoformat()}

    _CHECKS = {
        WatcherKind.WEB_PAGE: check_web_page,
        WatcherKind.SPEND: check_spend,
        WatcherKind.JOB_HEALTH: check_job_health,
        WatcherKind.IDLE: check_idle,
    }

    async def handle(payload: BaseModel, ctx: JobContext) -> BaseModel:
        assert isinstance(payload, WatchCheckIn)
        checked = hits = errors = 0

        for watcher in await watchers.enabled():
            check = _CHECKS.get(watcher.kind)
            if check is None:
                continue

            try:
                fired, message, new_state = await check(watcher)
            except Exception as exc:
                # One dead URL must not stop the rest. Same discipline
                # the notifier needed after a single bad row wedged it.
                errors += 1
                log.warning("watcher check failed", exc_info=True, extra={
                    "watcher": watcher.name, "kind": watcher.kind.value,
                })
                await watchers.record_check(watcher.id, {}, hit=False)
                continue

            checked += 1
            await watchers.record_check(watcher.id, new_state, hit=fired)

            if not fired:
                continue

            hits += 1
            trace_id = new_ulid()
            text = message
            if watcher.note:
                text += f"\n\nYou asked: {watcher.note}"

            # Through the outbox and the policy, like everything else.
            # A watcher does not get to interrupt on its own authority.
            await notifications.create(Notification(
                client_kind=default_client_kind,
                text=text,
                priority=watcher.priority,
                trace_id=trace_id,
            ))
            await events.append(Event(
                kind=EventKind.INITIATIVE_WATCHER_HIT,
                source="core.initiative", job_id=ctx.job_id,
                trace_id=trace_id,
                payload={"watcher": watcher.name, "kind": watcher.kind.value},
            ))
            log.info("watcher hit", extra={"watcher": watcher.name})

        return WatchCheckOut(checked=checked, hits=hits, errors=errors)

    registry.register(JobTypeSpec(
        type="watch.check",
        input_model=WatchCheckIn,
        output_model=WatchCheckOut,
        execution="idempotent",
        requires=[],
        default_priority=7,      # background work, never ahead of the owner
        timeout_s=300,
        handler=handle,
    ))
