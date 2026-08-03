# =============================================================================
# src/jarvis/core/initiative/policy.py - when JARVIS is allowed to speak
# =============================================================================
#
# The decision layer between "something happened" and "the owner's phone
# buzzes". Every unprompted message passes through here.
#
# BUILT BEFORE WATCHERS, deliberately. Watchers generate unprompted
# messages by design - something checked every ten minutes will
# eventually have something to say at three in the morning. Adding them
# to a system whose policy is "deliver everything immediately" produces
# an assistant the owner mutes within a week, and a muted assistant is
# worse than a silent one because he stops looking.
#
# PRIORITY IS THE WHOLE MECHANISM. The question a priority answers is
# not "how interesting is this" but "does this need the owner NOW, or
# does it merely concern him":
#
#   0-2  urgent - reaches him whatever the hour. Approvals live here:
#        work is PAUSED on his answer, so waiting until morning costs
#        more than the interruption.
#   3-5  normal - delivered during waking hours, deferred otherwise.
#        Finished work, failures worth knowing about.
#   6-9  ambient - batched into a digest. Individually not worth an
#        interruption; collectively worth two minutes.
#
# DEFERRED, NEVER DROPPED. A notification held during quiet hours waits.
# Dropping messages makes a system untrustworthy in a way that is hard
# to notice - the owner cannot tell the difference between "nothing
# happened" and "something happened and you decided not to say", and
# that uncertainty poisons everything else it tells him.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from jarvis.common.notifications import Notification


class Decision(StrEnum):
    DELIVER = "deliver"    # send it now
    DEFER = "defer"        # hold until quiet hours end
    DIGEST = "digest"      # batch with others


@dataclass(frozen=True)
class NotificationPolicy:
    """The rules. Owner-tunable; defaults chosen to be quiet."""

    # Nothing below urgent reaches the owner between these hours.
    quiet_start: time = time(22, 30)
    quiet_end: time = time(7, 30)

    # At or below this priority, deliver regardless of the hour.
    urgent_at_or_below: int = 2

    # At or above this priority, batch rather than interrupt.
    digest_at_or_above: int = 6

    # How long a digest accumulates before being sent.
    digest_window_minutes: int = 90

    # Manual snooze: nothing but urgent until this moment passes.
    # Set by the /quiet command.
    snooze_until: datetime | None = None

    def is_quiet_hour(self, moment: datetime) -> bool:
        """Whether the owner should not be disturbed right now.

        Handles the window crossing midnight, which is the normal case
        for sleeping hours and the easy thing to get wrong.
        """
        now = moment.time()
        if self.quiet_start <= self.quiet_end:
            return self.quiet_start <= now < self.quiet_end
        return now >= self.quiet_start or now < self.quiet_end

    def decide(
        self,
        notification: Notification,
        now: datetime,
        tz: ZoneInfo,
    ) -> Decision:
        """What to do with one notification, right now."""
        local = now.astimezone(tz)

        # Urgent overrides everything, including an explicit snooze. If
        # the owner has told the system to be quiet and a job is paused
        # waiting for his approval, telling him is still correct - the
        # alternative is work silently stuck until he happens to look.
        if notification.priority <= self.urgent_at_or_below:
            return Decision.DELIVER

        if self.snooze_until is not None and now < self.snooze_until:
            return Decision.DEFER

        if self.is_quiet_hour(local):
            return Decision.DEFER

        if notification.priority >= self.digest_at_or_above:
            return Decision.DIGEST

        return Decision.DELIVER

    def next_waking_moment(self, now: datetime, tz: ZoneInfo) -> datetime:
        """When a deferred notification becomes deliverable.

        Used only for logging and for telling the owner when he will
        hear about something - the delivery loop simply retries, so
        nothing depends on this being exact.
        """
        local = now.astimezone(tz)
        candidate = local.replace(
            hour=self.quiet_end.hour, minute=self.quiet_end.minute,
            second=0, microsecond=0,
        )
        if candidate <= local:
            candidate += timedelta(days=1)
        return candidate.astimezone(now.tzinfo)


def compose_digest(notifications: list[Notification]) -> str:
    """Turn several low-priority notifications into one message.

    Deliberately plain: this is a list of things that happened, and
    spending a model call to make it prettier would be paying for
    decoration. If digests ever need JARVIS's voice, that is a utility-
    tier call added here.
    """
    if len(notifications) == 1:
        return notifications[0].text

    lines = [f"While you were away, sir - {len(notifications)} things:", ""]
    for index, note in enumerate(notifications, start=1):
        first_line = note.text.strip().split("\n")[0]
        lines.append(f"{index}. {first_line[:200]}")
    return "\n".join(lines)
