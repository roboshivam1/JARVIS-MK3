# =============================================================================
# src/jarvis/common/life.py - the life log's shapes
# =============================================================================
#
# What one report about the owner's day can contain. The extraction model
# fills ExtractedReport from whatever he says ("up at 7:40, gym done, had
# dal rice in the mess, finished the Coursera module, tomorrow I want to
# fix the ebike wiring"); the service stores it.
#
# Dates are local YYYY-MM-DD and times local HH:MM. Checklist items are
# referred to by handle (C1, C2...), never by id.
# =============================================================================

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ActivityKind = Literal["gym", "study", "project", "learning", "other"]
MealSlot = Literal["breakfast", "lunch", "dinner", "snack"]
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def real_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError("must be a real YYYY-MM-DD date") from None
    return value


def clock(value: str | None) -> str | None:
    """'7:05' -> '07:05'. Rejects anything that is not a time of day."""
    if value is None:
        return None
    value = value.strip()
    if re.fullmatch(r"\d:[0-5]\d", value):
        value = "0" + value
    if not _TIME.match(value):
        raise ValueError("must be a 24-hour HH:MM time")
    return value


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TimeOfDay(_Item):
    date: str = Field(description="The local date this belongs to, YYYY-MM-DD. For "
                                  "sleep, the day that is ending.")
    time: str = Field(description="24-hour HH:MM.")

    @field_validator("date")
    @classmethod
    def _d(cls, v: str) -> str | None:
        return real_date(v)

    @field_validator("time")
    @classmethod
    def _t(cls, v: str) -> str | None:
        return clock(v)


class ActivityItem(_Item):
    date: str
    kind: ActivityKind
    action: Literal["log", "start", "stop"] = Field(
        "log", description="'start' when he is beginning it now, 'stop' when he "
                           "says he has finished something already started, "
                           "'log' for something already done.")
    subject: str | None = Field(None, description="What exactly: 'OS', 'legs', "
                                                  "'Coursera agentic AI module 2'.")
    project: str | None = Field(None, description="For project work: the project's "
                                                  "name as listed in PROJECTS.")
    started_at: str | None = None
    ended_at: str | None = None
    minutes: int | None = Field(None, ge=1, le=1440)
    note: str | None = None

    @field_validator("date")
    @classmethod
    def _d(cls, v: str) -> str | None:
        return real_date(v)

    @field_validator("started_at", "ended_at")
    @classmethod
    def _t(cls, v: str | None) -> str | None:
        return clock(v)


class MealItem(_Item):
    date: str
    slot: MealSlot
    status: Literal["eaten", "skipped"] = "eaten"
    what: str | None = None
    place: str | None = Field(None, description="home, mess, out...")

    @field_validator("date")
    @classmethod
    def _d(cls, v: str) -> str | None:
        return real_date(v)


class TargetItem(_Item):
    date: str = Field(description="The day it is meant for, YYYY-MM-DD.")
    text: str = Field(min_length=2, description="The target, short: 'Finish one "
                                                "module of the Coursera agentic AI course'.")

    @field_validator("date")
    @classmethod
    def _d(cls, v: str) -> str | None:
        return real_date(v)


class NoteItem(_Item):
    date: str
    text: str = Field(min_length=2)

    @field_validator("date")
    @classmethod
    def _d(cls, v: str) -> str | None:
        return real_date(v)


class ExtractedReport(BaseModel):
    """Everything one report said. The tool schema the model fills."""

    wake: TimeOfDay | None = None
    sleep: TimeOfDay | None = None
    activities: list[ActivityItem] = Field(default_factory=list)
    meals: list[MealItem] = Field(default_factory=list)
    done: list[str] = Field(default_factory=list,
                            description="Handles of CHECKLIST items he says are done.")
    new_targets: list[TargetItem] = Field(default_factory=list)
    day_note: NoteItem | None = None

    @property
    def empty(self) -> bool:
        return not (self.wake or self.sleep or self.activities or self.meals
                    or self.done or self.new_targets or self.day_note)


def _every_day() -> list[Weekday]:
    return ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class HabitIn(BaseModel):
    """A fixed target, as the owner creates or edits it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=60)
    kind: ActivityKind | None = None
    weekdays: list[Weekday] = Field(default_factory=_every_day, min_length=1)
    active: bool = True
