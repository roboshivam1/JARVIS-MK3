# =============================================================================
# src/jarvis/common/projects.py - the projects database's shapes
# =============================================================================
#
# What the extraction model may propose, and what the owner may edit.
# Same pattern as common/people.py: every proposal names its project as
# "existing:<id>" or "new:<tag>" (a project first mentioned in the same
# debrief), and rows it changes by HANDLE until the service translates
# handles back into ids.
# =============================================================================

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProjectKind(StrEnum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    MIXED = "mixed"
    RESEARCH = "research"
    CREATIVE = "creative"
    OTHER = "other"


class ProjectStatus(StrEnum):
    IDEA = "idea"
    ACTIVE = "active"
    PAUSED = "paused"
    SHIPPED = "shipped"
    ABANDONED = "abandoned"


class FactCategory(StrEnum):
    SPEC = "spec"                # a number, a dimension, a pin, a version
    CONSTRAINT = "constraint"    # budget, deadline, size, rules
    STATUS = "status"            # where it stands
    LESSON = "lesson"            # what went wrong and what it taught
    RISK = "risk"
    IDEA = "idea"                # not decided, worth keeping
    OTHER = "other"


class PartStatus(StrEnum):
    NEEDED = "needed"
    ORDERED = "ordered"
    HAVE = "have"
    USED = "used"
    DROPPED = "dropped"


class ProposalKind(StrEnum):
    NEW_PROJECT = "new_project"
    ALIAS = "alias"
    STATUS = "status"
    DECISION = "decision"
    FACT = "fact"
    PART = "part"
    PART_UPDATE = "part_update"
    TASK = "task"
    TASK_DONE = "task_done"
    LOG = "log"
    PERSON = "person"
    LINK = "link"


_REF_RE = re.compile(r"^(existing:[A-Za-z0-9]+|new:[A-Za-z0-9_]+)$")


def _real_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError("must be a real YYYY-MM-DD date") from None
    return value


class _Proposal(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _AboutProject(_Proposal):
    project_ref: str = Field(
        description="'existing:<handle>' for a project in PROJECTS, or 'new:<tag>' "
                    "matching a new_projects entry in this debrief.")

    @field_validator("project_ref")
    @classmethod
    def _ref_shape(cls, v: str) -> str:
        if not _REF_RE.match(v):
            raise ValueError(f"project_ref must be existing:<id> or new:<tag>, got {v!r}")
        return v


class NewProjectProposal(_Proposal):
    tag: str = Field(pattern=r"^[A-Za-z0-9_]+$",
                     description="Short tag like N1; others refer to it as new:N1.")
    name: str = Field(min_length=2, max_length=80)
    kind: ProjectKind = ProjectKind.OTHER
    status: ProjectStatus = ProjectStatus.ACTIVE
    pitch: str | None = Field(None, description="One line: what it is.")
    goal: str | None = Field(None, description="What 'done' looks like, if said.")
    started_on: str | None = Field(None, description="YYYY-MM-DD, if said.")

    @field_validator("started_on")
    @classmethod
    def _real(cls, v: str | None) -> str | None:
        return _real_date(v)


class AliasProposal(_AboutProject):
    alias: str = Field(min_length=2, max_length=60,
                       description="Another name the owner uses for it.")


class StatusProposal(_AboutProject):
    status: ProjectStatus


class DecisionProposal(_AboutProject):
    decision: str = Field(min_length=3, description="What was decided, one sentence.")
    why: str | None = Field(None, description="The reason given, in his words.")
    alternatives: str | None = Field(None, description="Options considered and dropped.")
    decided_on: str | None = Field(None, description="YYYY-MM-DD.")
    supersedes_decision_id: str | None = Field(
        None, description="Handle of a KNOWN DECISION this one replaces, e.g. D2.")

    @field_validator("decided_on")
    @classmethod
    def _real(cls, v: str | None) -> str | None:
        return _real_date(v)


class FactProposal(_AboutProject):
    category: FactCategory = FactCategory.OTHER
    text: str = Field(min_length=3, description="One atomic fact, standalone.")
    supersedes_fact_id: str | None = Field(
        None, description="Handle of a KNOWN FACT this one replaces, e.g. F3.")


class PartProposal(_AboutProject):
    name: str = Field(min_length=2)
    spec: str | None = None
    quantity: int = Field(1, ge=1)
    status: PartStatus = PartStatus.NEEDED
    source: str | None = Field(None, description="Where it comes from, if said.")
    cost_inr: float | None = Field(None, ge=0, description="Total for the quantity, in rupees.")


class PartUpdateProposal(_Proposal):
    part_id: str = Field(description="Handle of a KNOWN PART, e.g. B2.")
    status: PartStatus
    note: str | None = None


class TaskProposal(_AboutProject):
    text: str = Field(min_length=3, description="Something still to do.")
    due_date: str | None = Field(None, description="YYYY-MM-DD, if said.")

    @field_validator("due_date")
    @classmethod
    def _real(cls, v: str | None) -> str | None:
        return _real_date(v)


class TaskDoneProposal(_Proposal):
    task_id: str = Field(description="Handle of an OPEN TASK now done, e.g. T1.")


class LogProposal(_AboutProject):
    text: str = Field(min_length=3, description="What happened, one or two sentences.")
    happened_on: str = Field(description="YYYY-MM-DD; today unless he said otherwise.")

    @field_validator("happened_on")
    @classmethod
    def _real(cls, v: str | None) -> str | None:
        return _real_date(v)


class PersonProposal(_AboutProject):
    person_id: str = Field(description="Handle from PEOPLE, e.g. H4.")
    role: str | None = Field(None, description="Their part: mechanical, mentor, client...")


class LinkProposal(_AboutProject):
    label: str = Field(min_length=2)
    url: str = Field(pattern=r"^https?://\S+$")


class ExtractionResult(BaseModel):
    """Everything one debrief yielded. The tool schema the model fills."""

    new_projects: list[NewProjectProposal] = Field(default_factory=list)
    aliases: list[AliasProposal] = Field(default_factory=list)
    status_changes: list[StatusProposal] = Field(default_factory=list)
    decisions: list[DecisionProposal] = Field(default_factory=list)
    facts: list[FactProposal] = Field(default_factory=list)
    parts: list[PartProposal] = Field(default_factory=list)
    part_updates: list[PartUpdateProposal] = Field(default_factory=list)
    tasks: list[TaskProposal] = Field(default_factory=list)
    tasks_done: list[TaskDoneProposal] = Field(default_factory=list)
    log: list[LogProposal] = Field(default_factory=list)
    people: list[PersonProposal] = Field(default_factory=list)
    links: list[LinkProposal] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(len(getattr(self, f)) for f in RESULT_FIELDS.values())


PROPOSAL_MODELS: dict[ProposalKind, type[_Proposal]] = {
    ProposalKind.NEW_PROJECT: NewProjectProposal,
    ProposalKind.ALIAS: AliasProposal,
    ProposalKind.STATUS: StatusProposal,
    ProposalKind.DECISION: DecisionProposal,
    ProposalKind.FACT: FactProposal,
    ProposalKind.PART: PartProposal,
    ProposalKind.PART_UPDATE: PartUpdateProposal,
    ProposalKind.TASK: TaskProposal,
    ProposalKind.TASK_DONE: TaskDoneProposal,
    ProposalKind.LOG: LogProposal,
    ProposalKind.PERSON: PersonProposal,
    ProposalKind.LINK: LinkProposal,
}
RESULT_FIELDS: dict[ProposalKind, str] = {
    ProposalKind.NEW_PROJECT: "new_projects",
    ProposalKind.ALIAS: "aliases",
    ProposalKind.STATUS: "status_changes",
    ProposalKind.DECISION: "decisions",
    ProposalKind.FACT: "facts",
    ProposalKind.PART: "parts",
    ProposalKind.PART_UPDATE: "part_updates",
    ProposalKind.TASK: "tasks",
    ProposalKind.TASK_DONE: "tasks_done",
    ProposalKind.LOG: "log",
    ProposalKind.PERSON: "people",
    ProposalKind.LINK: "links",
}


class ProjectUpdate(BaseModel):
    """The owner correcting a project directly. Only fields sent change."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(None, min_length=2, max_length=80)
    kind: ProjectKind | None = None
    status: ProjectStatus | None = None
    pitch: str | None = None
    goal: str | None = None
    description: str | None = None
    repo_url: str | None = None
    folder: str | None = None

    def changes(self) -> dict[str, Any]:
        return {key: (value.value if isinstance(value, StrEnum) else value)
                for key, value in self.model_dump(exclude_unset=True).items()}
