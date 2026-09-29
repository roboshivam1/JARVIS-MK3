# =============================================================================
# src/jarvis/common/people.py - the people database's shapes
# =============================================================================
#
# A person is an ENTITY, not a sentence: facts, commitments, and dates
# hang off them. The vault (common/facts.py) holds free-standing claims
# about the owner's world; this holds structured records of the people
# in it.
#
# The proposal models do double duty. The extraction model answers
# through them - their JSON schema IS the tool schema it fills - and the
# review step re-validates them before anything is written. An edit made
# in the review page meets exactly the checks a fresh extraction does.
#
# person_ref says who a proposal is about:
#   existing:<id>  someone already in the database
#   new:<tag>      a new-person proposal in the same debrief
#
# extra="ignore" rather than the codebase's usual "forbid": these arrive
# from a model and from a browser form, and one stray key should not throw
# away an otherwise good proposal.
# =============================================================================

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PersonStatus(StrEnum):
    ACTIVE = "active"
    DORMANT = "dormant"
    LOST_TOUCH = "lost_touch"


class PersonFactCategory(StrEnum):
    BIOGRAPHICAL = "biographical"
    WORK_STUDY = "work_study"
    PREFERENCES = "preferences"
    SKILLS = "skills"
    FAMILY = "family"
    LIFE_EVENT = "life_event"
    OPINION = "opinion"
    TRAIT = "trait"
    FRICTION = "friction"      # conflicts, sore points, things to handle with care
    OTHER = "other"


class Sensitivity(StrEnum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"    # never auto-injected into conversation context


class CommitmentDirection(StrEnum):
    I_OWE = "i_owe"
    THEY_OWE = "they_owe"


class ProposalKind(StrEnum):
    NEW_PERSON = "new_person"
    ALIAS = "alias"
    FACT = "fact"
    COMMITMENT = "commitment"
    KEY_DATE = "key_date"


class ProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


_REF_RE = re.compile(r"^(existing:[A-Za-z0-9]+|new:[A-Za-z0-9_]+)$")


def _real_date(value: str, recurring: bool) -> bool:
    """Right shape AND on the calendar - a regex alone accepts 02-30.
    Recurring dates are checked against 2000, a leap year, so a
    29 February birthday is allowed."""
    shape = r"\d{2}-\d{2}" if recurring else r"\d{4}-\d{2}-\d{2}"
    if not re.fullmatch(shape, value):
        return False
    try:
        datetime.strptime(f"2000-{value}" if recurring else value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


class _Proposal(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _AboutPerson(_Proposal):
    person_ref: str = Field(
        description="'existing:<handle>' for someone on the roster, or "
                    "'new:<tag>' matching a new_people entry in this debrief."
    )

    @field_validator("person_ref")
    @classmethod
    def _ref_shape(cls, v: str) -> str:
        if not _REF_RE.match(v):
            raise ValueError(f"person_ref must be existing:<id> or new:<tag>, got {v!r}")
        return v

class NewPersonProposal(_Proposal):
    tag: str = Field(pattern=r"^[A-Za-z0-9_]+$",
                     description="Short tag like P1; others refer to it as new:P1.")
    name: str = Field(min_length=1)
    relationship: str | None = Field(
        None, description="friend, family, mentor, colleague, client...")
    closeness: int = Field(2, ge=1, le=3,
                           description="1 close, 2 regular, 3 acquaintance")
    how_met: str | None = None
    summary: str | None = Field(None, description="One line: who they are to the owner.")


class AliasProposal(_AboutPerson):
    alias: str = Field(min_length=1,
                       description="A nickname or other way the owner refers to them.")


class FactProposal(_AboutPerson):
    category: PersonFactCategory
    text: str = Field(min_length=3,
                      description="One atomic fact, a short standalone sentence.")
    sensitivity: Sensitivity = Sensitivity.NORMAL
    supersedes_fact_id: str | None = Field(
        None, description="Handle of a KNOWN FACT this one replaces, if any.")


class CommitmentProposal(_AboutPerson):
    direction: CommitmentDirection
    what: str = Field(min_length=3)
    due_date: str | None = Field(None, description="Absolute YYYY-MM-DD, if any.")

    @field_validator("due_date")
    @classmethod
    def _due_is_a_date(cls, v: str | None) -> str | None:
        if v is not None and not _real_date(v, recurring=False):
            raise ValueError("due_date must be a real YYYY-MM-DD date")
        return v


class KeyDateProposal(_AboutPerson):
    label: str = Field(min_length=2, description="e.g. 'birthday', 'end-sem exams'")
    date: str = Field(description="MM-DD if recurring, YYYY-MM-DD if one-off.")
    recurring: bool = True

    @model_validator(mode="after")
    def _date_matches_recurrence(self) -> "KeyDateProposal":
        if not _real_date(self.date, recurring=self.recurring):
            shape = "MM-DD" if self.recurring else "YYYY-MM-DD"
            raise ValueError(f"date must be a real {shape} date")
        return self


class ExtractionResult(BaseModel):
    """Everything one debrief yielded. The tool schema the model fills."""

    new_people: list[NewPersonProposal] = Field(default_factory=list)
    aliases: list[AliasProposal] = Field(default_factory=list)
    facts: list[FactProposal] = Field(default_factory=list)
    commitments: list[CommitmentProposal] = Field(default_factory=list)
    key_dates: list[KeyDateProposal] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return (len(self.new_people) + len(self.aliases) + len(self.facts)
                + len(self.commitments) + len(self.key_dates))


# Proposal kind -> the model that validates its payload, and the
# ExtractionResult field it lives in. Order matters: new people first,
# so everything that depends on them is staged after them.
PROPOSAL_MODELS: dict[ProposalKind, type[_Proposal]] = {
    ProposalKind.NEW_PERSON: NewPersonProposal,
    ProposalKind.ALIAS: AliasProposal,
    ProposalKind.FACT: FactProposal,
    ProposalKind.COMMITMENT: CommitmentProposal,
    ProposalKind.KEY_DATE: KeyDateProposal,
}
RESULT_FIELDS: dict[ProposalKind, str] = {
    ProposalKind.NEW_PERSON: "new_people",
    ProposalKind.ALIAS: "aliases",
    ProposalKind.FACT: "facts",
    ProposalKind.COMMITMENT: "commitments",
    ProposalKind.KEY_DATE: "key_dates",
}


# =============================================================================
# The owner's own edits
#
# Everything above describes what the extraction model PROPOSES. What
# follows is what the owner does directly on the People page - correcting
# a name, retiring a stale fact, sorting someone into a group. These skip
# the review queue by design: review exists to gate the model, not him.
# =============================================================================


class GroupKind(StrEnum):
    FAMILY = "family"
    SCHOOL = "school"
    COLLEGE = "college"
    LOCALITY = "locality"      # the colony, the neighbourhood
    PROJECT = "project"        # a team built around something being made
    WORK = "work"
    OTHER = "other"


class GroupIn(BaseModel):
    """A group as the owner creates or renames it."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=2, max_length=60)
    kind: GroupKind = GroupKind.OTHER
    description: str | None = Field(None, max_length=200)


class PersonUpdate(BaseModel):
    """A correction to a person. Every field optional: only what is sent
    changes, so the page can save one field at a time."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(None, min_length=1)
    relationship: str | None = None
    closeness: int | None = Field(None, ge=1, le=3)
    how_met: str | None = None
    summary: str | None = None
    status: PersonStatus | None = None

    def changes(self) -> dict[str, object]:
        """Only the fields actually sent, ready for a SET clause."""
        return {
            key: (value.value if isinstance(value, StrEnum) else value)
            for key, value in self.model_dump(exclude_unset=True).items()
        }


class OwnerFact(BaseModel):
    """A fact the owner writes himself, or his correction of one the
    model proposed. Editing supersedes rather than overwrites, so the
    older version stays readable in the person's history."""

    model_config = ConfigDict(extra="ignore")

    category: PersonFactCategory = PersonFactCategory.OTHER
    text: str = Field(min_length=3)
    sensitivity: Sensitivity = Sensitivity.NORMAL
