"""people/models.py — the shape of every proposal.

The LLM fills an ExtractionResult. Each item becomes one row in the
proposals table, and ReviewQueue re-validates it with the same model
before committing — so an edited proposal from the frontend gets
exactly the same checks as a fresh one from the LLM.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

FactCategory = Literal[
    "biographical", "work_study", "preferences", "skills", "family",
    "life_event", "opinion", "trait", "friction", "other",
]
Sensitivity = Literal["normal", "sensitive"]

_REF = re.compile(r"^(existing:\d+|new:[A-Za-z0-9_]+)$")


class _AboutPerson(BaseModel):
    person_ref: str = Field(
        description="'existing:<id>' for someone on the roster, "
                    "or 'new:<tag>' matching a new_person entry's tag."
    )

    @field_validator("person_ref")
    @classmethod
    def _check_ref(cls, v: str) -> str:
        if not _REF.match(v):
            raise ValueError(f"person_ref must be 'existing:<id>' or 'new:<tag>', got {v!r}")
        return v


class NewPerson(BaseModel):
    tag: str = Field(description="Short tag like P1. Others reference it as 'new:P1'.")
    name: str
    relationship: Optional[str] = Field(None, description="friend, family, mentor, colleague, client...")
    closeness: int = Field(2, ge=1, le=3, description="1 close, 2 regular, 3 acquaintance")
    how_met: Optional[str] = None
    summary: Optional[str] = Field(None, description="One line: who they are to the owner.")


class AliasProposal(_AboutPerson):
    alias: str = Field(description="Nickname or another way the owner refers to them.")


class FactProposal(_AboutPerson):
    category: FactCategory
    text: str = Field(description="One atomic fact, short standalone sentence.")
    sensitivity: Sensitivity = "normal"
    supersedes_fact_id: Optional[int] = Field(
        None, description="ID of an existing fact this one replaces, if any."
    )


class CommitmentProposal(_AboutPerson):
    direction: Literal["i_owe", "they_owe"]
    what: str
    due_date: Optional[str] = Field(None, description="YYYY-MM-DD, absolute.")

    @field_validator("due_date")
    @classmethod
    def _check_due(cls, v):
        if v is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("due_date must be YYYY-MM-DD")
        return v


class KeyDateProposal(_AboutPerson):
    label: str = Field(description="e.g. 'birthday', 'end-sem exams'")
    date: str = Field(description="MM-DD if recurring, YYYY-MM-DD if one-off.")
    recurring: bool = True

    @model_validator(mode="after")
    def _check_date(self):
        pattern = r"\d{2}-\d{2}" if self.recurring else r"\d{4}-\d{2}-\d{2}"
        if not re.fullmatch(pattern, self.date):
            fmt = "MM-DD" if self.recurring else "YYYY-MM-DD"
            raise ValueError(f"{'recurring' if self.recurring else 'one-off'} date must be {fmt}")
        return self


class ExtractionResult(BaseModel):
    new_people:  list[NewPerson]          = Field(default_factory=list)
    aliases:     list[AliasProposal]      = Field(default_factory=list)
    facts:       list[FactProposal]       = Field(default_factory=list)
    commitments: list[CommitmentProposal] = Field(default_factory=list)
    key_dates:   list[KeyDateProposal]    = Field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(len(getattr(self, f)) for f in type(self).model_fields)


# Maps a proposal's `kind` column to the model that validates its payload.
# Order matters: new_person first, so dependents always come after.
PAYLOAD_MODELS = {
    "new_person": NewPerson,
    "alias":      AliasProposal,
    "fact":       FactProposal,
    "commitment": CommitmentProposal,
    "key_date":   KeyDateProposal,
}
RESULT_FIELDS = {
    "new_person": "new_people", "alias": "aliases", "fact": "facts",
    "commitment": "commitments", "key_date": "key_dates",
}
