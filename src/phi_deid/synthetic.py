"""Synthetic labeled clinical notes.

The i2b2/n2c2 de-identification corpus is the real benchmark, but it requires a
data use agreement that takes time to approve. This module generates clinical-
note-style text with injected PHI and *exact* ground-truth spans, so the
evaluation harness runs from day one. Because we control injection, labels are
perfect and free.

Each note is a ``LabeledNote`` with ``text`` and a list of ``Span`` objects
(char offsets + entity type + value), aligned to the same entity-type names the
recognizers emit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List

from faker import Faker

from .recognizers import (
    ACCOUNT_NUMBER,
    BENEFICIARY_NUMBER,
    DATE_TIME,
    MEDICAL_RECORD_NUMBER,
)

PERSON = "PERSON"
PHONE_NUMBER = "PHONE_NUMBER"
EMAIL_ADDRESS = "EMAIL_ADDRESS"
US_SSN = "US_SSN"
LOCATION = "LOCATION"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity_type: str
    value: str


@dataclass
class LabeledNote:
    text: str
    spans: List[Span] = field(default_factory=list)


class _NoteBuilder:
    """Append literal text and PHI fields while tracking char offsets."""

    def __init__(self) -> None:
        self._parts: List[str] = []
        self._spans: List[Span] = []
        self._pos = 0

    def lit(self, text: str) -> "_NoteBuilder":
        self._parts.append(text)
        self._pos += len(text)
        return self

    def phi(self, value: str, entity_type: str) -> "_NoteBuilder":
        start = self._pos
        self._parts.append(value)
        self._pos += len(value)
        self._spans.append(Span(start, self._pos, entity_type, value))
        return self

    def build(self) -> LabeledNote:
        return LabeledNote("".join(self._parts), list(self._spans))


def _mrn(fake: Faker) -> str:
    return str(fake.random_number(digits=8, fix_len=True))


def _acct(fake: Faker) -> str:
    return str(fake.random_number(digits=10, fix_len=True))


def _beneficiary(fake: Faker) -> str:
    return fake.bothify("??########").upper()


# --- note templates -----------------------------------------------------------
# Each template returns a LabeledNote. Variety matters: PHI appears in different
# positions, formats, and surrounding language.


def _template_admit(fake: Faker) -> LabeledNote:
    b = _NoteBuilder()
    (
        b.lit("ADMISSION NOTE\n\nPatient ")
        .phi(fake.name(), PERSON)
        .lit(" (MRN: ")
        .phi(_mrn(fake), MEDICAL_RECORD_NUMBER)
        .lit(") was admitted on ")
        .phi(fake.date(pattern="%m/%d/%Y"), DATE_TIME)
        .lit(" to ")
        .phi(fake.city(), LOCATION)
        .lit(" General Hospital. Date of birth ")
        .phi(fake.date(pattern="%m/%d/%Y"), DATE_TIME)
        .lit(". SSN ")
        .phi(fake.ssn(), US_SSN)
        .lit(".\nPrimary contact: ")
        .phi(fake.phone_number(), PHONE_NUMBER)
        .lit(", ")
        .phi(fake.email(), EMAIL_ADDRESS)
        .lit(".")
    )
    return b.build()


def _template_followup(fake: Faker) -> LabeledNote:
    b = _NoteBuilder()
    (
        b.lit("FOLLOW-UP\n\n")
        .phi(fake.name(), PERSON)
        .lit(" returned for follow-up on ")
        .phi(fake.date(pattern="%B %d, %Y"), DATE_TIME)
        .lit(". Patient reports improvement. Insurance Member ID ")
        .phi(_beneficiary(fake), BENEFICIARY_NUMBER)
        .lit(", Account ")
        .phi(_acct(fake), ACCOUNT_NUMBER)
        .lit(".\nDiscussed results by phone (")
        .phi(fake.phone_number(), PHONE_NUMBER)
        .lit("). Next visit scheduled ")
        .phi(fake.date(pattern="%m/%d/%Y"), DATE_TIME)
        .lit(".")
    )
    return b.build()


def _template_referral(fake: Faker) -> LabeledNote:
    b = _NoteBuilder()
    (
        b.lit("REFERRAL\n\nReferring ")
        .phi(fake.name(), PERSON)
        .lit(" for ")
        .phi(fake.name(), PERSON)
        .lit(", residing in ")
        .phi(fake.city(), LOCATION)
        .lit(". Reachable at ")
        .phi(fake.email(), EMAIL_ADDRESS)
        .lit(" or ")
        .phi(fake.phone_number(), PHONE_NUMBER)
        .lit(". Chart MRN ")
        .phi(_mrn(fake), MEDICAL_RECORD_NUMBER)
        .lit(". Seen ")
        .phi(fake.date(pattern="%m-%d-%Y"), DATE_TIME)
        .lit(".")
    )
    return b.build()


def _template_discharge(fake: Faker) -> LabeledNote:
    b = _NoteBuilder()
    (
        b.lit("DISCHARGE SUMMARY\n\n")
        .phi(fake.name(), PERSON)
        .lit(" was discharged ")
        .phi(fake.date(pattern="%m/%d/%Y"), DATE_TIME)
        .lit(". Follow instructions mailed to address in ")
        .phi(fake.city(), LOCATION)
        .lit(". Beneficiary ")
        .phi(_beneficiary(fake), BENEFICIARY_NUMBER)
        .lit(". Questions: ")
        .phi(fake.phone_number(), PHONE_NUMBER)
        .lit(".")
    )
    return b.build()


_TEMPLATES: List[Callable[[Faker], LabeledNote]] = [
    _template_admit,
    _template_followup,
    _template_referral,
    _template_discharge,
]


def generate_notes(n: int = 50, seed: int = 7) -> List[LabeledNote]:
    """Generate ``n`` labeled synthetic clinical notes."""
    fake = Faker("en_US")
    Faker.seed(seed)
    random.seed(seed)
    notes: List[LabeledNote] = []
    for i in range(n):
        template = _TEMPLATES[i % len(_TEMPLATES)]
        notes.append(template(fake))
    return notes
