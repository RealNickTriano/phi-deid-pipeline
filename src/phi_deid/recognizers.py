"""Custom HIPAA recognizers for structured identifiers.

Presidio ships strong recognizers for SSN, email, phone, credit cards, IPs, etc.
These add the healthcare-specific structured identifiers from the HIPAA Safe
Harbor list that the defaults don't cover: medical record numbers, account
numbers, health-plan beneficiary/member numbers, and device/serial numbers.
A regex date recognizer is included too, since dates are a Safe Harbor
identifier and rules catch the common formats without needing the ML layer.

Each recognizer uses a high-confidence *labeled* pattern (the identifier
preceded by a label like "MRN:") plus an optional low-confidence bare-value
pattern that only survives when Presidio's context-aware scoring boosts it.
With the real spaCy model the context boost is active; in rules-only mode only
the labeled patterns clear the score threshold - a deliberate precision-first
default for the baseline.
"""

from __future__ import annotations

from typing import List

from presidio_analyzer import Pattern, PatternRecognizer

# --- entity type names (canonical, used across pipeline + eval) ---------------
MEDICAL_RECORD_NUMBER = "MEDICAL_RECORD_NUMBER"
ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
BENEFICIARY_NUMBER = "BENEFICIARY_NUMBER"
DEVICE_ID = "DEVICE_ID"
DATE_TIME = "DATE_TIME"  # aligns with Presidio's built-in date entity name


def _mrn_recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=MEDICAL_RECORD_NUMBER,
        name="MrnRecognizer",
        patterns=[
            Pattern(
                "mrn_labeled",
                r"\b(?:MRN|Medical\s+Record\s+(?:Number|No\.?|#))\s*[:#]?\s*"
                r"([A-Z]{0,3}\d{6,10})\b",
                0.85,
            ),
            Pattern("mrn_value", r"\b[A-Z]{0,3}\d{7,10}\b", 0.30),
        ],
        context=["mrn", "medical record", "record number", "chart", "patient id"],
    )


def _account_recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=ACCOUNT_NUMBER,
        name="AccountNumberRecognizer",
        patterns=[
            Pattern(
                "acct_labeled",
                r"\b(?:Account|Acct)\.?\s*(?:Number|No\.?|#)?\s*[:#]?\s*(\d{8,12})\b",
                0.85,
            ),
            Pattern("acct_value", r"\b\d{8,12}\b", 0.30),
        ],
        context=["account", "acct", "billing", "invoice"],
    )


def _beneficiary_recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=BENEFICIARY_NUMBER,
        name="BeneficiaryNumberRecognizer",
        patterns=[
            Pattern(
                "beneficiary_labeled",
                r"\b(?:Beneficiary|Member|Subscriber|Policy|Insurance)\s*"
                r"(?:ID|Number|No\.?|#)?\s*[:#]?\s*([A-Z]{0,4}\d{5,12})\b",
                0.80,
            ),
        ],
        context=["beneficiary", "member", "subscriber", "policy", "insurance", "plan"],
    )


def _device_recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=DEVICE_ID,
        name="DeviceIdRecognizer",
        patterns=[
            # The label keyword + separator are mandatory so the value group
            # can't latch onto an ordinary word (patterns are IGNORECASE, so a
            # bare ``[A-Z0-9][A-Z0-9-]{4,}`` would otherwise match "Serial").
            # The value must contain a digit, ruling out plain words entirely.
            Pattern(
                "device_labeled",
                r"\b(?:Serial|Device|Implant|Lot)\s+(?:Number|No\.?|#|ID)\s*"
                r"[:#]\s*([A-Z0-9-]*\d[A-Z0-9-]*)\b",
                0.75,
            ),
        ],
        context=["serial", "device", "implant", "lot", "model"],
    )


def _date_recognizer() -> PatternRecognizer:
    months = (
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    )
    return PatternRecognizer(
        supported_entity=DATE_TIME,
        name="DateRecognizer",
        patterns=[
            Pattern("date_numeric", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", 0.60),
            Pattern("date_iso", r"\b\d{4}-\d{2}-\d{2}\b", 0.60),
            Pattern(
                "date_monthname",
                rf"\b{months}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
                0.65,
            ),
        ],
        context=["date", "dob", "born", "admitted", "discharged", "visit", "seen"],
    )


def get_custom_recognizers() -> List[PatternRecognizer]:
    """Return all custom HIPAA recognizers to register with the analyzer."""
    return [
        _mrn_recognizer(),
        _account_recognizer(),
        _beneficiary_recognizer(),
        _device_recognizer(),
        _date_recognizer(),
    ]
