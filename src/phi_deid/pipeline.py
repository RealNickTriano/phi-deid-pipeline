"""The de-identification pipeline: detect PHI, then redact it.

Stage [2] detect  -> Presidio AnalyzerEngine (default + custom HIPAA recognizers)
Stage [4] redact  -> Presidio AnonymizerEngine (Phase 1: replace each span with
                     a ``<ENTITY_TYPE>`` tag)

Phase 3 will swap the simple tag replacement for utility-preserving operators
(consistent surrogates, date-shifting). The seams for that live in
``deidentify`` via the anonymizer ``operators`` argument.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from .nlp import build_nlp_engine
from .recognizers import get_custom_recognizers

logger = logging.getLogger(__name__)

# Below this score, candidates are dropped. Bare-value patterns (0.30) only
# survive when context-aware scoring boosts them above this line.
SCORE_THRESHOLD = 0.40


@dataclass(frozen=True)
class Entity:
    """A detected PHI span (engine-agnostic, easy to serialize/compare)."""

    entity_type: str
    start: int
    end: int
    score: float

    @classmethod
    def from_result(cls, r: RecognizerResult) -> "Entity":
        return cls(r.entity_type, r.start, r.end, round(float(r.score), 4))


@lru_cache(maxsize=1)
def build_analyzer() -> AnalyzerEngine:
    logger.debug("Building AnalyzerEngine (first call; result is cached)")
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    custom = get_custom_recognizers()
    for recognizer in custom:
        registry.add_recognizer(recognizer)
    logger.debug(
        "Registry loaded: %d predefined recognizers + %d custom (%s)",
        len(registry.recognizers) - len(custom),
        len(custom),
        ", ".join(r.name for r in custom),
    )
    return AnalyzerEngine(registry=registry, nlp_engine=build_nlp_engine())


@lru_cache(maxsize=1)
def build_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


def detect(text: str, language: str = "en") -> List[RecognizerResult]:
    """Return raw analyzer results above the score threshold."""
    logger.debug(
        "detect: analyzing %d chars (language=%s, score_threshold=%.2f)",
        len(text),
        language,
        SCORE_THRESHOLD,
    )
    results = build_analyzer().analyze(
        text=text, language=language, score_threshold=SCORE_THRESHOLD
    )
    logger.debug("detect: %d span(s) above threshold", len(results))
    for r in results:
        logger.debug(
            "  %-22s [%d:%d] score=%.4f text=%r",
            r.entity_type,
            r.start,
            r.end,
            r.score,
            text[r.start : r.end],
        )
    return results


def detect_entities(text: str, language: str = "en") -> List[Entity]:
    """Return detected PHI as simple Entity records."""
    return [Entity.from_result(r) for r in detect(text, language)]


def deidentify(text: str, language: str = "en") -> Tuple[str, List[Entity]]:
    """Detect PHI and return (redacted_text, entities).

    Phase 1 replaces each detected span with ``<ENTITY_TYPE>``. To move to
    Phase 3, pass an ``operators`` mapping to ``anonymize`` (e.g. surrogate
    generation per entity type, date-shifting).
    """
    results = detect(text, language)
    anonymized = build_anonymizer().anonymize(text=text, analyzer_results=results)
    logger.debug(
        "deidentify: redacted %d span(s); text %d -> %d chars",
        len(results),
        len(text),
        len(anonymized.text),
    )
    return anonymized.text, [Entity.from_result(r) for r in results]
