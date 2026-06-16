"""spaCy NLP engine setup for Presidio.

Phase 1 is rule-first, but Presidio's AnalyzerEngine still needs an NLP engine
for tokenization (and, once the real model is installed, for ML-based entity
detection in Phase 2).

If ``en_core_web_sm`` is installed, we use it and get spaCy's NER for free.
If it isn't (CI, restricted networks), we fall back to a blank tokenizer-only
pipeline so the rule-based recognizers still run end-to-end. Install the real
model with::

    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging

import spacy
from presidio_analyzer.nlp_engine import NlpEngine, SpacyNlpEngine

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "en_core_web_sm"


class BlankSpacyNlpEngine(SpacyNlpEngine):
    """Tokenizer-only engine: drives the rule layer without a downloaded model.

    Contributes tokenization (so context-aware scoring and offsets work) but no
    ML entities. That is exactly the Phase 1 baseline: rules do the detecting.
    """

    def load(self) -> None:  # noqa: D401 - override
        self.nlp = {"en": spacy.blank("en")}


def build_nlp_engine(model: str = DEFAULT_MODEL) -> NlpEngine:
    """Return a spaCy-backed NLP engine, falling back to blank if needed."""
    if spacy.util.is_package(model):
        engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": model}])
        engine.load()
        logger.info("Loaded spaCy model '%s' (ML entity detection active).", model)
        return engine

    logger.warning(
        "spaCy model '%s' not installed - running in RULES-ONLY mode. "
        "Free-text entities (names, locations) will be under-detected until you "
        "run: python -m spacy download %s",
        model,
        model,
    )
    engine = BlankSpacyNlpEngine()
    engine.load()
    return engine
