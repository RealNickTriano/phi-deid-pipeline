"""Reversible pseudonymization of detected PHI.

Plain redaction (``pipeline.deidentify``) is lossy: every span collapses to
``<ENTITY_TYPE>``, so all patients become ``<PERSON>`` and nothing can be
recovered. Pseudonymization keeps two things that redaction throws away:

1. **Typed context for an LLM.** Each value is replaced with a *type-tagged*
   pseudonym like ``<PERSON_3f9a1c7b2d>``. A chatbot still sees that two
   mentions refer to the same person, that a token is a date vs. a name, etc. --
   without ever seeing the real value.
2. **Recoverability.** :func:`pseudonymize` returns a ``vault`` mapping each
   pseudonym back to its original value, so :func:`reidentify` can reconstruct
   the original text from a pseudonymized message (or model reply).

Pseudonyms are deterministic and *value-consistent*: the same ``(entity_type,
value)`` always yields the same pseudonym, across notes and runs, so identifiers
stay linkable. A secret salt makes the pseudonym a keyed MAC (HMAC-SHA256): with
the salt unknown, an attacker can't brute-force small value spaces (e.g.
enumerate every SSN) back to their pseudonyms. The caller supplies the salt (the
CLI reads ``PHI_DEID_SALT``); any fixed salt must be kept secret in production.

    pseudonymized, vault = pseudonymize(text, entities, salt)
    # ... send `pseudonymized` to an LLM; keep `vault` protected ...
    original = reidentify(model_reply, vault)
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Dict, List, Tuple

from phi_deid.pipeline import Entity

# Length of the hex digest kept in the pseudonym. 10 hex chars = 40 bits, plenty
# to avoid collisions across a corpus while staying readable.
_HASH_LEN = 10


def make_pseudonym(entity_type: str, value: str, salt: str) -> str:
    """Return a stable, salted pseudonym for ``value`` of a given type.

    Format: ``f"{entity_type}_{hash}"`` where ``hash`` is a truncated
    HMAC-SHA256 over the type and value (so the same value of different types
    gets different pseudonyms). Deterministic for a fixed salt.
    """
    msg = f"{entity_type}{value}".encode("utf-8")
    digest = hmac.new(salt.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return f"{entity_type}_{digest[:_HASH_LEN]}"


def pseudonymize(
    text: str, entities: List[Entity], salt: str
) -> Tuple[str, Dict[str, str]]:
    """Replace every entity span with its pseudonym.

    Returns ``(pseudonymized_text, vault)`` where ``vault`` maps each pseudonym
    back to its original value. Spans are replaced right-to-left so each entity's
    offsets stay valid as the text is edited (pseudonyms differ in length from
    the values).
    """
    pseudonymized_text = text
    vault: Dict[str, str] = {}
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        value = text[entity.start : entity.end]
        pseudonym = make_pseudonym(entity.entity_type, value, salt)
        vault[pseudonym] = value
        pseudonymized_text = _insert_pseudonym(
            pseudonym, pseudonymized_text, entity.start, entity.end
        )

    return pseudonymized_text, vault


def reidentify(pseudonymized_text: str, vault: Dict[str, str]) -> str:
    """Inverse of :func:`pseudonymize`.

    Substitutes each ``<pseudonym>`` placeholder in ``pseudonymized_text`` back
    to its original value using the ``vault``. Longest pseudonyms are restored
    first so none can collide with the prefix of a longer one.
    """
    original_text = pseudonymized_text
    for pseudonym in sorted(vault, key=len, reverse=True):
        original_text = original_text.replace(f"<{pseudonym}>", vault[pseudonym])
    return original_text


def _insert_pseudonym(pseudonym: str, text: str, start: int, end: int) -> str:
    return f"{text[:start]}<{pseudonym}>{text[end:]}"
