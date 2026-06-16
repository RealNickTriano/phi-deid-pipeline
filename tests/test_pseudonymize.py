"""Unit tests for reversible PHI pseudonymization.

These build ``Entity`` records directly, so they exercise the pseudonymization
logic without the analyzer or any model.
"""

import re

from phi_deid.pipeline import Entity
from phi_deid.pseudonymize import make_pseudonym, pseudonymize, reidentify

SALT = "test-salt"


def _ent(start, end, etype="PERSON"):
    return Entity(etype, start, end, 0.9)


# --- make_pseudonym ----------------------------------------------------------


def test_pseudonym_format():
    p = make_pseudonym("PERSON", "Jane Doe", SALT)
    assert re.fullmatch(r"PERSON_[0-9a-f]{10}", p)


def test_pseudonym_is_deterministic():
    assert make_pseudonym("PERSON", "Jane Doe", SALT) == make_pseudonym(
        "PERSON", "Jane Doe", SALT
    )


def test_distinct_values_get_distinct_pseudonyms():
    assert make_pseudonym("PERSON", "Jane Doe", SALT) != make_pseudonym(
        "PERSON", "Bob Roe", SALT
    )


def test_same_value_different_type_differs():
    assert make_pseudonym("PERSON", "Jane Doe", SALT) != make_pseudonym(
        "DATE_TIME", "Jane Doe", SALT
    )


def test_salt_changes_pseudonym():
    assert make_pseudonym("PERSON", "Jane Doe", SALT) != make_pseudonym(
        "PERSON", "Jane Doe", "other-salt"
    )


# --- pseudonymize ------------------------------------------------------------


def test_pseudonymize_replaces_span_and_fills_vault():
    text = "Patient Jane Doe seen today."
    out, vault = pseudonymize(text, [_ent(8, 16)], SALT)
    person = make_pseudonym("PERSON", "Jane Doe", SALT)
    assert out == f"Patient <{person}> seen today."
    assert vault == {person: "Jane Doe"}


def test_pseudonymize_handles_unsorted_entities():
    # entities deliberately out of order; right-to-left edit must still be exact
    text = "Jane Doe on 03/14/2025."
    ents = [_ent(12, 22, "DATE_TIME"), _ent(0, 8)]
    out, vault = pseudonymize(text, ents, SALT)
    person = make_pseudonym("PERSON", "Jane Doe", SALT)
    date = make_pseudonym("DATE_TIME", "03/14/2025", SALT)
    assert out == f"<{person}> on <{date}>."


def test_repeated_value_reuses_pseudonym():
    text = "Jane Doe paged Jane Doe."
    out, vault = pseudonymize(text, [_ent(0, 8), _ent(15, 23)], SALT)
    person = make_pseudonym("PERSON", "Jane Doe", SALT)
    assert out == f"<{person}> paged <{person}>."
    assert vault == {person: "Jane Doe"}


def test_pseudonymize_no_entities_is_identity():
    text = "Nothing to redact here."
    out, vault = pseudonymize(text, [], SALT)
    assert out == text
    assert vault == {}


# --- reidentify (round trip) -------------------------------------------------


def test_round_trip_restores_original():
    text = "Jane Doe called Bob Roe on 03/14/2025."
    ents = [_ent(0, 8), _ent(16, 23), _ent(27, 37, "DATE_TIME")]
    out, vault = pseudonymize(text, ents, SALT)
    assert reidentify(out, vault) == text


def test_reidentify_on_partial_text():
    # a model reply that references only some pseudonyms still resolves them
    text = "Jane Doe and Bob Roe"
    out, vault = pseudonymize(text, [_ent(0, 8), _ent(13, 20)], SALT)
    person = make_pseudonym("PERSON", "Jane Doe", SALT)
    reply = f"Follow up with <{person}> next week."
    assert reidentify(reply, vault) == "Follow up with Jane Doe next week."


def test_reidentify_no_prefix_collision_across_many():
    # longest-first restoration must not let one pseudonym clobber another
    vault = {f"PERSON_{i:010d}": f"name{i}" for i in range(1, 12)}
    keys = list(vault)
    pseudonymized = f"<{keys[0]}> vs <{keys[9]}> vs <{keys[10]}>"
    restored = reidentify(pseudonymized, vault)
    assert restored == "name1 vs name10 vs name11"
