"""Unit tests for the evaluation harness (pure python, no model needed)."""

from phi_deid.evaluate import evaluate


def test_perfect_prediction():
    text = "John Smith has SSN 536-90-4399 today"
    gold = [(0, 10, "PERSON"), (19, 30, "US_SSN")]
    pred = [(0, 10, "PERSON"), (19, 30, "US_SSN")]
    report = evaluate([(text, gold, pred)])
    assert report.token.recall == 1.0
    assert report.token.precision == 1.0
    assert report.leak_rate == 0.0


def test_missed_entity_counts_as_leak():
    text = "John Smith has SSN 536-90-4399 today"
    gold = [(0, 10, "PERSON"), (19, 30, "US_SSN")]
    pred = [(19, 30, "US_SSN")]  # missed the name
    report = evaluate([(text, gold, pred)])
    assert report.leak_rate > 0.0
    assert report.per_type["PERSON"].recall == 0.0
    assert report.per_type["US_SSN"].recall == 1.0


def test_overshoot_boundary_still_matches():
    # predicted span overshoots the gold value (e.g. "MRN: 12345678")
    text = "MRN: 12345678 end"
    gold = [(5, 13, "MEDICAL_RECORD_NUMBER")]   # just the digits
    pred = [(0, 13, "MEDICAL_RECORD_NUMBER")]   # includes the label
    report = evaluate([(text, gold, pred)])
    assert report.entity.recall == 1.0


def test_false_positive_lowers_precision():
    text = "The weather was fine"
    gold = []
    pred = [(4, 11, "PERSON")]  # spurious
    report = evaluate([(text, gold, pred)])
    assert report.entity.precision == 0.0
