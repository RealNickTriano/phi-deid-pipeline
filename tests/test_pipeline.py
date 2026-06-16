"""Unit tests for the [reconcile] stage.

These build ``RecognizerResult`` objects directly, so they exercise the overlap
resolution logic without the analyzer, spaCy, or any model - they run anywhere.
"""

from presidio_analyzer import RecognizerResult

from phi_deid.pipeline import reconcile


def _types(results):
    return sorted((r.entity_type, r.start, r.end) for r in results)


def test_overlapping_spans_keep_highest_score():
    # an SSN that also matches the phone pattern on the same characters
    ssn = RecognizerResult("US_SSN", 0, 11, 0.85)
    phone = RecognizerResult("PHONE_NUMBER", 0, 11, 0.40)
    kept = reconcile([phone, ssn])
    assert _types(kept) == [("US_SSN", 0, 11)]


def test_non_overlapping_spans_all_kept():
    a = RecognizerResult("DATE_TIME", 0, 10, 0.6)
    b = RecognizerResult("EMAIL_ADDRESS", 20, 40, 1.0)
    kept = reconcile([a, b])
    assert _types(kept) == [("DATE_TIME", 0, 10), ("EMAIL_ADDRESS", 20, 40)]


def test_partial_overlap_drops_weaker():
    # email vs the URL that matches only its domain tail
    email = RecognizerResult("EMAIL_ADDRESS", 0, 25, 1.0)
    url = RecognizerResult("URL", 14, 25, 0.5)
    kept = reconcile([email, url])
    assert _types(kept) == [("EMAIL_ADDRESS", 0, 25)]


def test_tie_on_score_prefers_longer_span():
    short = RecognizerResult("ACCOUNT_NUMBER", 5, 13, 0.85)
    longr = RecognizerResult("MEDICAL_RECORD_NUMBER", 0, 13, 0.85)
    kept = reconcile([short, longr])
    assert _types(kept) == [("MEDICAL_RECORD_NUMBER", 0, 13)]


def test_results_returned_in_document_order():
    later = RecognizerResult("DATE_TIME", 30, 40, 0.6)
    earlier = RecognizerResult("US_SSN", 0, 11, 0.85)
    kept = reconcile([later, earlier])
    assert [r.start for r in kept] == [0, 30]
