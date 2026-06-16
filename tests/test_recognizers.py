"""Unit tests for the custom HIPAA recognizers.

These call ``recognizer.analyze`` directly with ``nlp_artifacts=None``, so they
exercise the regex patterns without needing the spaCy model or the full
AnalyzerEngine - they run anywhere, including CI.
"""

from phi_deid.recognizers import (
    ACCOUNT_NUMBER,
    BENEFICIARY_NUMBER,
    DATE_TIME,
    DEVICE_ID,
    MEDICAL_RECORD_NUMBER,
    get_custom_recognizers,
)

RECOGNIZERS = {r.name: r for r in get_custom_recognizers()}


def _detect(name, text, entity):
    rec = RECOGNIZERS[name]
    return rec.analyze(text=text, entities=[entity], nlp_artifacts=None)


def test_mrn_labeled_matches():
    results = _detect("MrnRecognizer", "Patient MRN: 12345678 admitted.", MEDICAL_RECORD_NUMBER)
    assert any(r.entity_type == MEDICAL_RECORD_NUMBER and r.score >= 0.8 for r in results)


def test_account_number_matches():
    results = _detect("AccountNumberRecognizer", "Account No. 0099887766 billed.", ACCOUNT_NUMBER)
    assert any(r.score >= 0.8 for r in results)


def test_beneficiary_matches():
    results = _detect("BeneficiaryNumberRecognizer", "Member ID AB123456 active.", BENEFICIARY_NUMBER)
    assert results, "expected a beneficiary match"


def test_device_id_matches():
    results = _detect("DeviceIdRecognizer", "Implant Serial No: XR-4490AB tracked.", DEVICE_ID)
    assert results, "expected a device id match"


def test_date_formats_match():
    rec = RECOGNIZERS["DateRecognizer"]
    for text in ["seen 03/14/2025", "born 2024-01-09", "admitted March 4, 2025"]:
        results = rec.analyze(text=text, entities=[DATE_TIME], nlp_artifacts=None)
        assert results, f"no date match in: {text}"


def test_no_false_positive_on_plain_prose():
    results = _detect("MrnRecognizer", "The patient felt much better today.", MEDICAL_RECORD_NUMBER)
    # bare-value pattern is low score and there is no 7-10 digit number here
    assert all(r.score < 0.4 for r in results) or not results
