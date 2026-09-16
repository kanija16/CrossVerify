"""
test_consistency.py
--------------------
Unit tests for the cross-modal consistency engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m3_crossmodal.validators import VerhoeffChecksum, Mod11WeightedChecksum, MOD11_WEIGHTS_FAMILY_B
from m3_crossmodal.consistency import (
    normalize_text,
    field_similarity,
    is_qr_readable,
    count_missing_ocr_fields,
    compute_text_qr_match_score,
    compute_consistency_vector,
)


def make_family_a_id(body_11_digits: str) -> str:
    return body_11_digits + VerhoeffChecksum.generate_check_digit(body_11_digits)


def make_family_b_id(body_9_digits: str) -> str:
    return body_9_digits + Mod11WeightedChecksum.generate_check_digit(body_9_digits, MOD11_WEIGHTS_FAMILY_B)


# ---------------------------------------------------------------------------
# qr_readable / unreadable QR semantics
# ---------------------------------------------------------------------------

def test_qr_readable_true_when_fields_present():
    assert is_qr_readable({"name": "Jane Doe"}) is True


def test_qr_readable_false_when_empty_or_none():
    assert is_qr_readable({}) is False
    assert is_qr_readable(None) is False


def test_consistency_vector_qr_unreadable_gives_none_score_not_zero():
    ocr = {"id_number": make_family_a_id("23456789012"), "name": "Jane Doe"}
    result = compute_consistency_vector("family_a", ocr, {})
    assert result["qr_readable"] is False
    assert result["text_qr_match_score"] is None  # not 0.0


# ---------------------------------------------------------------------------
# missing OCR fields
# ---------------------------------------------------------------------------

def test_count_missing_ocr_fields_counts_none_and_empty_strings():
    ocr = {"name": "Jane Doe", "dob": "", "id_number": None, "expiry": "2030-01-01"}
    assert count_missing_ocr_fields(ocr) == 2


def test_count_missing_ocr_fields_empty_dict_is_zero():
    assert count_missing_ocr_fields({}) == 0
    assert count_missing_ocr_fields(None) == 0


def test_count_missing_ocr_fields_ignores_internal_keys():
    ocr = {"name": "Jane Doe", "dob": "", "_raw_ocr_text": ""}
    assert count_missing_ocr_fields(ocr) == 1


def test_consistency_vector_missing_ocr_is_not_treated_as_forgery_flag():
    # OCR mostly missing, QR unreadable too — engine should just report
    # state, not raise or silently mark checksum/format as "forged".
    ocr = {"name": "", "id_number": None}
    result = compute_consistency_vector("family_a", ocr, {})
    assert result["missing_field_count"] == 2
    assert result["format_valid"] is False
    assert result["checksum_valid"] is False
    assert result["qr_readable"] is False


# ---------------------------------------------------------------------------
# OCR <-> QR matching
# ---------------------------------------------------------------------------

def test_normalize_text_basic():
    assert normalize_text("  Jane   Doe  ") == "Jane   Doe"
    assert normalize_text(None) == ""


def test_field_similarity_identical_is_one():
    assert field_similarity("jane doe", "jane doe") == 1.0


def test_field_similarity_different_is_less_than_one():
    assert field_similarity("jane doe", "john doe") < 1.0


def test_compute_text_qr_match_score_common_fields_match():
    ocr = {"name": "Jane Doe", "dob": "1990-01-01"}
    qr = {"name": "Jane Doe", "dob": "1990-01-01"}
    score = compute_text_qr_match_score(ocr, qr)
    assert score == 1.0


def test_compute_text_qr_match_score_partial_mismatch():
    ocr = {"name": "Jane Doe"}
    qr = {"name": "Jane D0e"}  # OCR/QR slightly disagree
    score = compute_text_qr_match_score(ocr, qr)
    assert 0.0 < score < 1.0


def test_compute_text_qr_match_score_none_when_qr_unreadable():
    assert compute_text_qr_match_score({"name": "Jane"}, {}) is None
    assert compute_text_qr_match_score({"name": "Jane"}, None) is None


def test_compute_text_qr_match_score_none_when_ocr_unavailable():
    assert compute_text_qr_match_score({}, {"name": "Jane"}) is None


def test_compute_text_qr_match_score_none_when_no_common_fields():
    assert compute_text_qr_match_score({"a": "1"}, {"b": "2"}) is None


# ---------------------------------------------------------------------------
# Full consistency vector: keys, types, and family dispatch
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "qr_readable",
    "text_qr_match_score",
    "checksum_valid",
    "format_valid",
    "missing_field_count",
}


def test_consistency_vector_has_exactly_the_five_expected_keys():
    ocr = {"id_number": make_family_a_id("23456789012"), "name": "Jane Doe"}
    qr = {"name": "Jane Doe"}
    result = compute_consistency_vector("family_a", ocr, qr)
    assert set(result.keys()) == EXPECTED_KEYS


def test_consistency_vector_types():
    ocr = {"id_number": make_family_a_id("23456789012"), "name": "Jane Doe"}
    qr = {"name": "Jane Doe"}
    result = compute_consistency_vector("family_a", ocr, qr)
    assert isinstance(result["qr_readable"], bool)
    assert isinstance(result["checksum_valid"], bool)
    assert isinstance(result["format_valid"], bool)
    assert isinstance(result["missing_field_count"], int)
    assert result["text_qr_match_score"] is None or isinstance(result["text_qr_match_score"], float)


def test_consistency_vector_family_a_valid_document():
    identifier = make_family_a_id("23456789012")
    ocr = {"id_number": identifier, "name": "Jane Doe"}
    qr = {"name": "Jane Doe"}
    result = compute_consistency_vector("family_a", ocr, qr)
    assert result["format_valid"] is True
    assert result["checksum_valid"] is True
    assert result["qr_readable"] is True
    assert result["text_qr_match_score"] == 1.0


def test_consistency_vector_family_b_valid_document():
    identifier = make_family_b_id("123456780")
    ocr = {"registry_id": identifier, "full_name": "Acme Corp"}
    qr = {"full_name": "Acme Corp"}
    result = compute_consistency_vector("family_b", ocr, qr)
    assert result["format_valid"] is True
    assert result["checksum_valid"] is True
    assert result["qr_readable"] is True
    assert result["text_qr_match_score"] == 1.0


def test_consistency_vector_bad_checksum_does_not_crash_engine():
    identifier = make_family_a_id("23456789012")
    tampered = identifier[:-1] + str((int(identifier[-1]) + 1) % 10)
    ocr = {"id_number": tampered, "name": "Jane Doe"}
    result = compute_consistency_vector("family_a", ocr, {})
    assert result["format_valid"] is True
    assert result["checksum_valid"] is False


def test_consistency_vector_format_invalid_with_valid_checksum():
    # Starts with 0 -> invalid format for Family A, but valid Verhoeff sequence
    identifier0 = make_family_a_id("03456789012")
    ocr = {"id_number": identifier0, "name": "Jane Doe"}
    result = compute_consistency_vector("family_a", ocr, {})
    assert result["format_valid"] is False
    assert result["checksum_valid"] is True
