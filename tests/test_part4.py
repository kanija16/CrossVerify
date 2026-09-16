"""
test_part4.py
-------------
Unit tests for Part 4 feature engineering, schema integrity, missing-data preservation,
and leakage guards.
"""

import math
import numpy as np
import pytest

from m3_crossmodal.part4_features import (
    FORBIDDEN_FEATURE_FIELDS,
    PART4_FEATURE_ORDER_NO_FAMILY,
    PART4_FEATURE_ORDER_WITH_FAMILY,
    assert_no_forbidden_features,
    compute_field_similarities,
    extract_part4_features,
    get_feature_vector,
)


def test_forbidden_feature_rejection():
    """Asserts that forbidden fields in columns immediately trigger a ValueError."""
    for forbidden in FORBIDDEN_FEATURE_FIELDS:
        with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
            assert_no_forbidden_features(["qr_readable", forbidden, "format_valid"])

    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        assert_no_forbidden_features(["some_image_path"])

    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        assert_no_forbidden_features(["final_label"])


def test_clean_features_pass_leakage_guard():
    """Verifies that all 16 canonical features pass the forbidden-field audit."""
    assert_no_forbidden_features(PART4_FEATURE_ORDER_WITH_FAMILY)
    assert_no_forbidden_features(PART4_FEATURE_ORDER_NO_FAMILY)


def test_extract_part4_features_family_a():
    """Tests feature extraction for a typical valid Family A document."""
    ocr_fields = {
        "name": "ROHIT VERMA",
        "dob": "1990-05-15",
        "gender": "M",
        "id_number": "212345678904",
        "address": "123 MAIN STREET",
    }
    qr_fields = {
        "name": "ROHIT VERMA",
        "dob": "1990-05-15",
        "gender": "M",
        "id_number": "212345678904",
        "address": "123 MAIN STREET",
    }
    consistency_vec = {
        "qr_readable": True,
        "text_qr_match_score": 1.0,
        "checksum_valid": True,
        "format_valid": True,
        "missing_field_count": 0,
    }

    feats = extract_part4_features(ocr_fields, qr_fields, consistency_vec, "family_a")
    assert feats["qr_readable"] == 1.0
    assert feats["text_qr_match_score"] == 1.0
    assert feats["checksum_valid"] == 1.0
    assert feats["format_valid"] == 1.0
    assert feats["missing_field_count"] == 0.0
    assert feats["ocr_field_presence_count"] == 5.0
    assert feats["ocr_field_presence_rate"] == 1.0
    assert feats["qr_field_count"] == 5.0
    assert feats["ocr_qr_key_overlap_count"] == 5.0
    assert feats["checksum_computed"] == 1.0
    assert feats["document_family"] == 0.0  # family_a

    # All 5 per-field similarities should be 1.0
    for i in range(5):
        assert feats[f"field_similarity__{i}"] == 1.0


def test_extract_part4_features_family_b():
    """Tests feature extraction for a typical valid Family B document."""
    ocr_fields = {
        "full_name": "ACME LOGISTICS INC",
        "registry_id": "1234567891",
        "entity_type": "CORPORATION",
        "jurisdiction_code": "DELAWARE",
        "registration_date": "2020-01-10",
    }
    qr_fields = {
        "full_name": "ACME LOGISTICS INC",
        "registry_id": "1234567891",
        "entity_type": "CORPORATION",
        "jurisdiction_code": "DELAWARE",
        "registration_date": "2020-01-10",
    }
    consistency_vec = {
        "qr_readable": True,
        "text_qr_match_score": 1.0,
        "checksum_valid": True,
        "format_valid": True,
        "missing_field_count": 0,
    }

    feats = extract_part4_features(ocr_fields, qr_fields, consistency_vec, "family_b")
    assert feats["document_family"] == 1.0  # family_b
    assert feats["checksum_computed"] == 1.0
    for i in range(5):
        assert feats[f"field_similarity__{i}"] == 1.0


def test_missing_qr_preserves_nan_not_zero():
    """
    CRITICAL TEST: When QR is unreadable, text_qr_match_score and per-field similarities
    must be NaN, NEVER 0.0 (non-comparable != forgery mismatch).
    """
    ocr_fields = {
        "name": "ROHIT VERMA",
        "dob": "1990-05-15",
        "gender": "M",
        "id_number": "212345678904",
        "address": "123 MAIN STREET",
    }
    qr_fields = {}  # unreadable QR
    consistency_vec = {
        "qr_readable": False,
        "text_qr_match_score": None,  # unreadable
        "checksum_valid": True,
        "format_valid": True,
        "missing_field_count": 0,
    }

    feats = extract_part4_features(ocr_fields, qr_fields, consistency_vec, "family_a")
    assert feats["qr_readable"] == 0.0
    assert np.isnan(feats["text_qr_match_score"]), "text_qr_match_score must be NaN when QR is unreadable"
    assert feats["qr_field_count"] == 0.0
    assert feats["ocr_qr_key_overlap_count"] == 0.0

    for i in range(5):
        assert np.isnan(feats[f"field_similarity__{i}"]), f"field_similarity__{i} must be NaN when QR is unreadable"


def test_partial_field_missing_preserves_nan():
    """Tests that missing individual fields yield NaN for that slot only."""
    ocr_fields = {
        "name": "ROHIT VERMA",
        "dob": "1990-05-15",
        "gender": None,  # missing
        "id_number": "212345678904",
        "address": "123 MAIN STREET",
    }
    qr_fields = {
        "name": "ROHIT VERMA",
        "dob": "1990-05-15",
        "gender": "M",
        "id_number": "212345678904",
        "address": "123 MAIN STREET",
    }
    consistency_vec = {
        "qr_readable": True,
        "text_qr_match_score": 1.0,
        "checksum_valid": True,
        "format_valid": True,
        "missing_field_count": 1,
    }

    feats = extract_part4_features(ocr_fields, qr_fields, consistency_vec, "family_a")
    assert feats["field_similarity__0"] == 1.0  # name
    assert feats["field_similarity__1"] == 1.0  # dob
    assert np.isnan(feats["field_similarity__2"]), "Slot 2 (gender) must be NaN when OCR is None"
    assert feats["field_similarity__3"] == 1.0  # id_number
    assert feats["field_similarity__4"] == 1.0  # address
    assert feats["ocr_field_presence_count"] == 4.0
    assert feats["ocr_field_presence_rate"] == 0.8
    assert feats["ocr_qr_key_overlap_count"] == 4.0


def test_checksum_computed_flag():
    """Tests that checksum_computed is 0 when identifier is absent or blank."""
    ocr_no_id = {"name": "TEST USER", "id_number": None}
    feats_no_id = extract_part4_features(
        ocr_no_id, {}, {"qr_readable": False, "checksum_valid": False, "format_valid": False}, "family_a"
    )
    assert feats_no_id["checksum_computed"] == 0.0

    ocr_blank_id = {"name": "TEST USER", "id_number": "   "}
    feats_blank = extract_part4_features(
        ocr_blank_id, {}, {"qr_readable": False, "checksum_valid": False, "format_valid": False}, "family_a"
    )
    assert feats_blank["checksum_computed"] == 0.0

    ocr_valid_id = {"name": "TEST USER", "id_number": "212345678904"}
    feats_valid = extract_part4_features(
        ocr_valid_id, {}, {"qr_readable": False, "checksum_valid": True, "format_valid": True}, "family_a"
    )
    assert feats_valid["checksum_computed"] == 1.0


def test_get_feature_vector_shapes():
    """Tests array conversion with and without document_family."""
    feat_dict = {k: 1.0 for k in PART4_FEATURE_ORDER_WITH_FAMILY}
    vec_no_fam = get_feature_vector(feat_dict, include_family=False)
    vec_with_fam = get_feature_vector(feat_dict, include_family=True)

    assert len(vec_no_fam) == 15
    assert len(vec_with_fam) == 16
    assert isinstance(vec_no_fam, np.ndarray)
    assert isinstance(vec_with_fam, np.ndarray)


def test_unknown_family_raises():
    """Verifies that an unknown document family raises ValueError."""
    with pytest.raises(ValueError, match="Unknown document_family"):
        extract_part4_features({}, {}, {}, "family_c")


def test_target_isolation():
    """Verifies that load_cached_features never includes final_label or tamper_type in X."""
    from m3_crossmodal.part4_cache import load_cached_features
    X, y, meta = load_cached_features("val", include_family=False)
    assert "final_label" not in X.columns
    assert "tamper_type" not in X.columns
    assert "record_id" not in X.columns
    assert "id" not in X.columns
    assert len(X) == len(y) == len(meta)


def test_identity_overlap_in_cache():
    """Verifies zero identity overlap across the cached split feature tables."""
    import pandas as pd
    tr = pd.read_csv("outputs/features/train_features.csv")
    va = pd.read_csv("outputs/features/val_features.csv")
    te = pd.read_csv("outputs/features/test_features.csv")

    tr_ids = set(tr["record_id"].astype(str))
    va_ids = set(va["record_id"].astype(str))
    te_ids = set(te["record_id"].astype(str))

    assert len(tr_ids.intersection(va_ids)) == 0
    assert len(tr_ids.intersection(te_ids)) == 0
    assert len(va_ids.intersection(te_ids)) == 0


def test_model_reproducibility():
    """Verifies that models trained with the fixed seed produce deterministic predictions."""
    from m3_crossmodal.part4_models import create_random_forest_pipeline, RANDOM_SEED
    from m3_crossmodal.part4_cache import load_cached_features
    X_val, y_val, _ = load_cached_features("val", include_family=False)

    m1 = create_random_forest_pipeline(random_state=RANDOM_SEED)
    m2 = create_random_forest_pipeline(random_state=RANDOM_SEED)

    m1.fit(X_val.iloc[:100], y_val[:100])
    m2.fit(X_val.iloc[:100], y_val[:100])

    p1 = m1.predict_proba(X_val.iloc[100:150])
    p2 = m2.predict_proba(X_val.iloc[100:150])

    np.testing.assert_allclose(p1, p2, atol=1e-7)

