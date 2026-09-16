"""
test_part1.py
-------------
Comprehensive unit and integration test suite for M3 Phase 1 — Part 1.

Tests verify:
1. Binary target mapping (genuine -> 0, forged -> 1)
2. Strict 5D feature ordering
3. Missing consistency key detection
4. Safe None handling for text_qr_match_score
5. Path resolution logic
6. Identity separation / leakage prevention
7. Forbidden feature guard
8. Feature matrix shape and finiteness
9. Target validity
10. Rule-based heuristic logic
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from m3_crossmodal import config, data_interface, features, baseline


def test_final_label_mapping():
    assert data_interface.binary_target("genuine") == 0
    assert data_interface.binary_target("forged") == 1
    assert data_interface.binary_target(0) == 0
    assert data_interface.binary_target(1) == 1
    with pytest.raises(ValueError):
        data_interface.binary_target("tampered")
    with pytest.raises(ValueError):
        data_interface.binary_target("invalid_label")


def test_consistency_feature_ordering():
    dummy_cv = {
        "qr_readable": True,
        "text_qr_match_score": 0.85,
        "checksum_valid": True,
        "format_valid": False,
        "missing_field_count": 2
    }
    feat = features.extract_consistency_features(dummy_cv)
    assert feat.shape == (5,)
    assert feat.dtype == np.float64
    assert feat[0] == 1.0   # qr_readable
    assert feat[1] == 0.85  # text_qr_match_score
    assert feat[2] == 1.0   # checksum_valid
    assert feat[3] == 0.0   # format_valid (False -> 0.0)
    assert feat[4] == 2.0   # missing_field_count


def test_missing_consistency_key_detection():
    incomplete_cv = {
        "qr_readable": True,
        "text_qr_match_score": 0.90,
        "checksum_valid": True,
        # missing format_valid
        "missing_field_count": 0
    }
    with pytest.raises(ValueError, match="missing required key 'format_valid'"):
        features.extract_consistency_features(incomplete_cv)


def test_none_score_handling():
    cv_with_none = {
        "qr_readable": False,
        "text_qr_match_score": None,
        "checksum_valid": True,
        "format_valid": True,
        "missing_field_count": 5
    }
    feat = features.extract_consistency_features(cv_with_none)
    assert feat.shape == (5,)
    assert feat[1] == config.DEFAULT_IMPUTE_TEXT_QR_MATCH_SCORE
    assert np.isfinite(feat).all()


def test_path_resolution():
    rel = "labels/family_a_000000_genuine.json"
    resolved = data_interface.resolve_label_path(rel, root=Path("/custom/root"))
    assert resolved == Path("/custom/root/labels/family_a_000000_genuine.json")

    abs_path = "/absolute/path/to/file.json"
    assert data_interface.resolve_label_path(abs_path) == Path(abs_path)


def test_leakage_guard():
    for forbidden in ["tamper_type", "record_id", "final_label", "cnn_label", "ground_truth_fields"]:
        assert forbidden in config.FORBIDDEN_FEATURE_FIELDS
        assert forbidden not in config.CONSISTENCY_FEATURE_ORDER


def test_feature_matrix_shape_and_finite():
    records = [
        data_interface.M3Record(
            id=f"test_{i}",
            record_id=f"rec_{i}",
            split="train",
            document_family="family_a",
            template_variant="v1",
            tamper_type="genuine" if i == 0 else "forged",
            image_path=Path(f"img_{i}.png"),
            label_path=Path(f"lbl_{i}.json"),
            target=i % 2,
            consistency_vector={
                "qr_readable": True,
                "text_qr_match_score": 0.95 - (i * 0.1),
                "checksum_valid": i == 0,
                "format_valid": True,
                "missing_field_count": i
            },
            ocr_extracted_fields={"name": "Test"},
            qr_decoded_fields={"name": "Test"}
        ) for i in range(4)
    ]
    X, y, groups = features.build_baseline_matrix(records)
    assert X.shape == (4, 5)
    assert y.shape == (4,)
    assert len(groups) == 4
    assert np.isfinite(X).all()
    assert set(y.tolist()) == {0, 1}


def test_rule_based_logic():
    # Row 0: genuine (all valid)
    # Row 1: invalid checksum
    # Row 2: invalid format
    # Row 3: unreadable QR
    X = np.array([
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.9, 0.0, 1.0, 0.0],
        [1.0, 0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 5.0],
    ])
    preds = baseline.rule_based_baseline(X, config.CONSISTENCY_FEATURE_ORDER)
    assert preds[0] == 0  # genuine
    assert preds[1] == 1  # forged
    assert preds[2] == 1  # forged
    assert preds[3] == 1  # forged


def test_real_data_identity_separation():
    # Verify directly on the real Desktop CSVs if accessible
    if config.TRAIN_CSV.exists() and config.VAL_CSV.exists() and config.TEST_CSV.exists():
        import pandas as pd
        df_tr = pd.read_csv(config.TRAIN_CSV)
        df_va = pd.read_csv(config.VAL_CSV)
        df_te = pd.read_csv(config.TEST_CSV)
        ids_tr = set(df_tr["record_id"])
        ids_va = set(df_va["record_id"])
        ids_te = set(df_te["record_id"])
        assert len(ids_tr & ids_va) == 0, "Leakage between train and val"
        assert len(ids_tr & ids_te) == 0, "Leakage between train and test"
        assert len(ids_va & ids_te) == 0, "Leakage between val and test"
        assert len(ids_tr) == 700
        assert len(ids_va) == 150
        assert len(ids_te) == 150
