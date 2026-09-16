"""
test_forensic_pipeline.py
-------------------------
Unit and Regression Tests for Stage 1 Image Forensic Pipeline.

Verifies:
1. Deterministic feature extraction and finite outputs.
2. Strict rejection of forbidden metadata/leakage columns.
3. Feature schema integrity (16 canonical forensic features).
4. Train/Val feature cache integrity (7,700 and 1,650 rows, 0 NaNs).
5. Train/Val identity disjointness.
6. Zero test split access (no test artifacts in outputs/forensics/).
7. Gating report verification.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from m3_crossmodal.forensics.region_features import (
    FORENSIC_FEATURE_NAMES,
    FORBIDDEN_FORENSIC_COLUMNS,
    assert_clean_forensic_features,
    extract_region_anomaly_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIR_FORENSICS = PROJECT_ROOT / "outputs" / "forensics"


def test_deterministic_feature_extraction():
    """Verify feature extractor produces identical values given identical pixel array."""
    np.random.seed(42)
    sample_img = np.random.randint(0, 256, (640, 1000, 3), dtype=np.uint8)

    feats1 = extract_region_anomaly_features(sample_img, "family_a")
    feats2 = extract_region_anomaly_features(sample_img, "family_a")

    assert len(feats1) == 16
    for k in FORENSIC_FEATURE_NAMES:
        assert k in feats1
        assert np.isfinite(feats1[k])
        assert feats1[k] == feats2[k]


def test_forbidden_column_rejection():
    """Verify assert_clean_forensic_features raises ValueError on leakage fields."""
    for col in FORBIDDEN_FORENSIC_COLUMNS:
        with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
            assert_clean_forensic_features(["forensic_reg_lap_std_mean", col])

    for col in ["image_path", "test_path", "cnn_label", "final_label"]:
        with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
            assert_clean_forensic_features([col])


def test_gating_report_exists_and_verified():
    """Verify gating diagnostic report was generated and confirms no generator shortcuts."""
    report_path = DIR_FORENSICS / "generator_artifact_gating_report.md"
    assert report_path.exists(), "Missing generator_artifact_gating_report.md"
    content = report_path.read_text()
    assert "TRAIN ONLY" in content
    assert "No Generator Metadata Leakage" in content


def test_train_val_forensic_caches_exist():
    """Verify train and val caches exist with exact row counts and no missing values."""
    train_csv = DIR_FORENSICS / "train_forensic_features.csv"
    val_csv = DIR_FORENSICS / "val_forensic_features.csv"

    assert train_csv.exists(), "Missing train_forensic_features.csv"
    assert val_csv.exists(), "Missing val_forensic_features.csv"

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)

    assert len(df_tr) == 7700, f"Expected 7,700 train rows, got {len(df_tr)}"
    assert len(df_va) == 1650, f"Expected 1,650 val rows, got {len(df_va)}"

    for col in FORENSIC_FEATURE_NAMES:
        assert col in df_tr.columns
        assert col in df_va.columns
        assert not df_tr[col].isnull().any()
        assert not df_va[col].isnull().any()


def test_identity_disjointness_train_val():
    """Verify record_ids between train and val forensic caches are 100% disjoint."""
    train_csv = DIR_FORENSICS / "train_forensic_features.csv"
    val_csv = DIR_FORENSICS / "val_forensic_features.csv"

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)

    tr_records = set(df_tr["record_id"].unique())
    va_records = set(df_va["record_id"].unique())

    assert len(tr_records) == 700
    assert len(va_records) == 150
    assert tr_records.isdisjoint(va_records), "Train and Val identities must be disjoint!"


def test_zero_test_split_access_in_forensics():
    """Verify outputs/forensics/ contains zero test split files or labels."""
    for p in DIR_FORENSICS.glob("*"):
        assert "test" not in p.name.lower(), f"Suspicious test file in forensics: {p.name}"
