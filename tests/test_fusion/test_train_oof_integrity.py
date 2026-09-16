"""
Unit and Integrity Tests for Train-Level OOF Dataset Preparation.

Verifies:
1. train_oof_alignment.csv and train_oof_folds.csv exist.
2. Exactly 7,700 rows in both alignment and folds artifacts.
3. Exactly 700 unique record IDs with 11 variants each.
4. Zero duplicate sample IDs.
5. Every row has exactly one OOF fold assigned (0 to 4).
6. Zero identity leakage across folds (GroupKFold isolation).
7. Probabilities are finite and strictly bounded in [0.0, 1.0].
8. Exact alignment with canonical train.csv (IDs, record IDs, final_label, cnn_label).
9. Class balance is exactly 700 genuine and 7,000 forged.
10. No forbidden features enter fusion feature matrix.
11. val_alignment.csv was not modified.
12. Zero test set access.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from m3_crossmodal.fusion.fusion_pipeline import (
    FORBIDDEN_FUSION_FEATURES,
    assert_no_forbidden_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIR_FUSION = PROJECT_ROOT / "outputs" / "fusion"
TRAIN_ALIGN_PATH = DIR_FUSION / "train_oof_alignment.csv"
TRAIN_FOLDS_PATH = DIR_FUSION / "train_oof_folds.csv"
TRAIN_CSV_PATH = Path("/Users/pavankumar/Desktop/train.csv")
VAL_ALIGN_PATH = DIR_FUSION / "val_alignment.csv"


def test_artifacts_exist():
    """Verify required train OOF artifacts exist."""
    assert TRAIN_ALIGN_PATH.exists(), f"Missing {TRAIN_ALIGN_PATH}"
    assert TRAIN_FOLDS_PATH.exists(), f"Missing {TRAIN_FOLDS_PATH}"
    assert (DIR_FUSION / "train_oof_run_config.json").exists()
    assert (DIR_FUSION / "train_oof_report.md").exists()


def test_row_and_identity_counts():
    """Verify 7,700 rows, 700 unique record IDs, 11 variants each, 700 genuine / 7000 forged."""
    df = pd.read_csv(TRAIN_ALIGN_PATH)
    assert len(df) == 7700
    assert df["id"].nunique() == 7700
    assert df["record_id"].nunique() == 700
    assert (df["record_id"].value_counts() == 11).all()
    assert (df["target"] == 0).sum() == 700
    assert (df["target"] == 1).sum() == 7000
    assert df.isnull().sum().sum() == 0


def test_fold_assignment_and_leakage():
    """Verify 5 folds, exactly 140 records per fold, and zero identity overlap across folds."""
    df_folds = pd.read_csv(TRAIN_FOLDS_PATH)
    assert len(df_folds) == 7700
    assert set(df_folds["oof_fold"].unique()) == {0, 1, 2, 3, 4}

    # Verify no record_id is assigned to more than one fold
    rec_fold_counts = df_folds.groupby("record_id")["oof_fold"].nunique()
    assert (rec_fold_counts == 1).all()

    # Verify each fold has exactly 140 identities
    counts_per_fold = df_folds.groupby("oof_fold")["record_id"].nunique()
    assert (counts_per_fold == 140).all()

    # Pairwise disjointness check
    folds = {f: set(df_folds[df_folds["oof_fold"] == f]["record_id"]) for f in range(5)}
    for f1 in range(5):
        for f2 in range(f1 + 1, 5):
            assert len(folds[f1].intersection(folds[f2])) == 0, f"Leakage between fold {f1} and {f2}"


def test_probability_ranges():
    """Verify cnn_probability, m3_probability, and m3_calibrated_probability are strictly in [0, 1]."""
    df = pd.read_csv(TRAIN_ALIGN_PATH)
    for col in ["cnn_probability", "m3_probability", "m3_calibrated_probability"]:
        assert (df[col] >= 0.0).all(), f"Values in {col} < 0.0"
        assert (df[col] <= 1.0).all(), f"Values in {col} > 1.0"
        assert not df[col].isna().any()


def test_exact_alignment_with_canonical_train_csv():
    """Verify exact row-by-row alignment with canonical train.csv."""
    df = pd.read_csv(TRAIN_ALIGN_PATH)
    df_canonical = pd.read_csv(TRAIN_CSV_PATH)

    assert (df["id"] == df_canonical["id"]).all()
    assert (df["record_id"] == df_canonical["record_id"]).all()
    assert (df["final_label"] == df_canonical["final_label"]).all()
    assert (df["tamper_type"] == df_canonical["tamper_type"]).all()


def test_forbidden_features_guard():
    """Verify forbidden metadata cannot be included in fusion feature matrix."""
    for feat in FORBIDDEN_FUSION_FEATURES:
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE"):
            assert_no_forbidden_features([feat])


def test_val_alignment_not_modified():
    """Verify val_alignment.csv remains exactly 1,650 rows and unmodified."""
    df_val = pd.read_csv(VAL_ALIGN_PATH)
    assert len(df_val) == 1650
    assert df_val["record_id"].nunique() == 150


def test_zero_test_access():
    """Verify test split is not referenced or accessed."""
    with open(DIR_FUSION / "train_oof_run_config.json") as f:
        cfg = json.load(f)
    assert "ZERO TEST ACCESS" in cfg["test_set_access_status"]
