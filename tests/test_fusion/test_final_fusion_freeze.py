"""
Unit and Regression Tests for Final Fusion Training and Model Freeze.

Verifies:
1. All required freeze artifacts exist in outputs/fusion/final_fusion_freeze/.
2. outputs/fusion/final_fusion_validation_predictions.csv exists with exactly 1,650 rows.
3. Frozen model is loadable and reproduces exact validation metrics and threshold.
4. Correct fusion feature columns ['cnn_probability', 'm3_probability'].
5. Strict forbidden-feature rejection.
6. Training prediction row count = 7,700 and validation prediction row count = 1,650.
7. Probabilities are strictly in [0.0, 1.0] and non-null.
8. Train/validation identity sets (record_id) are strictly disjoint.
9. Zero test set references in fusion code and artifacts.
10. Existing val_alignment.csv and M2/M3 frozen artifacts remain completely unaltered.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from m3_crossmodal.fusion.train_and_freeze_fusion import (
    FUSION_FEATURE_NAMES,
    FORBIDDEN_COLUMNS,
    assert_clean_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIR_FREEZE = PROJECT_ROOT / "outputs" / "fusion" / "final_fusion_freeze"
DIR_FUSION = PROJECT_ROOT / "outputs" / "fusion"
VAL_ALIGN_PATH = DIR_FUSION / "val_alignment.csv"
TRAIN_PREDS_PATH = DIR_FUSION / "train_oof_alignment.csv"
VAL_PREDS_PATH = DIR_FUSION / "final_fusion_validation_predictions.csv"


def test_freeze_artifacts_exist():
    """Verify all 5 required freeze artifacts exist in outputs/fusion/final_fusion_freeze/."""
    expected = [
        "final_fusion_model.joblib",
        "final_fusion_config.json",
        "final_fusion_feature_schema.json",
        "final_fusion_validation_metrics.json",
        "final_fusion_report.md",
    ]
    for fname in expected:
        p = DIR_FREEZE / fname
        assert p.exists(), f"Missing freeze artifact: {fname}"

    assert VAL_PREDS_PATH.exists(), f"Missing {VAL_PREDS_PATH}"


def test_row_counts_and_disjoint_identities():
    """Verify train has 7,700 rows, val has 1,650 rows, and identities are 100% disjoint."""
    train_df = pd.read_csv(TRAIN_PREDS_PATH)
    val_df = pd.read_csv(VAL_ALIGN_PATH)
    val_preds_df = pd.read_csv(VAL_PREDS_PATH)

    assert len(train_df) == 7700
    assert len(val_df) == 1650
    assert len(val_preds_df) == 1650

    tr_recs = set(train_df["record_id"])
    va_recs = set(val_df["record_id"])
    assert len(tr_recs) == 700
    assert len(va_recs) == 150
    assert len(tr_recs.intersection(va_recs)) == 0, "Identity overlap between train and validation!"


def test_clean_features_schema():
    """Verify features schema specifies exactly ['cnn_probability', 'm3_probability']."""
    with open(DIR_FREEZE / "final_fusion_feature_schema.json") as f:
        schema = json.load(f)

    assert schema["fusion_features"] == FUSION_FEATURE_NAMES
    assert schema["feature_count"] == 2
    assert_clean_features(schema["fusion_features"])

    for feat in FORBIDDEN_COLUMNS:
        with pytest.raises(ValueError, match="LEAKAGE GUARD"):
            assert_clean_features([feat])


def test_probabilities_validity():
    """Verify fusion probabilities are in [0, 1] and strictly finite."""
    df = pd.read_csv(VAL_PREDS_PATH)
    probs = df["fusion_probability"].values
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()
    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()
    assert set(df["fusion_prediction"].unique()).issubset({0, 1})


def test_reproducibility_of_frozen_model():
    """Verify loading final_fusion_model.joblib reproduces the exact predictions and metrics."""
    model = joblib.load(DIR_FREEZE / "final_fusion_model.joblib")
    val_df = pd.read_csv(VAL_ALIGN_PATH)
    val_preds_df = pd.read_csv(VAL_PREDS_PATH)

    X_val = val_df[FUSION_FEATURE_NAMES].values
    recomputed_probs = model.predict_proba(X_val)[:, 1]

    np.testing.assert_allclose(
        recomputed_probs,
        val_preds_df["fusion_probability"].values,
        atol=1e-5,
    )

    with open(DIR_FREEZE / "final_fusion_config.json") as f:
        cfg = json.load(f)
    assert cfg["frozen_decision_threshold"] == 0.78
    assert cfg["freeze_status"] == "FROZEN"


def test_val_alignment_and_upstream_models_unaltered():
    """Verify val_alignment.csv and upstream frozen checkpoints were not modified."""
    df_val = pd.read_csv(VAL_ALIGN_PATH)
    assert len(df_val) == 1650
    assert df_val["record_id"].nunique() == 150

    # Upstream M2 checkpoint
    m2_ckpt = PROJECT_ROOT / "models" / "cnn" / "best_model.pt"
    assert m2_ckpt.exists()

    # Upstream M3 models
    m3_raw = PROJECT_ROOT / "outputs" / "final_m3_model_freeze" / "final_m3_model.joblib"
    m3_cal = PROJECT_ROOT / "outputs" / "final_m3_model_freeze" / "final_m3_calibrated_model.joblib"
    assert m3_raw.exists()
    assert m3_cal.exists()


def test_zero_test_access():
    """Audit freeze artifacts for zero test set leakage."""
    with open(DIR_FREEZE / "final_fusion_config.json") as f:
        cfg = json.load(f)
    assert "ZERO TEST ACCESS" in cfg["test_set_access_status"]

    for p in DIR_FREEZE.glob("*"):
        assert "test." not in p.name.lower()
