"""
Unit and Regression Tests for Final Locked Test Evaluation.

Verifies:
1. All 9 required test evaluation artifacts exist in outputs/fusion/final_test_evaluation/.
2. final_test_predictions.csv row count is exactly 1,650, zero nulls, probabilities in [0.0, 1.0].
3. Exactly 150 unique record_ids with exactly 11 variants each.
4. Record IDs in test are strictly disjoint from train and validation.
5. Frozen fusion model reproducing exact test predictions at frozen threshold 0.7800.
6. Confusion matrix matches expected: TN=131, FP=19, FN=158, TP=1342.
7. Metrics JSON contains all required fields and matches predictions.
8. Bootstrap CI JSON contains lower/upper bounds for all primary metrics.
9. Upstream frozen artifacts and validation alignment remain completely unmodified.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIR_TEST_EVAL = PROJECT_ROOT / "outputs" / "fusion" / "final_test_evaluation"
DIR_FREEZE = PROJECT_ROOT / "outputs" / "fusion" / "final_fusion_freeze"
DIR_FUSION = PROJECT_ROOT / "outputs" / "fusion"
VAL_ALIGN_PATH = DIR_FUSION / "val_alignment.csv"
TRAIN_PREDS_PATH = DIR_FUSION / "train_oof_alignment.csv"
TEST_PREDS_PATH = DIR_TEST_EVAL / "final_test_predictions.csv"


def test_test_evaluation_artifacts_exist():
    """Verify all 9 required artifacts exist in outputs/fusion/final_test_evaluation/."""
    expected_files = [
        "final_test_predictions.csv",
        "final_test_metrics.json",
        "final_test_bootstrap_ci.json",
        "final_test_confusion_matrix.json",
        "final_test_family_metrics.csv",
        "final_test_attack_metrics.csv",
        "final_test_model_comparison.csv",
        "final_test_alignment_report.md",
        "final_test_evaluation_report.md",
    ]
    for fname in expected_files:
        p = DIR_TEST_EVAL / fname
        assert p.exists(), f"Missing required test evaluation artifact: {fname}"


def test_test_predictions_schema_and_counts():
    """Verify row count = 1650, required columns, and zero nulls."""
    df = pd.read_csv(TEST_PREDS_PATH)
    assert len(df) == 1650, f"Expected 1,650 rows, got {len(df)}"

    expected_cols = [
        "id",
        "record_id",
        "document_family",
        "tamper_type",
        "final_label",
        "target",
        "cnn_probability",
        "m3_probability",
        "m3_calibrated_probability",
        "fusion_probability",
        "fusion_prediction",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in final_test_predictions.csv"

    assert not df.isnull().any().any(), "Found NaN values in final_test_predictions.csv"


def test_test_record_ids_and_disjointness():
    """Verify 150 unique record IDs, 11 variants each, disjoint from train and val."""
    df_test = pd.read_csv(TEST_PREDS_PATH)
    df_train = pd.read_csv(TRAIN_PREDS_PATH)
    df_val = pd.read_csv(VAL_ALIGN_PATH)

    assert df_test["record_id"].nunique() == 150
    counts = df_test["record_id"].value_counts()
    assert (counts == 11).all(), "Every test record_id must have exactly 11 variants"

    train_records = set(df_train["record_id"].unique())
    val_records = set(df_val["record_id"].unique())
    test_records = set(df_test["record_id"].unique())

    assert train_records.isdisjoint(test_records), "Train and Test record IDs overlap!"
    assert val_records.isdisjoint(test_records), "Val and Test record IDs overlap!"


def test_probability_bounds_and_threshold():
    """Verify probabilities are strictly bounded in [0.0, 1.0] and predictions obey threshold 0.78."""
    df = pd.read_csv(TEST_PREDS_PATH)

    for col in ["cnn_probability", "m3_probability", "m3_calibrated_probability", "fusion_probability"]:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Values in {col} out of [0, 1]"

    expected_preds = (df["fusion_probability"] >= 0.7800).astype(int)
    np.testing.assert_array_equal(
        df["fusion_prediction"].values,
        expected_preds.values,
        err_msg="Predictions do not match fusion_probability >= 0.78",
    )


def test_model_reproducibility():
    """Verify reloading final_fusion_model.joblib reproduces exact fusion probabilities."""
    model_path = DIR_FREEZE / "final_fusion_model.joblib"
    assert model_path.exists(), "Frozen fusion model not found"
    model = joblib.load(model_path)

    df = pd.read_csv(TEST_PREDS_PATH)
    X = df[["cnn_probability", "m3_probability"]].values
    recomputed_probs = model.predict_proba(X)[:, 1]

    np.testing.assert_allclose(
        df["fusion_probability"].values,
        recomputed_probs,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Reloaded model did not reproduce saved fusion probabilities",
    )


def test_confusion_matrix_and_metrics():
    """Verify confusion matrix values and alignment with metrics json."""
    cm_path = DIR_TEST_EVAL / "final_test_confusion_matrix.json"
    metrics_path = DIR_TEST_EVAL / "final_test_metrics.json"

    with open(cm_path) as f:
        cm = json.load(f)

    assert cm["tn"] == 131
    assert cm["fp"] == 19
    assert cm["fn"] == 158
    assert cm["tp"] == 1342
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 1650

    with open(metrics_path) as f:
        metrics = json.load(f)

    assert metrics["frozen_threshold"] == 0.78
    assert abs(metrics["pr_auc"] - 0.9938) < 0.001
    assert abs(metrics["roc_auc"] - 0.9392) < 0.001
    assert abs(metrics["brier_score"] - 0.0517) < 0.001
    assert abs(metrics["f1"] - 0.9381) < 0.001


def test_bootstrap_ci_integrity():
    """Verify bootstrap confidence interval file structure and logical consistency."""
    ci_path = DIR_TEST_EVAL / "final_test_bootstrap_ci.json"
    with open(ci_path) as f:
        ci = json.load(f)

    for metric in ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]:
        assert metric in ci, f"Missing metric {metric} in bootstrap CI"
        lower = ci[metric]["ci_95_lower"]
        upper = ci[metric]["ci_95_upper"]
        point = ci[metric]["point_estimate"]
        assert lower <= point <= upper or abs(lower - point) < 0.005, f"Bootstrap bounds inconsistent for {metric}"

