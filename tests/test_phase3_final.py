"""
Unit tests for Phase 3 Final Model Freeze and Test Evaluation.
Verifies:
1. Final model freeze manifest contains 'model_status': 'FROZEN'.
2. Final threshold artifact exists and specifies tau=0.2500.
3. Feature schema matches exactly 15 canonical features.
4. No forbidden columns exist in the frozen feature schema.
5. Saved model joblib artifacts exist and are loadable.
6. Final test metrics and confusion matrix artifacts exist.
7. Verification that test evaluation occurred with frozen parameters.
"""

import json
from pathlib import Path
import joblib
import pytest

from m3_crossmodal.part4_features import (
    FORBIDDEN_FEATURE_FIELDS,
    PART4_FEATURE_ORDER_NO_FAMILY,
    assert_no_forbidden_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIR_FREEZE = PROJECT_ROOT / "outputs" / "final_m3_model_freeze"
DIR_FINAL = PROJECT_ROOT / "outputs" / "final_m3"


def test_freeze_manifest_integrity():
    """Verify freeze manifest fields, status, and contract."""
    manifest_p = DIR_FREEZE / "final_freeze_manifest.json"
    assert manifest_p.exists(), f"Missing freeze manifest: {manifest_p}"

    with open(manifest_p) as f:
        data = json.load(f)

    assert data["model_status"] == "FROZEN"
    assert "freeze_timestamp" in data
    assert data["decision_threshold"] == 0.2500
    assert len(data["features"]) == 15
    assert data["features"] == PART4_FEATURE_ORDER_NO_FAMILY


def test_feature_schema_artifact():
    """Verify feature schema contains exactly 15 features and zero forbidden columns."""
    schema_p = DIR_FREEZE / "final_feature_schema.json"
    assert schema_p.exists()

    with open(schema_p) as f:
        schema = json.load(f)

    assert schema["total_features"] == 15
    feats = schema["feature_order"]
    assert_no_forbidden_features(feats)
    for col in feats:
        assert col not in FORBIDDEN_FEATURE_FIELDS


def test_threshold_artifact():
    """Verify threshold artifact specifies 0.2500 with operational metrics."""
    th_p = DIR_FREEZE / "final_threshold.json"
    assert th_p.exists()

    with open(th_p) as f:
        th_data = json.load(f)

    assert th_data["frozen_decision_threshold"] == 0.2500
    assert th_data["validation_specificity"] >= 0.89


def test_model_joblib_loadable():
    """Verify that both raw and calibrated frozen model joblibs load cleanly."""
    model_p = DIR_FREEZE / "final_m3_model.joblib"
    cal_p = DIR_FREEZE / "final_m3_calibrated_model.joblib"
    assert model_p.exists()
    assert cal_p.exists()

    model = joblib.load(model_p)
    cal_model = joblib.load(cal_p)

    assert hasattr(model, "predict_proba")
    assert hasattr(cal_model, "predict_proba")


def test_final_test_metrics_exist():
    """Verify final test metrics and confusion matrix artifacts exist."""
    test_metrics_p = DIR_FINAL / "final_test_metrics.json"
    test_cm_p = DIR_FINAL / "final_test_confusion_matrix.json"
    report_p = DIR_FINAL / "FINAL_M3_REPORT.md"

    assert test_metrics_p.exists()
    assert test_cm_p.exists()
    assert report_p.exists()

    with open(test_metrics_p) as f:
        metrics = json.load(f)

    assert metrics["split"] == "test"
    assert metrics["samples"] == 1650
    assert metrics["threshold"] == 0.2500
    assert metrics["pr_auc"] >= 0.9700
    assert metrics["roc_auc"] >= 0.8400
    assert metrics["specificity"] >= 0.9000
    assert metrics["precision"] >= 0.9500
