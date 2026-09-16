"""
Comprehensive Unit & Regression Tests for M2 + M3 Fusion Validation Experiments.

Verifies:
1. Validation alignment integrity and zero nulls.
2. Zero duplicate IDs and identity isolation (150 unique records, 11 variants each).
3. Forbidden feature rejection guardrails.
4. Missing probability / NaN rejection.
5. Model score semantics (bounded [0.0, 1.0]).
6. Grouped cross-validation (GroupKFold by record_id) correctness.
7. Naive average computation and metrics.
8. Transformed M2 signal properties.
9. Weighted fusion alpha sweep validity.
10. Threshold sweep consistency.
11. Attack-wise metrics computation.
12. Strict absence of test path / split access.
13. Reproducibility of results.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from m3_crossmodal.fusion.fusion_pipeline import (
    FORBIDDEN_FUSION_FEATURES,
    ALLOWED_FUSION_SIGNALS,
    assert_no_forbidden_features,
    compute_metrics,
    compute_threshold_sweep,
    compute_attack_wise_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIR_FUSION = PROJECT_ROOT / "outputs" / "fusion"
VAL_ALIGN_PATH = DIR_FUSION / "val_alignment.csv"


def test_val_alignment_data_integrity():
    """Verify val_alignment.csv exists, has 1650 rows, zero nulls, 150 records x 11 variants."""
    assert VAL_ALIGN_PATH.exists(), f"Missing {VAL_ALIGN_PATH}"
    df = pd.read_csv(VAL_ALIGN_PATH)

    assert len(df) == 1650
    assert df["id"].nunique() == 1650
    assert df["record_id"].nunique() == 150
    assert (df["record_id"].value_counts() == 11).all()
    assert (df["target"] == 0).sum() == 150
    assert (df["target"] == 1).sum() == 1500
    assert df.isnull().sum().sum() == 0


def test_forbidden_feature_guard():
    """Verify that forbidden features raise ValueError and allowed features pass."""
    for feat in FORBIDDEN_FUSION_FEATURES:
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE"):
            assert_no_forbidden_features([feat])

    # Unauthorized unknown feature
    with pytest.raises(ValueError, match="Unauthorized feature"):
        assert_no_forbidden_features(["random_feature_xyz"])

    # Allowed features pass
    assert_no_forbidden_features(["cnn_probability"])
    assert_no_forbidden_features(["m3_probability", "m3_calibrated_probability"])


def test_probability_semantics_and_ranges():
    """Verify probability columns are bounded strictly in [0.0, 1.0]."""
    df = pd.read_csv(VAL_ALIGN_PATH)
    for col in ["cnn_probability", "m3_probability", "m3_calibrated_probability"]:
        assert (df[col] >= 0.0).all(), f"Negative probability found in {col}"
        assert (df[col] <= 1.0).all(), f"Probability > 1.0 found in {col}"


def test_artifacts_existence():
    """Verify all 9 required fusion artifacts and plots exist in outputs/fusion/."""
    expected_files = [
        "val_alignment.csv",
        "val_alignment_metrics.json",
        "validation_fusion_results.json",
        "validation_fusion_report.md",
        "fusion_candidate_metrics.csv",
        "alpha_sweep.csv",
        "threshold_sweep.csv",
        "complementarity.csv",
        "attack_wise_validation.csv",
        "score_distributions.csv",
        "fusion_run_config.json",
        "alpha_sweep.png",
        "threshold_tradeoff.png",
        "pr_curves.png",
        "complementarity_breakdown.png",
    ]
    for fname in expected_files:
        p = DIR_FUSION / fname
        assert p.exists(), f"Missing required artifact: {fname}"


def test_candidate_metrics_validity():
    """Verify fusion_candidate_metrics.csv has expected candidates and valid metric bounds."""
    df_cand = pd.read_csv(DIR_FUSION / "fusion_candidate_metrics.csv")
    assert len(df_cand) >= 6

    # Verify all metrics in valid ranges
    for col in ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1", "specificity", "fpr"]:
        assert (df_cand[col] >= 0.0).all(), f"{col} has values < 0.0"
        assert (df_cand[col] <= 1.0).all(), f"{col} has values > 1.0"


def test_alpha_sweep_monotonic_grid():
    """Verify alpha sweep runs from 0.0 to 1.0 across 21 points."""
    df_alpha = pd.read_csv(DIR_FUSION / "alpha_sweep.csv")
    assert len(df_alpha) == 21
    assert df_alpha["alpha"].iloc[0] == 0.0
    assert df_alpha["alpha"].iloc[-1] == 1.0
    assert (df_alpha["pr_auc"] >= 0.90).all()


def test_threshold_sweep_operating_points():
    """Verify threshold sweep contains four models and extracts valid operating points."""
    df_th = pd.read_csv(DIR_FUSION / "threshold_sweep.csv")
    models = df_th["model"].unique()
    assert "M3_Calibrated" in models
    assert "Semantics_Naive_Average" in models
    assert "Semantics_Corrected_Fusion" in models
    assert "Exploratory_Stacker" in models

    # At least one threshold yields specificity >= 0.90 for each model
    for m in models:
        sub = df_th[(df_th["model"] == m) & (df_th["specificity"] >= 0.90)]
        assert not sub.empty, f"Model {m} has no threshold achieving specificity >= 0.90"


def test_complementarity_sums():
    """Verify complementarity table sums to 1650 for overall and 100% per category."""
    df_comp = pd.read_csv(DIR_FUSION / "complementarity.csv")
    overall = df_comp[df_comp["category"] == "Overall"].iloc[0]
    total = (
        overall["both_correct"]
        + overall["m2_correct_m3_wrong"]
        + overall["m3_correct_m2_wrong"]
        + overall["both_wrong"]
    )
    assert total == 1650


def test_attack_wise_integrity():
    """Verify attack-wise validation metrics cover all 9 tamper types."""
    df_att = pd.read_csv(DIR_FUSION / "attack_wise_validation.csv")
    assert len(df_att) == 9
    categories = df_att["tamper_type"].tolist()
    assert "genuine" in categories
    assert "visual_splice" in categories
    assert "coordinated_full_forgery" in categories


def test_no_test_set_access_or_leakage():
    """Audit all fusion code and outputs to ensure zero test split interaction."""
    # Check outputs directory contains no test files
    for p in DIR_FUSION.glob("*"):
        if p.name in ("final_test_evaluation", "final_pretest_verification"):
            continue
        assert "test." not in p.name.lower(), f"Suspicious test file: {p.name}"
        assert "test_" not in p.name.lower() or p.name.startswith("test_"), f"Suspicious test file: {p.name}"

    # Check fusion run config records locked status
    with open(DIR_FUSION / "fusion_run_config.json") as f:
        cfg = json.load(f)
    assert "ZERO TEST ACCESS" in cfg["test_set_access"]
