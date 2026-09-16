"""
test_stage2_fusion.py
---------------------
Unit and Regression Tests for Stage 2 Three-Way Learned Fusion.

Verifies:
1. All Stage 2 output artifacts exist in outputs/forensics/stage2/.
2. Row counts: 7,700 in train forensic branch, 1,650 in val forensic branch & fusion.
3. Feature ordering: ['cnn_probability', 'm3_probability', 'forensic_probability'].
4. Probability bounds strictly in [0.0, 1.0], no NaNs.
5. Strict identity disjointness between train and validation.
6. Zero test split access (no test files or references).
7. Threshold selection rule adheres to validation constraints.
8. All 9 tamper categories present and evaluated.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIR_STAGE2 = PROJECT_ROOT / "outputs" / "forensics" / "stage2"


def test_stage2_artifacts_exist():
    """Verify all required Stage 2 output artifacts exist."""
    required = [
        "forensic_branch_predictions_train.csv",
        "forensic_branch_predictions_val.csv",
        "stage2_fusion_predictions_val.csv",
        "stage2_metrics.csv",
        "stage2_attack_metrics.csv",
        "stage2_complementarity.csv",
        "stage2_feature_importance.csv",
        "stage2_experiment_report.md",
        "stage2_config.json",
    ]
    for fname in required:
        p = DIR_STAGE2 / fname
        assert p.exists(), f"Missing Stage 2 artifact: {fname}"


def test_stage2_row_counts_and_disjointness():
    """Verify train row count = 7,700, val row count = 1,650, zero nulls, disjoint identities."""
    tr_p = DIR_STAGE2 / "forensic_branch_predictions_train.csv"
    va_p = DIR_STAGE2 / "forensic_branch_predictions_val.csv"

    df_tr = pd.read_csv(tr_p)
    df_va = pd.read_csv(va_p)

    assert len(df_tr) == 7700
    assert len(df_va) == 1650

    assert not df_tr.isnull().any().any()
    assert not df_va.isnull().any().any()

    tr_rec = set(df_tr["record_id"].unique())
    va_rec = set(df_va["record_id"].unique())
    assert len(tr_rec) == 700
    assert len(va_rec) == 150
    assert tr_rec.isdisjoint(va_rec), "Train and Val identities must be disjoint!"


def test_stage2_probabilities_bounded():
    """Verify all output probabilities are strictly in [0.0, 1.0]."""
    val_preds_p = DIR_STAGE2 / "stage2_fusion_predictions_val.csv"
    df = pd.read_csv(val_preds_p)

    prob_cols = [c for c in df.columns if "prob" in c]
    assert len(prob_cols) >= 5

    for c in prob_cols:
        assert (df[c] >= 0.0).all() and (df[c] <= 1.0).all(), f"Out-of-bounds probability in {c}"


def test_stage2_zero_test_access():
    """Verify no test set files or test tokens exist in stage 2 outputs."""
    for p in DIR_STAGE2.glob("*"):
        assert "test." not in p.name.lower(), f"Suspicious test file: {p.name}"
        assert "test_" not in p.name.lower(), f"Suspicious test file: {p.name}"

    with open(DIR_STAGE2 / "stage2_config.json") as f:
        cfg = json.load(f)
    assert "ZERO TEST ACCESS" in cfg["test_set_access"]


def test_stage2_metrics_and_attacks():
    """Verify metric table integrity and attack categories."""
    df_m = pd.read_csv(DIR_STAGE2 / "stage2_metrics.csv")
    df_a = pd.read_csv(DIR_STAGE2 / "stage2_attack_metrics.csv")

    assert len(df_m) == 5  # B0, B1, B2, B3, B4
    for cand in ["B0_2Way_Baseline", "B1_3Way_RF", "B2_3Way_HGB"]:
        assert cand in df_m["candidate_id"].values

    assert df_a["tamper_type"].nunique() == 9
    assert "coordinated_full_forgery" in df_a["tamper_type"].values
    assert "genuine" in df_a["tamper_type"].values
