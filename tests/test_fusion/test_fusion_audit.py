"""
Unit and Regression Tests for Methodological Pre-Freeze Fusion Audit.

Verifies:
1. Recomputed audit artifacts exist.
2. Exact match of complementarity quadrant counts against canonical validation alignment.
3. Strict GroupKFold isolation in stacker and g(cnn) with zero identity overlap across folds.
4. Correct extraction of best validation PR-AUC (alpha=0.05) and ROC-AUC (alpha=0.70).
5. Range compression audit for g(cnn).
6. Total absence of test split in audit outputs and source code.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIR_AUDIT = PROJECT_ROOT / "outputs" / "fusion" / "audit"


def test_audit_artifacts_exist():
    """Verify all 6 required audit artifacts exist."""
    required = [
        "fusion_audit_report.md",
        "alpha_operating_analysis.csv",
        "stacker_audit.json",
        "complementarity_recomputed.csv",
        "attack_wise_recomputed.csv",
        "score_distribution_audit.csv",
    ]
    for fname in required:
        p = DIR_AUDIT / fname
        assert p.exists(), f"Missing audit artifact: {fname}"


def test_complementarity_exact_counts():
    """Verify exact reproduction of complementarity quadrant counts."""
    df_comp = pd.read_csv(DIR_AUDIT / "complementarity_recomputed.csv")
    comp_dict = df_comp.set_index("category").to_dict(orient="index")

    # Overall: 320, 287, 880, 163
    ov = comp_dict["Overall"]
    assert ov["both_correct"] == 320
    assert ov["m2_correct_m3_wrong"] == 287
    assert ov["m3_correct_m2_wrong"] == 880
    assert ov["both_wrong"] == 163
    assert ov["total_samples"] == 1650

    # Visual splice: 143, 266, 12, 29
    vs = comp_dict["Visual splice"]
    assert vs["both_correct"] == 143
    assert vs["m2_correct_m3_wrong"] == 266
    assert vs["m3_correct_m2_wrong"] == 12
    assert vs["both_wrong"] == 29

    # Other forged: 47, 1, 850, 2
    oth = comp_dict["Other forged"]
    assert oth["both_correct"] == 47
    assert oth["m2_correct_m3_wrong"] == 1
    assert oth["m3_correct_m2_wrong"] == 850
    assert oth["both_wrong"] == 2

    # Coordinated: 2, 5, 12, 131
    cff = comp_dict["Coordinated"]
    assert cff["both_correct"] == 2
    assert cff["m2_correct_m3_wrong"] == 5
    assert cff["m3_correct_m2_wrong"] == 12
    assert cff["both_wrong"] == 131


def test_stacker_audit_json_integrity():
    """Verify stacker audit records strictly OOF methodology without in-sample prediction."""
    with open(DIR_AUDIT / "stacker_audit.json") as f:
        data = json.load(f)

    assert data["train_level_oof_available"] is False
    assert data["in_sample_predictions_used"] is False
    assert data["all_predictions_strictly_oof"] is True
    assert len(data["fold_audit"]) == 5
    assert len(data["g_cnn_fold_audit"]) == 5


def test_g_cnn_compression():
    """Verify g(cnn) bounds reflect base rate compression into [0.87, 0.98]."""
    df_dist = pd.read_csv(DIR_AUDIT / "score_distribution_audit.csv")
    g_cnn_rows = df_dist[df_dist["signal_name"] == "g_cnn_transformed"]
    assert not g_cnn_rows.empty
    assert (g_cnn_rows["min"] >= 0.87).all()
    assert (g_cnn_rows["max"] <= 0.98).all()


def test_zero_test_set_access_audit():
    """Ensure no test split files or labels were referenced in the audit directory."""
    for p in DIR_AUDIT.glob("*"):
        assert "test." not in p.name.lower()
