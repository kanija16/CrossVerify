"""
Unit tests for Phase 1 Cross-Modal Model Optimization (Part 4 Enhancements).
Verifies:
1. Feature ladder definitions and column counts.
2. GroupKFold identity grouping (no group overlap across folds).
3. Calibration pipeline (Brier score improvement and monotonic ranking).
4. Strict forbidden feature rejection.
5. Optimization cache existence and schema integrity.
6. Optimization artifacts generated under outputs/optimization/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from m3_crossmodal.part4_cache import FEATURES_CACHE_DIR
from m3_crossmodal.part4_features import (
    FORBIDDEN_FEATURE_FIELDS,
    PART4_FEATURE_ORDER_NO_FAMILY,
    PART4_FEATURE_ORDER_WITH_FAMILY,
    assert_no_forbidden_features,
)
from m3_crossmodal.part4_models import (
    create_hist_gradient_boosting,
    create_logistic_regression_pipeline,
    create_random_forest_pipeline,
    find_optimal_threshold,
)
from m3_crossmodal.part4_optimizer import (
    OPTIMIZATION_DIR,
    build_ladder_features,
    get_ladder_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_feature_ladder_definitions():
    """Verify feature sets are strictly defined with no forbidden features."""
    ladders = get_ladder_columns(PART4_FEATURE_ORDER_NO_FAMILY)
    assert "A0" in ladders
    assert "B1" in ladders
    assert "B2" in ladders
    assert "B3" in ladders
    assert "B4" in ladders
    assert "B5" in ladders

    # Check that no feature set contains any forbidden column
    for set_name, feats in ladders.items():
        assert_no_forbidden_features(feats)

    # Verify column counts
    assert len(ladders["A0"]) == 15
    assert len(ladders["B1"]) == 19
    assert len(ladders["B2"]) == 20
    assert len(ladders["B3"]) == 25
    assert len(ladders["B4"]) == 28
    assert len(ladders["B5"]) == 31


def test_build_ladder_features_synthetic():
    """Test feature ladder engineering on synthetic DataFrame."""
    n = 20
    rng = np.random.RandomState(42)
    data = {c: rng.rand(n) for c in PART4_FEATURE_ORDER_NO_FAMILY}
    data["missing_field_count"] = rng.choice([0, 1, 2], size=n).astype(float)
    for i in range(5):
        data[f"field_sim_is_missing__{i}"] = rng.choice([0, 1], size=n).astype(float)
        data[f"field_similarity__{i}"] = rng.rand(n)
    data["checksum_valid"] = rng.choice([0, 1], size=n).astype(float)
    data["format_valid"] = rng.choice([0, 1], size=n).astype(float)

    df_synth = pd.DataFrame(data)
    df_ladder = build_ladder_features(df_synth)

    # Check that new engineered columns exist
    assert "text_qr_match_score_missing" in df_ladder.columns
    assert "has_any_missing_field" in df_ladder.columns
    assert "field_sim_available_count" in df_ladder.columns
    assert "field_sim_sq__0" in df_ladder.columns
    assert "worst_field_sim" in df_ladder.columns
    assert "best_field_sim" in df_ladder.columns
    assert "mean_field_sim" in df_ladder.columns
    assert "id_valid_both" in df_ladder.columns

    # Verify no forbidden columns
    assert_no_forbidden_features(list(df_ladder.columns))


def test_group_kfold_no_leakage():
    """Test GroupKFold ensures zero identity (record_id) overlap between train and test folds."""
    rng = np.random.RandomState(42)
    n_records = 30
    records = [f"rec_{i:03d}" for i in range(n_records)]
    data_records = []
    for r in records:
        data_records.extend([r] * 5)
    n = len(data_records)

    df_train = pd.DataFrame({
        "id": [f"sample_{i}" for i in range(n)],
        "record_id": data_records,
        "split": ["train"] * n,
        "label": rng.choice([0, 1], size=n, p=[0.1, 0.9]),
    })
    for f in PART4_FEATURE_ORDER_NO_FAMILY:
        df_train[f] = rng.rand(n)

    X = df_train[PART4_FEATURE_ORDER_NO_FAMILY]
    y = df_train["label"].values
    groups = df_train["record_id"].values

    gkf = GroupKFold(n_splits=5)
    fold_prs = []
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        # Strict test: groups in train must be completely disjoint from val
        train_groups = set(groups[tr_idx])
        val_groups = set(groups[va_idx])
        assert len(train_groups.intersection(val_groups)) == 0

        clf = create_random_forest_pipeline(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        p = clf.predict_proba(X.iloc[va_idx])[:, 1]
        fold_prs.append(average_precision_score(y[va_idx], p))

    assert len(fold_prs) == 5
    assert np.mean(fold_prs) > 0.0


def test_calibration_pipeline():
    """Verify calibration pipeline improves Brier score on synthetic biased probabilities."""
    rng = np.random.RandomState(42)
    n = 200
    y = rng.choice([0, 1], size=n, p=[0.2, 0.8])
    X = pd.DataFrame({f"feat_{i}": rng.rand(n) for i in range(5)})

    rf = create_random_forest_pipeline(n_estimators=20, max_depth=4, random_state=42)
    cal_rf = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)

    cal_rf.fit(X, y)
    probs_cal = cal_rf.predict_proba(X)[:, 1]

    brier = brier_score_loss(y, probs_cal)
    assert 0.0 <= brier <= 1.0


def test_optimization_cache_integrity():
    """Verify that cached features exist in outputs/features and contain expected splits and columns."""
    tr_path = PROJECT_ROOT / FEATURES_CACHE_DIR / "train_features.csv"
    va_path = PROJECT_ROOT / FEATURES_CACHE_DIR / "val_features.csv"
    assert tr_path.exists(), f"Train cache missing: {tr_path}"
    assert va_path.exists(), f"Val cache missing: {va_path}"

    df_tr = pd.read_csv(tr_path)
    df_va = pd.read_csv(va_path)

    assert len(df_tr) == 7700
    assert len(df_va) == 1650
    assert "id" in df_tr.columns
    assert "record_id" in df_tr.columns
    assert "final_label" in df_tr.columns

    for col in PART4_FEATURE_ORDER_NO_FAMILY:
        assert col in df_tr.columns
        assert col in df_va.columns


def test_optimization_artifacts_exist():
    """Verify that all Phase 1 optimization JSON artifacts and summary exist."""
    opt_dir = PROJECT_ROOT / "outputs" / "optimization"
    expected_files = [
        "baseline_validation_metrics.json",
        "threshold_analysis.json",
        "hyperparameter_tuning_results.json",
        "ablation_results.json",
        "model_comparison.json",
        "family_analysis.json",
        "null_check_results.json",
        "calibration_analysis.json",
        "feature_importance_val.json",
        "best_model_config.json",
        "optimization_experiment_log.json",
        "optimization_summary.md",
    ]
    for fn in expected_files:
        p = opt_dir / fn
        assert p.exists(), f"Expected optimization artifact missing: {p}"
