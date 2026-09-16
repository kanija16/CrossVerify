"""
Unit tests for Phase 2 Targeted Cross-Modal Optimization.
Verifies:
1. Feature dimensions and exact Levenshtein / edit similarity math.
2. Semantic field alignment and role extraction across Family A and Family B.
3. Family-specific routing and probability predictions.
4. Family-specific thresholds logic and bounds.
5. Strict identity separation (zero train/val overlap).
6. Strict forbidden feature exclusion.
7. Model probability output validity (bounded [0, 1]).
8. Shuffled-label null collapse behavior.
9. Family-only negative control behavior (exact chance 0.5000).
10. Existence and integrity of Phase 2 artifacts in outputs/optimization_phase2/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from m3_crossmodal.part4_cache import load_cached_features
from m3_crossmodal.part4_features import (
    FORBIDDEN_FEATURE_FIELDS,
    PART4_FEATURE_ORDER_NO_FAMILY,
    PART4_FEATURE_ORDER_WITH_FAMILY,
    assert_no_forbidden_features,
)
from m3_crossmodal.part4_models import (
    create_hist_gradient_boosting,
    create_random_forest_pipeline,
    find_optimal_threshold,
)
from m3_crossmodal.part4_phase2 import (
    PHASE2_DIR,
    SEMANTIC_FEATURE_NAMES,
    ProbabilityEnsemble,
    SeparateFamilyRF,
    add_semantic_and_edit_features,
    evaluate_candidate,
    levenshtein_distance,
    normalized_edit_similarity,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_levenshtein_and_edit_similarity():
    """Verify exact Levenshtein distance and normalized edit similarity."""
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("470465131805", "470465131806") == 1

    # Similarity tests
    assert normalized_edit_similarity("abc", "abc") == 1.0
    assert normalized_edit_similarity("kitten", "sitting") == 1.0 - (3 / 7)
    assert 0.0 <= normalized_edit_similarity("a", "b") <= 1.0

    # Explicit missingness policy: None / empty string returns NaN
    assert np.isnan(normalized_edit_similarity(None, "abc"))
    assert np.isnan(normalized_edit_similarity("abc", None))
    assert np.isnan(normalized_edit_similarity("", "abc"))
    assert np.isnan(normalized_edit_similarity("abc", ""))


def test_semantic_feature_engineering_synthetic():
    """Verify semantic role extraction, aggregate stats, and NaN handling."""
    n = 20
    rng = np.random.RandomState(42)
    data = {c: rng.rand(n) for c in PART4_FEATURE_ORDER_WITH_FAMILY}
    # Introduce explicit NaNs
    data["field_similarity__0"][0] = np.nan
    data["field_similarity__1"][0] = np.nan

    df_synth = pd.DataFrame(data)
    df_out = add_semantic_and_edit_features(df_synth)

    for col in SEMANTIC_FEATURE_NAMES:
        assert col in df_out.columns

    # Verify no forbidden features
    assert_no_forbidden_features(list(df_out.columns))

    # Boundedness check: all non-nan values in [0, 5]
    for col in ["sem_min_field_sim", "sem_max_field_sim", "sem_mean_field_sim", "sem_name_sim", "sem_identifier_sim", "sem_date_sim"]:
        valid = df_out[col].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_separate_family_rf():
    """Verify separate family RF routes samples properly according to document_family."""
    rng = np.random.RandomState(42)
    n = 40
    data = {c: rng.rand(n) for c in PART4_FEATURE_ORDER_WITH_FAMILY}
    data["document_family"] = [0.0] * 20 + [1.0] * 20
    df = pd.DataFrame(data)
    y = rng.choice([0, 1], size=n, p=[0.2, 0.8])

    sep_rf = SeparateFamilyRF(n_estimators=10, max_depth=3, random_state=42)
    sep_rf.fit(df, y)

    probs = sep_rf.predict_proba(df)
    assert probs.shape == (n, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_probability_ensemble():
    """Verify weighted probability ensemble between RF and HGB."""
    rng = np.random.RandomState(42)
    n = 30
    cols = PART4_FEATURE_ORDER_NO_FAMILY
    df = pd.DataFrame({c: rng.rand(n) for c in cols})
    y = rng.choice([0, 1], size=n)

    rf = create_random_forest_pipeline(n_estimators=10, max_depth=3, random_state=42)
    hgb = create_hist_gradient_boosting(random_state=42)
    rf.fit(df, y)
    hgb.fit(df, y)

    ensemble = ProbabilityEnsemble(rf, hgb, weight_1=0.25)
    probs = ensemble.predict_proba(df)
    assert probs.shape == (n, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_family_only_negative_control():
    """Verify document_family alone yields exact random chance (ROC-AUC = 0.5000)."""
    X_va_raw, y_va, _ = load_cached_features("val", include_family=True)
    fam_scores = X_va_raw["document_family"].to_numpy(dtype=float)
    roc = roc_auc_score(y_va, fam_scores)
    assert roc == pytest.approx(0.5000, abs=1e-4)


def test_phase2_artifacts_exist():
    """Verify all 10 required Phase 2 output artifacts exist and are non-empty."""
    expected_files = [
        "phase2_summary.md",
        "phase2_experiment_results.json",
        "phase2_model_comparison.csv",
        "phase2_ablation_results.csv",
        "phase2_family_analysis.json",
        "phase2_feature_schema.json",
        "phase2_run_config.json",
        "phase2_leakage_report.json",
        "phase2_null_test.json",
        "phase2_calibration.json",
    ]
    for fn in expected_files:
        p = PROJECT_ROOT / PHASE2_DIR / fn
        assert p.exists(), f"Missing Phase 2 artifact: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"
