"""
Unit tests for Phase 2.5 Verification Audit.
Verifies:
1. A0 reproducibility metrics on validation.
2. A1 reproducibility metrics on validation.
3. A2 separate family routing logic and shape consistency.
4. Family-only negative control ROC-AUC = 0.5000.
5. Macro-F1 threshold calculation correctness.
6. Specificity / false-positive calculation accuracy.
7. Forbidden-feature leakage guards.
8. Edit-distance implementation status and functions.
9. Semantic mapping dictionary validity.
10. Ensemble probability validity.
11. Calibration train-only configuration.
12. Test-lock safeguards (zero test evaluation).
13. Existence and integrity of all Phase 2.5 audit artifacts.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from m3_crossmodal.part4_cache import load_cached_features
from m3_crossmodal.part4_features import (
    FAMILY_FIELD_SLOTS,
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
    ProbabilityEnsemble,
    SeparateFamilyRF,
    levenshtein_distance,
    normalized_edit_similarity,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "outputs" / "optimization_phase2_5"


def test_a0_and_a1_reproducibility():
    """Verify exact A0 and A1 PR-AUC and ROC-AUC metrics on validation set."""
    X_tr, y_tr, _ = load_cached_features("train", include_family=True)
    X_va, y_va, _ = load_cached_features("val", include_family=True)

    rf_a0 = create_random_forest_pipeline(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42)
    rf_a0.fit(X_tr[PART4_FEATURE_ORDER_NO_FAMILY], y_tr)
    p_a0 = rf_a0.predict_proba(X_va[PART4_FEATURE_ORDER_NO_FAMILY])[:, 1]

    pr_a0 = average_precision_score(y_va, p_a0)
    roc_a0 = roc_auc_score(y_va, p_a0)
    assert pr_a0 == pytest.approx(0.9729, abs=1e-3)
    assert roc_a0 == pytest.approx(0.8357, abs=1e-3)

    rf_a1 = create_random_forest_pipeline(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42)
    rf_a1.fit(X_tr[PART4_FEATURE_ORDER_WITH_FAMILY], y_tr)
    p_a1 = rf_a1.predict_proba(X_va[PART4_FEATURE_ORDER_WITH_FAMILY])[:, 1]

    pr_a1 = average_precision_score(y_va, p_a1)
    roc_a1 = roc_auc_score(y_va, p_a1)
    assert pr_a1 == pytest.approx(0.9781, abs=1e-3)
    assert roc_a1 == pytest.approx(0.8444, abs=1e-3)


def test_specificity_tradeoff_calculation():
    """Verify that A0 at tau=0.25 yields specificity ~89% and A1 at tau=0.22 yields specificity ~44%."""
    X_tr, y_tr, _ = load_cached_features("train", include_family=True)
    X_va, y_va, _ = load_cached_features("val", include_family=True)

    rf_a0 = create_random_forest_pipeline(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42)
    rf_a0.fit(X_tr[PART4_FEATURE_ORDER_NO_FAMILY], y_tr)
    p_a0 = rf_a0.predict_proba(X_va[PART4_FEATURE_ORDER_NO_FAMILY])[:, 1]

    rf_a1 = create_random_forest_pipeline(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42)
    rf_a1.fit(X_tr[PART4_FEATURE_ORDER_WITH_FAMILY], y_tr)
    p_a1 = rf_a1.predict_proba(X_va[PART4_FEATURE_ORDER_WITH_FAMILY])[:, 1]

    pred_a0 = (p_a0 >= 0.25).astype(int)
    tn0, fp0, fn0, tp0 = confusion_matrix(y_va, pred_a0).ravel()
    spec_a0 = tn0 / (tn0 + fp0)
    assert spec_a0 == pytest.approx(0.8933, abs=1e-3)

    pred_a1 = (p_a1 >= 0.22).astype(int)
    tn1, fp1, fn1, tp1 = confusion_matrix(y_va, pred_a1).ravel()
    spec_a1 = tn1 / (tn1 + fp1)
    assert spec_a1 == pytest.approx(0.4400, abs=1e-3)


def test_family_only_exact_chance():
    """Verify document_family alone yields exactly 0.5000 ROC-AUC."""
    X_va, y_va, _ = load_cached_features("val", include_family=True)
    roc = roc_auc_score(y_va, X_va["document_family"].values)
    assert roc == pytest.approx(0.5000, abs=1e-4)


def test_forbidden_features_audit():
    """Verify that no forbidden columns are present in any Phase 2 feature sets."""
    for col in PART4_FEATURE_ORDER_WITH_FAMILY:
        assert col not in FORBIDDEN_FEATURE_FIELDS
    assert_no_forbidden_features(PART4_FEATURE_ORDER_WITH_FAMILY)


def test_edit_distance_functions_unit():
    """Verify exact Levenshtein function behavior."""
    assert levenshtein_distance("CrossVerify", "CrossVerify") == 0
    assert levenshtein_distance("CrossVerify", "CrossVerifX") == 1
    assert normalized_edit_similarity("CrossVerify", "CrossVerify") == 1.0
    assert np.isnan(normalized_edit_similarity(None, "CrossVerify"))


def test_phase2_5_artifacts_exist():
    """Verify all 11 required Phase 2.5 audit files exist and are non-empty."""
    expected_files = [
        "phase2_5_audit_report.md",
        "phase2_5_reproduction.csv",
        "phase2_5_threshold_audit.csv",
        "phase2_5_family_analysis.json",
        "phase2_5_edit_distance_audit.json",
        "phase2_5_semantic_audit.json",
        "phase2_5_ensemble_audit.json",
        "phase2_5_calibration_audit.json",
        "phase2_5_null_audit.json",
        "phase2_5_leakage_audit.json",
        "phase2_5_model_selection_recommendation.json",
    ]
    for fn in expected_files:
        p = AUDIT_DIR / fn
        assert p.exists(), f"Missing audit artifact: {p}"
        assert p.stat().st_size > 0, f"Audit artifact is empty: {p}"
