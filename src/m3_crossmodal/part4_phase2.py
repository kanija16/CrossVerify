"""
part4_phase2.py
----------------
Phase 2 Targeted Cross-Modal Optimization Module.

Contains:
1. Exact normalized Levenshtein edit distance and similarity calculations.
2. Semantic field alignment and role extraction across Family A and Family B.
3. Family-aware model architectures (pooled, family-conditioned, separate family models).
4. Family-specific threshold routing and evaluation.
5. Controlled RF + HGB probability ensembling.
6. Rigorous leakage verification and null test controls.
7. Post-hoc attack-wise validation breakdown.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .part4_cache import load_cached_features
from .part4_features import (
    CORE_CONSISTENCY_FEATURES,
    COVERAGE_FEATURES,
    FAMILY_FIELD_SLOTS,
    FIELD_SIMILARITY_FEATURES,
    FORBIDDEN_FEATURE_FIELDS,
    PART4_FEATURE_ORDER_NO_FAMILY,
    PART4_FEATURE_ORDER_WITH_FAMILY,
    VALIDATOR_DETAIL_FEATURES,
    assert_no_forbidden_features,
)
from .part4_models import (
    RANDOM_SEED,
    create_hist_gradient_boosting,
    create_logistic_regression_pipeline,
    create_random_forest_pipeline,
    find_optimal_threshold,
)

PHASE2_DIR = Path("outputs/optimization_phase2")


# ---------------------------------------------------------------------------
# 1. Exact Normalized Levenshtein Edit Distance and Similarity
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes exact Levenshtein edit distance (insertions, deletions, substitutions).
    Time: O(len(s1) * len(s2)), Space: O(min(len(s1), len(s2))).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_edit_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """
    Computes normalized edit similarity: 1.0 - (edit_distance / max_len).
    Range: [0.0, 1.0].
    Returns np.nan if either string is None (explicit missingness policy).
    """
    if s1 is None or s2 is None:
        return np.nan
    s1_str = str(s1).strip()
    s2_str = str(s2).strip()
    if not s1_str or not s2_str:
        return np.nan

    max_l = max(len(s1_str), len(s2_str))
    if max_l == 0:
        return 1.0
    dist = levenshtein_distance(s1_str, s2_str)
    sim = 1.0 - (dist / max_l)
    return float(max(0.0, min(1.0, sim)))


# ---------------------------------------------------------------------------
# 2. Semantic Role Feature Engineering
# ---------------------------------------------------------------------------

SEMANTIC_FEATURE_NAMES = [
    "sem_min_field_sim",
    "sem_max_field_sim",
    "sem_mean_field_sim",
    "sem_num_exact_matches",
    "sem_num_severe_mismatches",
    "sem_name_sim",
    "sem_identifier_sim",
    "sem_date_sim",
]


def add_semantic_and_edit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs semantic alignment and edit similarity features from raw tabular features.
    Strictly NaN-safe and range-bounded in [0.0, 1.0].
    """
    f = df.copy()
    sim_cols = [f"field_similarity__{i}" for i in range(5)]
    sim_mat = f[sim_cols].to_numpy()

    with np.errstate(all="ignore"):
        f["sem_min_field_sim"] = np.nanmin(sim_mat, axis=1)
        f["sem_max_field_sim"] = np.nanmax(sim_mat, axis=1)
        f["sem_mean_field_sim"] = np.nanmean(sim_mat, axis=1)

    all_nan = np.isnan(sim_mat).all(axis=1)
    f.loc[all_nan, "sem_min_field_sim"] = np.nan
    f.loc[all_nan, "sem_max_field_sim"] = np.nan
    f.loc[all_nan, "sem_mean_field_sim"] = np.nan

    # Exact matches (sim == 1.0) and severe mismatches (sim < 0.8)
    f["sem_num_exact_matches"] = np.nansum(sim_mat == 1.0, axis=1).astype(float)
    f["sem_num_severe_mismatches"] = np.nansum((sim_mat < 0.8) & (~np.isnan(sim_mat)), axis=1).astype(float)
    f.loc[all_nan, "sem_num_exact_matches"] = np.nan
    f.loc[all_nan, "sem_num_severe_mismatches"] = np.nan

    # Semantic roles:
    # Family A slots: 0: name, 1: dob, 2: gender, 3: id_number, 4: address
    # Family B slots: 0: full_name, 1: registry_id, 2: entity_type, 3: jurisdiction_code, 4: registration_date
    if "document_family" in f.columns:
        is_fam_a = (f["document_family"] == 0.0).to_numpy()
        f["sem_name_sim"] = f["field_similarity__0"]
        f["sem_identifier_sim"] = np.where(is_fam_a, f["field_similarity__3"], f["field_similarity__1"])
        f["sem_date_sim"] = np.where(is_fam_a, f["field_similarity__1"], f["field_similarity__4"])
    else:
        # Default fallback if family column not present
        f["sem_name_sim"] = f["field_similarity__0"]
        f["sem_identifier_sim"] = f["field_similarity__3"]
        f["sem_date_sim"] = f["field_similarity__1"]

    return f


# ---------------------------------------------------------------------------
# 3. Model Wrappers & Ensembles
# ---------------------------------------------------------------------------

class SeparateFamilyRF:
    """
    Family-routed model: trains separate Random Forests for Family A and Family B.
    At inference, routes samples based on document_family.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 8, min_samples_split: int = 4, random_state: int = 42):
        self.rf_a = create_random_forest_pipeline(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
        self.rf_b = create_random_forest_pipeline(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, family_col: str = "document_family"):
        self.feature_cols = [c for c in X.columns if c != family_col]
        assert_no_forbidden_features(self.feature_cols)

        mask_a = (X[family_col] == 0.0).values
        mask_b = (X[family_col] == 1.0).values

        self.rf_a.fit(X.loc[mask_a, self.feature_cols], y[mask_a])
        self.rf_b.fit(X.loc[mask_b, self.feature_cols], y[mask_b])
        return self

    def predict_proba(self, X: pd.DataFrame, family_col: str = "document_family") -> np.ndarray:
        assert_no_forbidden_features(self.feature_cols)
        probs = np.zeros((len(X), 2), dtype=float)

        mask_a = (X[family_col] == 0.0).values
        mask_b = (X[family_col] == 1.0).values

        if np.any(mask_a):
            probs[mask_a] = self.rf_a.predict_proba(X.loc[mask_a, self.feature_cols])
        if np.any(mask_b):
            probs[mask_b] = self.rf_b.predict_proba(X.loc[mask_b, self.feature_cols])
        return probs


class ProbabilityEnsemble:
    """
    Weighted probability ensemble between two probabilistic classifiers.
    """

    def __init__(self, model_1: Any, model_2: Any, weight_1: float = 0.5):
        self.model_1 = model_1
        self.model_2 = model_2
        self.weight_1 = weight_1
        self.weight_2 = 1.0 - weight_1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.model_1.predict_proba(X)
        p2 = self.model_2.predict_proba(X)
        return self.weight_1 * p1 + self.weight_2 * p2


# ---------------------------------------------------------------------------
# 4. Comprehensive Evaluation Helper
# ---------------------------------------------------------------------------

def evaluate_candidate(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: Optional[float] = None,
    candidate_name: str = "",
) -> Dict[str, Any]:
    """
    Computes PR-AUC, ROC-AUC, F1, Precision, Recall, Accuracy, Brier Score, and Confusion Matrix.
    """
    pr_auc = float(average_precision_score(y_true, probs))
    roc_auc = float(roc_auc_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))

    if threshold is None:
        th, _ = find_optimal_threshold(y_true, probs, criterion="macro_f1")
    else:
        th = float(threshold)

    preds = (probs >= th).astype(int)
    f1 = float(f1_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    acc = float(accuracy_score(y_true, preds))
    cm = confusion_matrix(y_true, preds).tolist()

    return {
        "candidate": candidate_name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "threshold": round(th, 4),
        "confusion_matrix": cm,
    }
