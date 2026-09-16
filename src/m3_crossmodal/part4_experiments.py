"""
part4_experiments.py
--------------------
Orchestration of Part 4 experimental protocols:
- Ablation experiments A0–A6
- Algorithm comparisons (Logistic Regression, Random Forest, HistGradientBoosting)
- Validation-based threshold tuning (maximizing F1)
- Frozen single-shot test set evaluation
- Identity-grouped bootstrap confidence intervals
- McNemar's paired prediction test vs baseline
- Post-hoc attack-wise (9 classes) and family-wise breakdowns
- Permutation feature importance
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from .part4_features import (
    CORE_CONSISTENCY_FEATURES,
    COVERAGE_FEATURES,
    FIELD_SIMILARITY_FEATURES,
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
    evaluate_model,
    find_optimal_threshold,
    save_model_artifact,
)

# ---------------------------------------------------------------------------
# Ablation Feature Sets
# ---------------------------------------------------------------------------

ABLATION_FEATURE_SETS: Dict[str, List[str]] = {
    "A0": CORE_CONSISTENCY_FEATURES,
    "A1": CORE_CONSISTENCY_FEATURES + COVERAGE_FEATURES,
    "A2": CORE_CONSISTENCY_FEATURES + COVERAGE_FEATURES + FIELD_SIMILARITY_FEATURES,
    "A3": CORE_CONSISTENCY_FEATURES + COVERAGE_FEATURES + FIELD_SIMILARITY_FEATURES + VALIDATOR_DETAIL_FEATURES,
    "A4": PART4_FEATURE_ORDER_NO_FAMILY,
    "A5": PART4_FEATURE_ORDER_WITH_FAMILY,
    "A6": ["document_family"],
}

TAMPER_TYPES_9 = [
    "genuine",
    "text_qr_mismatch",
    "qr_only_mismatch",
    "checksum_invalid",
    "format_invalid",
    "field_missing",
    "visual_splice",
    "coordinated_full_forgery",
    "fine_grained_edit",
]


def run_ablation_experiments(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_type: str = "hgbt",
) -> Dict[str, Dict[str, Any]]:
    """
    Runs ablations A0 through A6 using the specified model architecture.
    Threshold is optimized on validation set to maximize F1.
    """
    ablation_results = {}

    for name, cols in ABLATION_FEATURE_SETS.items():
        assert_no_forbidden_features(cols)
        X_tr = X_train[cols].copy()
        X_v = X_val[cols].copy()

        if model_type == "hgbt":
            model = create_hist_gradient_boosting(random_state=RANDOM_SEED)
        elif model_type == "rf":
            model = create_random_forest_pipeline(random_state=RANDOM_SEED)
        elif model_type == "lr":
            model = create_logistic_regression_pipeline(random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_tr, y_train)

        # Predict probabilities on validation
        val_probs = model.predict_proba(X_v)[:, 1]
        best_th, best_f1 = find_optimal_threshold(y_val, val_probs)

        metrics = evaluate_model(model, X_v, y_val, threshold=best_th, split_name="val")
        ablation_results[name] = {
            "feature_set": cols,
            "num_features": len(cols),
            "metrics": metrics.to_dict(),
        }
        print(f"[{name}] features={len(cols)} -> Val ROC-AUC: {metrics.roc_auc:.4f}, PR-AUC: {metrics.pr_auc:.4f}, F1: {metrics.f1:.4f} (th={best_th:.2f})")

    return ablation_results


def compare_algorithms_on_validation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Fits and compares Logistic Regression, Random Forest, and HistGradientBoosting
    on the specified feature set. Selects optimal threshold on validation for each.
    """
    assert_no_forbidden_features(feature_cols)
    X_tr = X_train[feature_cols].copy()
    X_v = X_val[feature_cols].copy()

    models = {
        "logistic_regression": create_logistic_regression_pipeline(C=1.0, random_state=RANDOM_SEED),
        "random_forest": create_random_forest_pipeline(n_estimators=100, random_state=RANDOM_SEED),
        "hist_gradient_boosting": create_hist_gradient_boosting(random_state=RANDOM_SEED),
    }

    results = {}
    for m_name, model in models.items():
        model.fit(X_tr, y_train)
        val_probs = model.predict_proba(X_v)[:, 1]
        best_th, _ = find_optimal_threshold(y_val, val_probs)
        v_metrics = evaluate_model(model, X_v, y_val, threshold=best_th, split_name="val")
        results[m_name] = {
            "model": model,
            "threshold": best_th,
            "metrics": v_metrics.to_dict(),
        }
        print(f"[MODEL COMPARISON: {m_name}] Val ROC-AUC: {v_metrics.roc_auc:.4f}, PR-AUC: {v_metrics.pr_auc:.4f}, F1: {v_metrics.f1:.4f}")

    return results


def run_identity_grouped_bootstrap(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    threshold: float,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates test set performance with identity-grouped bootstrap resampling.
    Resamples unique identities (record_id base: id.split('_')[0]), preserving all
    11 variants per identity together.
    """
    rng = np.random.RandomState(random_state)

    # In M1 schema, record_id or metadata can extract the identity group:
    # Each identity has 11 records in test (150 unique identities * 11 = 1650 records)
    record_ids = meta_test["record_id"].tolist()
    # Group indices by identity:
    # Identity ID is the prefix before variant or can be grouped by consecutive blocks / ID map
    identity_map: Dict[str, List[int]] = {}
    for idx, rid in enumerate(record_ids):
        # Extract identity identifier (e.g. record_id or prefix)
        # In CrossVerify, identity can be parsed or grouped in blocks of 11
        id_key = rid.rsplit("_", 1)[0] if "_" in rid else str(idx // 11)
        identity_map.setdefault(id_key, []).append(idx)

    unique_identities = list(identity_map.keys())
    n_identities = len(unique_identities)

    boot_metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    probs = model.predict_proba(X_test)[:, 1]

    for _ in range(n_bootstraps):
        sampled_ids = rng.choice(unique_identities, size=n_identities, replace=True)
        sample_indices = []
        for sid in sampled_ids:
            sample_indices.extend(identity_map[sid])

        y_b = y_test[sample_indices]
        p_b = probs[sample_indices]
        pred_b = (p_b >= threshold).astype(int)

        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        try:
            boot_metrics["roc_auc"].append(float(roc_auc_score(y_b, p_b)))
        except Exception:
            pass
        boot_metrics["pr_auc"].append(float(average_precision_score(y_b, p_b)))
        boot_metrics["accuracy"].append(float(accuracy_score(y_b, pred_b)))
        boot_metrics["precision"].append(float(precision_score(y_b, pred_b, zero_division=0)))
        boot_metrics["recall"].append(float(recall_score(y_b, pred_b, zero_division=0)))
        boot_metrics["f1"].append(float(f1_score(y_b, pred_b, zero_division=0)))

    ci_results = {}
    for k, vals in boot_metrics.items():
        if vals:
            ci_results[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "ci_lower_95": round(float(np.percentile(vals, 2.5)), 4),
                "ci_upper_95": round(float(np.percentile(vals, 97.5)), 4),
            }
    return ci_results


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_new: np.ndarray,
) -> Dict[str, float]:
    """
    Computes McNemar's test on paired prediction disagreements between
    the baseline model and the new model on the same test set.
    """
    # Contingency table of predictions:
    # b: base correct, new incorrect
    # c: base incorrect, new correct
    b = int(np.sum((y_pred_base == y_true) & (y_pred_new != y_true)))
    c = int(np.sum((y_pred_base != y_true) & (y_pred_new == y_true)))

    # McNemar statistic with continuity correction
    if b + c > 0:
        stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val = float(stats.chi2.sf(stat, df=1))
    else:
        stat = 0.0
        p_val = 1.0

    return {
        "b_base_correct_new_wrong": b,
        "c_base_wrong_new_correct": c,
        "mcnemar_statistic": round(stat, 4),
        "p_value": round(p_val, 6),
    }


def compute_attack_wise_breakdown(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    meta_test: pd.DataFrame,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Post-hoc breakdown across all 9 tamper types on test set.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    results = {}
    y_pred = (y_prob >= threshold).astype(int)

    for tt in TAMPER_TYPES_9:
        mask = (meta_test["tamper_type"] == tt).to_numpy()
        n = int(np.sum(mask))
        if n == 0:
            continue

        y_sub = y_true[mask]
        p_sub = y_prob[mask]
        pred_sub = y_pred[mask]

        acc = float(accuracy_score(y_sub, pred_sub))
        prec = float(precision_score(y_sub, pred_sub, zero_division=0))
        rec = float(recall_score(y_sub, pred_sub, zero_division=0))
        f1 = float(f1_score(y_sub, pred_sub, zero_division=0))

        # ROC-AUC is only defined if both classes are present in the slice
        roc_auc_val = None
        if len(np.unique(y_sub)) > 1:
            try:
                roc_auc_val = round(float(roc_auc_score(y_sub, p_sub)), 4)
            except Exception:
                roc_auc_val = None

        results[tt] = {
            "n": n,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": roc_auc_val,
        }

    return results


def compute_family_wise_breakdown(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    X_test_with_family: pd.DataFrame,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Post-hoc breakdown between Family A and Family B on test set.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    results = {}
    y_pred = (y_prob >= threshold).astype(int)
    fam_col = X_test_with_family["document_family"].to_numpy()

    for fam_name, fam_val in [("family_a", 0.0), ("family_b", 1.0)]:
        mask = (fam_col == fam_val)
        n = int(np.sum(mask))
        if n == 0:
            continue

        y_sub = y_true[mask]
        p_sub = y_prob[mask]
        pred_sub = y_pred[mask]

        results[fam_name] = {
            "n": n,
            "accuracy": round(float(accuracy_score(y_sub, pred_sub)), 4),
            "precision": round(float(precision_score(y_sub, pred_sub, zero_division=0)), 4),
            "recall": round(float(recall_score(y_sub, pred_sub, zero_division=0)), 4),
            "f1": round(float(f1_score(y_sub, pred_sub, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_sub, p_sub)), 4),
            "pr_auc": round(float(average_precision_score(y_sub, p_sub)), 4),
        }

    return results


def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 10,
    random_state: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """
    Computes permutation feature importance on the test set.
    """
    perm = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats, random_state=random_state
    )
    cols = list(X_test.columns)
    importance_list = []
    for i, col in enumerate(cols):
        importance_list.append({
            "feature": col,
            "importance_mean": round(float(perm.importances_mean[i]), 5),
            "importance_std": round(float(perm.importances_std[i]), 5),
        })

    importance_list.sort(key=lambda x: x["importance_mean"], reverse=True)
    return importance_list


def compute_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """
    Computes calibration curve empirical fractions and mean predicted values.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {
        "prob_true": [round(float(v), 4) for v in prob_true],
        "prob_pred": [round(float(v), 4) for v in prob_pred],
    }
