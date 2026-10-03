"""
ablation_runner.py
------------------
Ablation Experiments & Hard-Attack Evaluation for CrossVerify M3 Image Forensics.

Runs on TRAIN + VALIDATION ONLY.
Zero test data access.

Ablations:
- A0: Frozen 15-feature M3 baseline
- A1: 16-feature Forensic-only model
- A2: 31-feature Combined (15 M3 + 16 Forensic) model

Models:
- Random Forest (100 estimators, max_depth=12, random_state=42)
- HistGradientBoosting (max_iter=100, max_depth=6, random_state=42)

Produces:
- outputs/forensics/ablation_metrics.csv
- outputs/forensics/attack_wise_ablations.csv
- outputs/forensics/coordinated_full_forgery_analysis.csv
- outputs/forensics/feature_importance.csv
- outputs/forensics/forensics_experiment_report.md
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
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

from m3_crossmodal.forensics.region_features import (
    FORENSIC_FEATURE_NAMES,
    assert_clean_forensic_features,
)
from m3_crossmodal.part4_features import (
    PART4_FEATURE_ORDER_NO_FAMILY,
    assert_no_forbidden_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DIR_FEATURES = PROJECT_ROOT / "outputs" / "features"
DIR_FORENSICS = PROJECT_ROOT / "outputs" / "forensics"

ALL_TAMPER_TYPES = [
    "genuine",
    "text_qr_mismatch",
    "qr_only_mismatch",
    "checksum_invalid",
    "format_invalid",
    "field_missing",
    "visual_splice",
    "fine_grained_edit",
    "coordinated_full_forgery",
]


def load_dataset(split: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Loads M3 features and Forensic features for a given split (train or val).
    Strictly asserts split != 'test'.
    """
    if split == "test":
        raise ValueError("FORBIDDEN: Test split must never be loaded in this phase.")

    m3_path = DIR_FEATURES / f"{split}_features.csv"
    forensic_path = DIR_FORENSICS / f"{split}_forensic_features.csv"

    if not m3_path.exists():
        raise FileNotFoundError(f"Missing M3 features at {m3_path}")
    if not forensic_path.exists():
        raise FileNotFoundError(f"Missing forensic features at {forensic_path}")

    df_m3 = pd.read_csv(m3_path)
    df_forensic = pd.read_csv(forensic_path)

    assert len(df_m3) == len(df_forensic), f"Row count mismatch: M3={len(df_m3)}, Forensic={len(df_forensic)}"
    np.testing.assert_array_equal(df_m3["id"].values, df_forensic["id"].values, err_msg="ID mismatch between caches")

    # Features
    X_m3 = df_m3[PART4_FEATURE_ORDER_NO_FAMILY].copy()
    assert_no_forbidden_features(X_m3.columns)

    X_forensic = df_forensic[FORENSIC_FEATURE_NAMES].copy()
    assert_clean_forensic_features(list(X_forensic.columns))

    y = df_m3["final_label"].values.astype(int)
    meta = df_m3[["id", "record_id", "document_family", "tamper_type", "final_label"]].copy()

    return X_m3, X_forensic, y, meta


def find_optimal_threshold_val(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float = 0.85,
) -> Tuple[float, Dict[str, float]]:
    """
    Selects threshold on validation set that maximizes F1 while satisfying min_specificity.
    """
    best_tau = 0.50
    best_f1 = -1.0
    best_metrics = {}

    for tau in np.linspace(0.10, 0.90, 81):
        preds = (y_prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if spec >= min_specificity and f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_metrics = {
                "threshold": round(float(tau), 4),
                "f1": round(float(f1), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "specificity": round(float(spec), 4),
                "accuracy": round(float((tp + tn) / len(y_true)), 4),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

    if best_f1 < 0:
        # Fallback to tau=0.5
        preds = (y_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        best_tau = 0.5
        best_metrics = {
            "threshold": 0.5,
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "specificity": round(float(tn / (tn + fp)), 4),
            "accuracy": round(float((tp + tn) / len(y_true)), 4),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    return float(best_tau), best_metrics


def evaluate_attack_breakdown(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tau: float,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes recall per tamper type and specificity on genuine."""
    preds = (y_prob >= tau).astype(int)
    records = []

    for tt in ALL_TAMPER_TYPES:
        mask = (meta_df["tamper_type"] == tt).values
        n_cat = int(np.sum(mask))
        if n_cat == 0:
            continue

        cat_true = y_true[mask]
        cat_pred = preds[mask]

        if tt == "genuine":
            # Specificity = TN / N
            spec = float(np.mean(cat_pred == 0))
            records.append({
                "tamper_type": tt,
                "role": "genuine",
                "N": n_cat,
                "metric": "specificity",
                "score": round(spec, 4),
                "tp_or_tn": int(np.sum(cat_pred == 0)),
                "fn_or_fp": int(np.sum(cat_pred == 1)),
            })
        else:
            # Recall = TP / N
            rec = float(np.mean(cat_pred == 1))
            records.append({
                "tamper_type": tt,
                "role": "forged_attack",
                "N": n_cat,
                "metric": "recall",
                "score": round(rec, 4),
                "tp_or_tn": int(np.sum(cat_pred == 1)),
                "fn_or_fp": int(np.sum(cat_pred == 0)),
            })

    return pd.DataFrame(records)


def run_ablations() -> Dict[str, Any]:
    print("Loading train and val feature sets...")
    X_m3_train, X_forensic_train, y_train, meta_train = load_dataset("train")
    X_m3_val, X_forensic_val, y_val, meta_val = load_dataset("val")

    # Combine feature sets
    X_combined_train = pd.concat([X_m3_train, X_forensic_train], axis=1)
    X_combined_val = pd.concat([X_m3_val, X_forensic_val], axis=1)

    ablation_configs = [
        ("A0_RF_Baseline", "RandomForest", X_m3_train, X_m3_val, "A0: 15 M3 Features Baseline"),
        ("A0_HGB_Baseline", "HistGradientBoosting", X_m3_train, X_m3_val, "A0: 15 M3 Features Baseline"),
        ("A1_RF_ForensicOnly", "RandomForest", X_forensic_train, X_forensic_val, "A1: 16 Forensic-Only Features"),
        ("A1_HGB_ForensicOnly", "HistGradientBoosting", X_forensic_train, X_forensic_val, "A1: 16 Forensic-Only Features"),
        ("A2_RF_Combined", "RandomForest", X_combined_train, X_combined_val, "A2: 15 M3 + 16 Forensic Combined"),
        ("A2_HGB_Combined", "HistGradientBoosting", X_combined_train, X_combined_val, "A2: 15 M3 + 16 Forensic Combined"),
    ]

    metrics_rows = []
    attack_dfs = []
    trained_models = {}
    val_probs = {}

    for ab_id, model_type, X_tr, X_va, desc in ablation_configs:
        print(f"\nEvaluating {ab_id} ({desc})...")
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
            )

        model.fit(X_tr, y_train)
        probs = model.predict_proba(X_va)[:, 1]
        trained_models[ab_id] = model
        val_probs[ab_id] = probs

        pr_auc = average_precision_score(y_val, probs)
        roc_auc = roc_auc_score(y_val, probs)
        brier = brier_score_loss(y_val, probs)

        tau_opt, opt_m = find_optimal_threshold_val(y_val, probs, min_specificity=0.85)

        metrics_rows.append({
            "ablation_id": ab_id,
            "description": desc,
            "model": model_type,
            "n_features": X_tr.shape[1],
            "pr_auc": round(float(pr_auc), 5),
            "roc_auc": round(float(roc_auc), 5),
            "brier_score": round(float(brier), 5),
            "threshold": tau_opt,
            "accuracy": opt_m["accuracy"],
            "precision": opt_m["precision"],
            "recall": opt_m["recall"],
            "f1": opt_m["f1"],
            "specificity": opt_m["specificity"],
            "tn": opt_m["tn"],
            "fp": opt_m["fp"],
            "fn": opt_m["fn"],
            "tp": opt_m["tp"],
        })

        # Attack breakdown
        df_att = evaluate_attack_breakdown(y_val, probs, tau_opt, meta_val)
        df_att["ablation_id"] = ab_id
        attack_dfs.append(df_att)

    metrics_df = pd.DataFrame(metrics_rows)
    all_attacks_df = pd.concat(attack_dfs, ignore_index=True)

    metrics_df.to_csv(DIR_FORENSICS / "ablation_metrics.csv", index=False)
    all_attacks_df.to_csv(DIR_FORENSICS / "attack_wise_ablations.csv", index=False)

    # Focus analysis on coordinated_full_forgery
    coord_analysis = all_attacks_df[all_attacks_df["tamper_type"] == "coordinated_full_forgery"].copy()
    coord_analysis.to_csv(DIR_FORENSICS / "coordinated_full_forgery_analysis.csv", index=False)

    # Permutation feature importance on A2_RF_Combined
    print("\nComputing permutation feature importance for A2_RF_Combined...")
    rf_combined = trained_models["A2_RF_Combined"]
    perm_res = permutation_importance(
        rf_combined,
        X_combined_val,
        y_val,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )
    feat_imp_df = pd.DataFrame({
        "feature": X_combined_val.columns,
        "importance_mean": perm_res.importances_mean,
        "importance_std": perm_res.importances_std,
        "feature_group": ["m3" if c in PART4_FEATURE_ORDER_NO_FAMILY else "forensic" for c in X_combined_val.columns],
    }).sort_values(by="importance_mean", ascending=False)
    feat_imp_df.to_csv(DIR_FORENSICS / "feature_importance.csv", index=False)

    # Generate experiment report
    report_path = DIR_FORENSICS / "forensics_experiment_report.md"
    with open(report_path, "w") as f:
        f.write("# CrossVerify M3 — Stage 1 Forensic Feature Engineering Report\n\n")
        f.write("**Status:** Validation Complete (TRAIN + VALIDATION ONLY)\n")
        f.write("**Test Split Access:** ZERO (Strictly Locked & Unloaded)\n\n")
        f.write("---\n\n")
        f.write("## 1. Overall Validation Metrics Across Ablations\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 2. Attack-Wise Recall Breakdown\n\n")
        
        # Pivot attack table for easy reading
        piv = all_attacks_df.pivot(index="tamper_type", columns="ablation_id", values="score")
        f.write(piv.to_markdown())
        f.write("\n\n---\n\n")
        f.write("## 3. Coordinated Full Forgery Focus ($N=150$ Validation Samples)\n\n")
        f.write(coord_analysis[["ablation_id", "N", "metric", "score", "tp_or_tn", "fn_or_fp"]].to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 4. Top 15 Feature Importances (A2 Combined RF)\n\n")
        f.write(feat_imp_df.head(15).to_markdown(index=False))
        f.write("\n\n---\n\n")

    print(f"\nAll forensic ablation artifacts generated in {DIR_FORENSICS}")
    return {
        "metrics": metrics_df,
        "attacks": all_attacks_df,
        "coord": coord_analysis,
        "importance": feat_imp_df,
    }


if __name__ == "__main__":
    run_ablations()
