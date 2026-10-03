"""
stage2_fusion.py
----------------
Stage 2: Three-Way Learned Fusion for CrossVerify M3.

Combines:
1. cnn_probability (M2 ResNet18 visual-splice detector)
2. m3_probability (Frozen M3 Random Forest cross-modal consistency detector)
3. forensic_probability (New A2 Forensic branch: 15 M3 + 16 Region Anomaly features)

Evaluates on TRAIN + VALIDATION ONLY.
Zero test data access.

Candidates evaluated:
- B0: 2-way frozen fusion [cnn_prob, m3_prob] (Baseline)
- B1: 3-way fusion [cnn_prob, m3_prob, forensic_rf_prob]
- B2: 3-way fusion [cnn_prob, m3_prob, forensic_hgb_prob]
- B3: 2-way direct fusion [cnn_prob, forensic_rf_prob]
- B4: 2-way direct fusion [cnn_prob, forensic_hgb_prob]

Saves all outputs to: outputs/forensics/stage2/
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
DIR_FUSION = PROJECT_ROOT / "outputs" / "fusion"
DIR_STAGE2 = DIR_FORENSICS / "stage2"

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


def load_alignment_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads existing upstream train and val alignment files."""
    train_align_p = DIR_FUSION / "train_oof_alignment.csv"
    val_align_p = DIR_FUSION / "val_alignment.csv"

    if not train_align_p.exists():
        raise FileNotFoundError(f"Missing {train_align_p}")
    if not val_align_p.exists():
        raise FileNotFoundError(f"Missing {val_align_p}")

    df_tr = pd.read_csv(train_align_p)
    df_va = pd.read_csv(val_align_p)

    return df_tr, df_va


def load_31_features(split: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Loads combined 31 A2 features for train or val."""
    if split == "test":
        raise ValueError("FORBIDDEN: Test split must never be loaded in this phase.")

    m3_p = DIR_FEATURES / f"{split}_features.csv"
    forensic_p = DIR_FORENSICS / f"{split}_forensic_features.csv"

    df_m3 = pd.read_csv(m3_p)
    df_forensic = pd.read_csv(forensic_p)

    assert len(df_m3) == len(df_forensic)
    np.testing.assert_array_equal(df_m3["id"].values, df_forensic["id"].values)

    X_m3 = df_m3[PART4_FEATURE_ORDER_NO_FAMILY]
    assert_no_forbidden_features(X_m3.columns)

    X_forensic = df_forensic[FORENSIC_FEATURE_NAMES]
    assert_clean_forensic_features(list(X_forensic.columns))

    X_31 = pd.concat([X_m3, X_forensic], axis=1)
    y = df_m3["final_label"].values.astype(int)
    meta = df_m3[["id", "record_id", "document_family", "tamper_type", "final_label"]].copy()

    return X_31, y, meta


def train_forensic_branches() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Trains A2 RF and A2 HGB models on train (31 features) and produces
    probabilities for both train and val.
    """
    print("Training A2 Forensic Branches on canonical training split (31 features, N=7,700)...")
    X_tr_31, y_tr, meta_tr = load_31_features("train")
    X_va_31, y_va, meta_va = load_31_features("val")

    # 1. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_tr_31, y_tr)
    rf_tr_prob = rf_model.predict_proba(X_tr_31)[:, 1]
    rf_va_prob = rf_model.predict_proba(X_va_31)[:, 1]

    # 2. HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
    )
    hgb_model.fit(X_tr_31, y_tr)
    hgb_tr_prob = hgb_model.predict_proba(X_tr_31)[:, 1]
    hgb_va_prob = hgb_model.predict_proba(X_va_31)[:, 1]

    # Assemble prediction dataframes
    df_preds_tr = meta_tr.copy()
    df_preds_tr["target"] = y_tr
    df_preds_tr["forensic_probability_rf"] = np.clip(rf_tr_prob, 0.0, 1.0)
    df_preds_tr["forensic_probability_hgb"] = np.clip(hgb_tr_prob, 0.0, 1.0)

    df_preds_va = meta_va.copy()
    df_preds_va["target"] = y_va
    df_preds_va["forensic_probability_rf"] = np.clip(rf_va_prob, 0.0, 1.0)
    df_preds_va["forensic_probability_hgb"] = np.clip(hgb_va_prob, 0.0, 1.0)

    models = {"rf": rf_model, "hgb": hgb_model}
    return df_preds_tr, df_preds_va, models


def find_optimal_threshold_stage2(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float = 0.90,
) -> Tuple[float, Dict[str, float]]:
    """
    Selects threshold on validation set maximizing F1 subject to min_specificity >= 0.90.
    Falls back to highest specificity threshold if 0.90 is not met.
    """
    best_tau = 0.50
    best_f1 = -1.0
    best_metrics = {}

    for tau in np.linspace(0.05, 0.95, 91):
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
                "fpr": round(float(fp / (tn + fp)), 4),
                "accuracy": round(float((tp + tn) / len(y_true)), 4),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

    if best_f1 < 0:
        # If min_specificity (0.90) cannot be met, fallback to max specificity operating point
        print(f"Notice: Specificity >= {min_specificity} not met. Finding closest defensible operating point...")
        max_spec = -1.0
        for tau in np.linspace(0.05, 0.95, 91):
            preds = (y_prob >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            if spec > max_spec:
                max_spec = spec
                best_tau = tau
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                best_metrics = {
                    "threshold": round(float(tau), 4),
                    "f1": round(float(f1), 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "specificity": round(float(spec), 4),
                    "fpr": round(float(fp / (tn + fp)), 4),
                    "accuracy": round(float((tp + tn) / len(y_true)), 4),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }

    return float(best_tau), best_metrics


def run_stage2_experiments() -> Dict[str, Any]:
    DIR_STAGE2.mkdir(parents=True, exist_ok=True)

    # 1. Load upstream alignment files
    df_align_tr, df_align_va = load_alignment_data()

    # 2. Train forensic branches and get probabilities
    df_forensic_tr, df_forensic_va, forensic_models = train_forensic_branches()

    # Save forensic branch prediction files
    df_forensic_tr.to_csv(DIR_STAGE2 / "forensic_branch_predictions_train.csv", index=False)
    df_forensic_va.to_csv(DIR_STAGE2 / "forensic_branch_predictions_val.csv", index=False)
    print(f"Saved forensic branch predictions to {DIR_STAGE2}")

    # 3. Assemble combined alignment matrices
    # Train
    y_tr = df_align_tr["target"].values.astype(int)
    p_cnn_tr = df_align_tr["cnn_probability"].values
    p_m3_tr = df_align_tr["m3_probability"].values
    p_forensic_rf_tr = df_forensic_tr["forensic_probability_rf"].values
    p_forensic_hgb_tr = df_forensic_tr["forensic_probability_hgb"].values

    # Val
    y_va = df_align_va["target"].values.astype(int)
    p_cnn_va = df_align_va["cnn_probability"].values
    p_m3_va = df_align_va["m3_probability"].values
    p_forensic_rf_va = df_forensic_va["forensic_probability_rf"].values
    p_forensic_hgb_va = df_forensic_va["forensic_probability_hgb"].values

    # 4. Define candidate configurations
    candidates = [
        ("B0_2Way_Baseline", ["cnn_probability", "m3_probability"],
         np.column_stack([p_cnn_tr, p_m3_tr]), np.column_stack([p_cnn_va, p_m3_va]),
         "Existing 2-way frozen fusion baseline [cnn_prob, m3_prob]"),
        
        ("B1_3Way_RF", ["cnn_probability", "m3_probability", "forensic_probability"],
         np.column_stack([p_cnn_tr, p_m3_tr, p_forensic_rf_tr]), np.column_stack([p_cnn_va, p_m3_va, p_forensic_rf_va]),
         "3-way learned fusion [cnn_prob, m3_prob, forensic_rf_prob]"),
        
        ("B2_3Way_HGB", ["cnn_probability", "m3_probability", "forensic_probability"],
         np.column_stack([p_cnn_tr, p_m3_tr, p_forensic_hgb_tr]), np.column_stack([p_cnn_va, p_m3_va, p_forensic_hgb_va]),
         "3-way learned fusion [cnn_prob, m3_prob, forensic_hgb_prob]"),
        
        ("B3_2Way_A2_RF", ["cnn_probability", "forensic_probability"],
         np.column_stack([p_cnn_tr, p_forensic_rf_tr]), np.column_stack([p_cnn_va, p_forensic_rf_va]),
         "2-way direct fusion [cnn_prob, forensic_rf_prob]"),
        
        ("B4_2Way_A2_HGB", ["cnn_probability", "forensic_probability"],
         np.column_stack([p_cnn_tr, p_forensic_hgb_tr]), np.column_stack([p_cnn_va, p_forensic_hgb_va]),
         "2-way direct fusion [cnn_prob, forensic_hgb_prob]"),
    ]

    metrics_records = []
    attack_records = []
    val_preds_df = df_align_va[["id", "record_id", "document_family", "tamper_type", "final_label", "target", "cnn_probability", "m3_probability"]].copy()
    val_preds_df["forensic_probability_rf"] = p_forensic_rf_va
    val_preds_df["forensic_probability_hgb"] = p_forensic_hgb_va

    fitted_stackers = {}

    for cand_id, feat_names, X_tr, X_va, desc in candidates:
        print(f"\nTraining stacker for {cand_id} ({desc})...")
        lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        lr.fit(X_tr, y_tr)
        probs_va = lr.predict_proba(X_va)[:, 1]
        fitted_stackers[cand_id] = lr

        val_preds_df[f"{cand_id}_prob"] = probs_va

        # Metrics
        pr_auc = average_precision_score(y_va, probs_va)
        roc_auc = roc_auc_score(y_va, probs_va)
        brier = brier_score_loss(y_va, probs_va)

        tau_opt, m_opt = find_optimal_threshold_stage2(y_va, probs_va, min_specificity=0.90)
        preds_va = (probs_va >= tau_opt).astype(int)
        val_preds_df[f"{cand_id}_pred"] = preds_va

        metrics_records.append({
            "candidate_id": cand_id,
            "description": desc,
            "input_features": str(feat_names),
            "weights": str([round(float(w), 4) for w in lr.coef_[0]]),
            "intercept": round(float(lr.intercept_[0]), 4),
            "pr_auc": round(float(pr_auc), 5),
            "roc_auc": round(float(roc_auc), 5),
            "brier_score": round(float(brier), 5),
            "threshold": tau_opt,
            "accuracy": m_opt["accuracy"],
            "precision": m_opt["precision"],
            "recall": m_opt["recall"],
            "f1": m_opt["f1"],
            "specificity": m_opt["specificity"],
            "fpr": m_opt["fpr"],
            "tn": m_opt["tn"],
            "fp": m_opt["fp"],
            "fn": m_opt["fn"],
            "tp": m_opt["tp"],
        })

        # Attack breakdown
        for tt in ALL_TAMPER_TYPES:
            mask = (val_preds_df["tamper_type"] == tt).values
            n_cat = int(np.sum(mask))
            cat_preds = preds_va[mask]

            if tt == "genuine":
                spec = float(np.mean(cat_preds == 0))
                attack_records.append({
                    "candidate_id": cand_id,
                    "tamper_type": tt,
                    "role": "genuine",
                    "N": n_cat,
                    "correct": int(np.sum(cat_preds == 0)),
                    "errors": int(np.sum(cat_preds == 1)),
                    "metric_name": "specificity",
                    "metric_value": round(spec, 4),
                })
            else:
                rec = float(np.mean(cat_preds == 1))
                attack_records.append({
                    "candidate_id": cand_id,
                    "tamper_type": tt,
                    "role": "forged_attack",
                    "N": n_cat,
                    "correct": int(np.sum(cat_preds == 1)),
                    "errors": int(np.sum(cat_preds == 0)),
                    "metric_name": "recall",
                    "metric_value": round(rec, 4),
                })

    df_metrics = pd.DataFrame(metrics_records)
    df_attacks = pd.DataFrame(attack_records)

    # Save validation predictions and metrics
    val_preds_df.to_csv(DIR_STAGE2 / "stage2_fusion_predictions_val.csv", index=False)
    df_metrics.to_csv(DIR_STAGE2 / "stage2_metrics.csv", index=False)
    df_attacks.to_csv(DIR_STAGE2 / "stage2_attack_metrics.csv", index=False)

    # 5. Targeted Hard-Attack & Complementarity Analysis
    # Samples where M2, M3, Forensic, and Fusion succeed/fail
    tau_b0 = df_metrics.loc[df_metrics["candidate_id"] == "B0_2Way_Baseline", "threshold"].values[0]
    tau_b1 = df_metrics.loc[df_metrics["candidate_id"] == "B1_3Way_RF", "threshold"].values[0]

    val_preds_df["m2_pred_05"] = (val_preds_df["cnn_probability"] >= 0.50).astype(int)
    val_preds_df["m3_pred_025"] = (val_preds_df["m3_probability"] >= 0.25).astype(int)
    val_preds_df["forensic_rf_pred_05"] = (val_preds_df["forensic_probability_rf"] >= 0.50).astype(int)

    comp_records = []
    for tt in ALL_TAMPER_TYPES:
        sub = val_preds_df[val_preds_df["tamper_type"] == tt]
        n_tt = len(sub)
        is_gen = (tt == "genuine")

        m2_corr = np.sum(sub["m2_pred_05"] == (0 if is_gen else 1))
        m3_corr = np.sum(sub["m3_pred_025"] == (0 if is_gen else 1))
        forensic_corr = np.sum(sub["forensic_rf_pred_05"] == (0 if is_gen else 1))
        b0_corr = np.sum(sub["B0_2Way_Baseline_pred"] == (0 if is_gen else 1))
        b1_corr = np.sum(sub["B1_3Way_RF_pred"] == (0 if is_gen else 1))

        # Focus on: M3 = wrong AND forensic = correct
        m3_wrong = sub["m3_pred_025"] != (0 if is_gen else 1)
        forensic_right = sub["forensic_rf_pred_05"] == (0 if is_gen else 1)
        rescued_by_forensic = int(np.sum(m3_wrong & forensic_right))

        comp_records.append({
            "tamper_type": tt,
            "N": n_tt,
            "m2_correct": int(m2_corr),
            "m3_correct": int(m3_corr),
            "forensic_correct": int(forensic_corr),
            "b0_2way_correct": int(b0_corr),
            "b1_3way_correct": int(b1_corr),
            "m3_wrong_and_forensic_right": rescued_by_forensic,
        })

    df_comp = pd.DataFrame(comp_records)
    df_comp.to_csv(DIR_STAGE2 / "stage2_complementarity.csv", index=False)

    # 6. Feature Importance on B1 3-way Logistic Regression
    b1_model = fitted_stackers["B1_3Way_RF"]
    X_va_b1 = np.column_stack([p_cnn_va, p_m3_va, p_forensic_rf_va])
    perm_b1 = permutation_importance(b1_model, X_va_b1, y_va, n_repeats=10, random_state=42, scoring="roc_auc")
    df_feat_imp = pd.DataFrame({
        "feature": ["cnn_probability", "m3_probability", "forensic_probability"],
        "logistic_weight": b1_model.coef_[0],
        "permutation_importance_mean": perm_b1.importances_mean,
        "permutation_importance_std": perm_b1.importances_std,
    })
    df_feat_imp.to_csv(DIR_STAGE2 / "stage2_feature_importance.csv", index=False)

    # 7. Config JSON
    stage2_cfg = {
        "status": "VALIDATION_COMPLETE",
        "test_set_access": "ZERO TEST ACCESS (Strictly Locked)",
        "train_data": "canonical train.csv (N=7,700, 700 identities)",
        "val_data": "canonical val.csv (N=1,650, 150 identities)",
        "training_data_limitation_note": "frozen-upstream training predictions with grouped fold annotations",
        "candidates": [c[0] for c in candidates],
        "primary_candidate": "B1_3Way_RF",
        "threshold_selection_rule": "maximize F1 subject to specificity >= 0.90 on validation",
    }
    with open(DIR_STAGE2 / "stage2_config.json", "w") as f:
        json.dump(stage2_cfg, f, indent=2)

    # 8. Markdown Experiment Report
    with open(DIR_STAGE2 / "stage2_experiment_report.md", "w") as f:
        f.write("# CrossVerify M3 — Stage 2 Three-Way Learned Fusion Report\n\n")
        f.write("**Status:** Validation Complete (TRAIN + VALIDATION ONLY)\n")
        f.write("**Test Split Access:** ZERO (Strictly Locked & Unloaded)\n\n")
        f.write("---\n\n")
        f.write("## 1. Candidate Model Descriptions & Stacker Weights\n\n")
        f.write(df_metrics[["candidate_id", "description", "input_features", "weights", "intercept"]].to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 2. Validation Metrics Across Candidates ($N=1,650$)\n\n")
        f.write(df_metrics[["candidate_id", "pr_auc", "roc_auc", "brier_score", "threshold", "accuracy", "precision", "recall", "f1", "specificity", "fpr"]].to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 3. Attack-Wise Validation Breakdown\n\n")
        piv = df_attacks.pivot(index="tamper_type", columns="candidate_id", values="metric_value")
        f.write(piv.to_markdown())
        f.write("\n\n---\n\n")
        f.write("## 4. Coordinated Full Forgery Focus ($N=150$ Validation Samples)\n\n")
        coord_df = df_attacks[df_attacks["tamper_type"] == "coordinated_full_forgery"]
        f.write(coord_df[["candidate_id", "N", "correct", "errors", "metric_value"]].to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 5. Hard-Attack & Complementarity Analysis\n\n")
        f.write(df_comp.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 6. Three-Way Stacker Feature Importance (B1 3-Way RF)\n\n")
        f.write(df_feat_imp.to_markdown(index=False))
        f.write("\n\n---\n\n")

    print(f"\nStage 2 experiments complete! All artifacts saved to {DIR_STAGE2}")
    return {
        "metrics": df_metrics,
        "attacks": df_attacks,
        "comp": df_comp,
        "importance": df_feat_imp,
    }


if __name__ == "__main__":
    run_stage2_experiments()
