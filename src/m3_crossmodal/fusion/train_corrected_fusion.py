"""
CrossVerify Phase 1B-B: Corrected Fusion Training & Validation Threshold Selection.

Implements:
1. Alignment of genuine 5-fold OOF predictions from Member 2 (CNN) and Member 3 (Cross-modal).
2. Training of corrected late-fusion Logistic Regression strictly on 7,700 genuine OOF training rows.
3. Diagnostic evaluation on OOF training predictions (labeled as diagnostics only).
4. Upstream validation prediction collation from canonical val.csv.
5. Inference of corrected fusion model on validation upstream predictions.
6. Operating threshold selection under operational constraint: max F1 subject to Specificity >= 0.90.
7. Exporting corrected artifacts and configurations with provenance hashes.

Enforces zero-test-set access: canonical test split is never loaded, fitted, or evaluated.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Paths
TRAIN_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"

CNN_OOF_PATH = PROJECT_ROOT / "outputs" / "fusion" / "oof" / "cnn_oof_predictions.csv"
M3_OOF_PATH = PROJECT_ROOT / "outputs" / "fusion" / "oof" / "train_oof_predictions.csv"

VAL_ALIGN_PATH = PROJECT_ROOT / "outputs" / "fusion" / "val_alignment.csv"
CNN_VAL_PATH = PROJECT_ROOT / "models" / "cnn" / "val_predictions.csv"

OUTPUT_OOF_ALIGNED_PATH = PROJECT_ROOT / "outputs" / "fusion" / "oof" / "corrected_fusion_oof.csv"
OUTPUT_VAL_UPSTREAM_PATH = PROJECT_ROOT / "outputs" / "fusion" / "val_upstream_predictions.csv"
OUTPUT_VAL_PREDS_PATH = PROJECT_ROOT / "outputs" / "fusion" / "corrected_fusion_validation_predictions.csv"

MODELS_FUSION_DIR = PROJECT_ROOT / "models" / "fusion"
CORRECTED_MODEL_PATH = MODELS_FUSION_DIR / "corrected_fusion_model.joblib"
CORRECTED_CONFIG_PATH = MODELS_FUSION_DIR / "corrected_fusion_config.json"
OUTPUT_CONFIG_PATH = PROJECT_ROOT / "outputs" / "fusion" / "corrected_fusion_config.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def step1_align_oof_data() -> pd.DataFrame:
    print("=" * 80)
    print("STEP 1: ALIGNING GENUINE OOF PREDICTIONS")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    cnn_oof = pd.read_csv(CNN_OOF_PATH)
    m3_oof = pd.read_csv(M3_OOF_PATH)

    assert len(cnn_oof) == 7700, f"CNN OOF expected 7700 rows, got {len(cnn_oof)}"
    assert len(m3_oof) == 7700, f"M3 OOF expected 7700 rows, got {len(m3_oof)}"
    assert len(train_df) == 7700, f"train.csv expected 7700 rows, got {len(train_df)}"

    # Check ID alignment
    assert (cnn_oof["id"] == train_df["id"]).all(), "CNN OOF IDs do not match train.csv"
    assert (m3_oof["id"] == train_df["id"]).all(), "M3 OOF IDs do not match train.csv"

    # Check record_id alignment
    assert (cnn_oof["record_id"] == train_df["record_id"]).all(), "CNN OOF record_ids do not match train.csv"
    assert (m3_oof["record_id"] == train_df["record_id"]).all(), "M3 OOF record_ids do not match train.csv"

    # Check final_label alignment
    assert (cnn_oof["final_label"] == train_df["final_label"]).all(), "CNN OOF final_label does not match train.csv"
    assert (m3_oof["final_label"] == train_df["final_label"]).all(), "M3 OOF final_label does not match train.csv"

    # Check fold alignment
    assert (cnn_oof["fold"] == m3_oof["fold"]).all(), "Fold assignments between CNN OOF and M3 OOF do not match"

    # Check zero nulls
    assert cnn_oof["cnn_probability_oof"].isnull().sum() == 0, "Missing values in cnn_probability_oof"
    assert m3_oof["m3_probability_oof"].isnull().sum() == 0, "Missing values in m3_probability_oof"

    # Ensure probabilities bounded in [0, 1]
    assert cnn_oof["cnn_probability_oof"].min() >= 0.0 and cnn_oof["cnn_probability_oof"].max() <= 1.0
    assert m3_oof["m3_probability_oof"].min() >= 0.0 and m3_oof["m3_probability_oof"].max() <= 1.0

    print("[PASS] Verified exact alignment of 7,700 rows across CNN OOF and M3 OOF.")

    # Create aligned dataframe with required columns
    corrected_oof_df = pd.DataFrame({
        "id": train_df["id"],
        "document_family": train_df["document_family"],
        "record_id": train_df["record_id"],
        "template_variant": train_df["template_variant"],
        "tamper_type": train_df["tamper_type"],
        "final_label": train_df["final_label"],
        "cnn_label": train_df["cnn_label"],
        "fold": cnn_oof["fold"],
        "cnn_probability_oof": cnn_oof["cnn_probability_oof"],
        "m3_probability_oof": m3_oof["m3_probability_oof"],
    })

    OUTPUT_OOF_ALIGNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    corrected_oof_df.to_csv(OUTPUT_OOF_ALIGNED_PATH, index=False)
    print(f"Saved aligned OOF artifact to: {OUTPUT_OOF_ALIGNED_PATH}")
    return corrected_oof_df


def step2_train_corrected_fusion_model(oof_df: pd.DataFrame) -> Tuple[LogisticRegression, Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING CORRECTED FUSION MODEL ON 7,700 GENUINE OOF ROWS")
    print("=" * 80)

    feature_names = ["cnn_probability_oof", "m3_probability_oof"]
    X_train = oof_df[feature_names].values
    y_train = (oof_df["final_label"] == "forged").astype(int).values

    assert len(X_train) == 7700
    assert len(y_train) == 7700

    # Fit LogisticRegression using frozen parameters
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)

    coef_cnn = float(model.coef_[0][0])
    coef_m3 = float(model.coef_[0][1])
    intercept = float(model.intercept_[0])

    print(f"Corrected Fusion Model Fitted Successfully:")
    print(f"  Feature 0 (cnn_probability_oof) coefficient: {coef_cnn:.8f}")
    print(f"  Feature 1 (m3_probability_oof)  coefficient: {coef_m3:.8f}")
    print(f"  Intercept:                                  {intercept:.8f}")
    print(f"  Equation: P(forged) = sigmoid({coef_cnn:.6f} * cnn_prob + {coef_m3:.6f} * m3_prob + ({intercept:.6f}))")

    MODELS_FUSION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CORRECTED_MODEL_PATH)
    print(f"Saved corrected fusion model to: {CORRECTED_MODEL_PATH}")

    train_metadata = {
        "model_architecture": "LogisticRegression",
        "hyperparameters": {
            "C": 1.0,
            "solver": "lbfgs",
            "random_state": 42,
        },
        "feature_order": feature_names,
        "training_row_count": len(X_train),
        "target_mapping": {"genuine": 0, "forged": 1},
        "learned_parameters": {
            "cnn_probability_oof_weight": coef_cnn,
            "m3_probability_oof_weight": coef_m3,
            "intercept": intercept,
        },
        "sklearn_version": sklearn.__version__,
        "model_path": str(CORRECTED_MODEL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model_sha256": sha256_file(CORRECTED_MODEL_PATH),
    }

    return model, train_metadata


def step3_evaluate_oof_diagnostics(model: LogisticRegression, oof_df: pd.DataFrame) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATING OOF FUSION TRAINING DIAGNOSTICS (DIAGNOSTIC ONLY)")
    print("=" * 80)

    feature_names = ["cnn_probability_oof", "m3_probability_oof"]
    X_train = oof_df[feature_names].values
    y_train = (oof_df["final_label"] == "forged").astype(int).values

    train_probs = model.predict_proba(X_train)[:, 1]
    train_preds_05 = (train_probs >= 0.5).astype(int)

    cm = confusion_matrix(y_train, train_preds_05)
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_train, train_preds_05, zero_division=0))
    rec = float(recall_score(y_train, train_preds_05, zero_division=0))
    f1 = float(f1_score(y_train, train_preds_05, zero_division=0))
    acc = float(accuracy_score(y_train, train_preds_05))
    roc_auc = float(roc_auc_score(y_train, train_probs))
    pr_auc = float(average_precision_score(y_train, train_probs))

    diagnostics = {
        "status": "OOF fusion training diagnostics ONLY (NOT a generalization estimate)",
        "sample_count": len(y_train),
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "threshold_0_5": {
            "f1": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "specificity": round(spec, 6),
            "accuracy": round(acc, 6),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
    }

    print("OOF Training Diagnostics (Evaluated on the 7,700 OOF fitting rows at threshold 0.50):")
    print(f"  ROC-AUC:     {roc_auc:.4f}")
    print(f"  PR-AUC:      {pr_auc:.4f}")
    print(f"  F1 (t=0.5):  {f1:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return diagnostics


def step4_generate_val_upstream_predictions() -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("STEP 4: COLLATING VALIDATION UPSTREAM PREDICTIONS (CANONICAL val.csv)")
    print("=" * 80)

    val_split = pd.read_csv(VAL_SPLIT_PATH)
    val_align = pd.read_csv(VAL_ALIGN_PATH)
    cnn_val = pd.read_csv(CNN_VAL_PATH)

    assert len(val_split) == 1650, f"Expected 1650 val rows, got {len(val_split)}"
    assert len(val_align) == 1650, f"Expected 1650 val_align rows, got {len(val_align)}"
    assert len(cnn_val) == 1650, f"Expected 1650 cnn_val rows, got {len(cnn_val)}"

    assert (val_align["id"] == val_split["id"]).all()
    assert (val_align["record_id"] == val_split["record_id"]).all()
    assert (val_align["final_label"] == val_split["final_label"]).all()

    assert (cnn_val["id"] == val_split["id"]).all()
    # Ensure CNN probabilities match the deployment model inference
    max_cnn_diff = np.max(np.abs(val_align["cnn_probability"].values - cnn_val["cnn_probability"].values))
    assert max_cnn_diff < 1e-6, f"CNN probability discrepancy: {max_cnn_diff}"

    # Upstream validation probabilities
    cnn_prob_val = val_align["cnn_probability"].values
    m3_prob_val = val_align["m3_probability"].values

    assert not np.isnan(cnn_prob_val).any(), "NaN in cnn_prob_val"
    assert not np.isnan(m3_prob_val).any(), "NaN in m3_prob_val"
    assert (cnn_prob_val >= 0.0).all() and (cnn_prob_val <= 1.0).all()
    assert (m3_prob_val >= 0.0).all() and (m3_prob_val <= 1.0).all()

    target_val = (val_split["final_label"] == "forged").astype(int).values

    val_upstream_df = pd.DataFrame({
        "id": val_split["id"],
        "record_id": val_split["record_id"],
        "document_family": val_split["document_family"],
        "tamper_type": val_split["tamper_type"],
        "final_label": val_split["final_label"],
        "target": target_val,
        "cnn_probability_val": cnn_prob_val,
        "m3_probability_val": m3_prob_val,
    })

    val_upstream_df.to_csv(OUTPUT_VAL_UPSTREAM_PATH, index=False)
    print(f"Saved validation upstream predictions to: {OUTPUT_VAL_UPSTREAM_PATH}")
    return val_upstream_df


def step5_and_6_validation_inference_and_threshold_selection(
    model: LogisticRegression,
    val_df: pd.DataFrame,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    print("\n" + "=" * 80)
    print("STEP 5 & 6: VALIDATION FUSION INFERENCE & OPERATING THRESHOLD SELECTION")
    print("=" * 80)

    X_val = np.column_stack([val_df["cnn_probability_val"].values, val_df["m3_probability_val"].values])
    y_val = val_df["target"].values

    # Step 5: Inference
    val_probs = model.predict_proba(X_val)[:, 1]

    # Step 6: Threshold Selection
    # Grid: 0.01 to 0.99 inclusive
    threshold_grid = np.linspace(0.01, 0.99, 99)
    candidates: List[Dict[str, Any]] = []

    best_th = 0.50
    best_f1 = -1.0
    best_metrics: Dict[str, Any] = {}

    for th in threshold_grid:
        th_val = round(float(th), 4)
        pred = (val_probs >= th).astype(int)
        cm = confusion_matrix(y_val, pred)
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        if spec >= 0.90:
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            acc = float((tp + tn) / len(y_val))

            metric_entry = {
                "threshold": th_val,
                "f1": round(f1, 6),
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "specificity": round(spec, 6),
                "accuracy": round(acc, 6),
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
            }
            candidates.append(metric_entry)

            if f1 > best_f1:
                best_f1 = f1
                best_th = th_val
                best_metrics = metric_entry

    assert len(candidates) > 0, "No candidate threshold satisfied Specificity >= 0.90 on validation set!"
    print(f"Candidate thresholds satisfying Specificity >= 0.90: {len(candidates)}")
    print(f"Selected Operating Threshold: {best_th:.2f}")
    print(f"  Validation F1:          {best_metrics['f1']:.4f}")
    print(f"  Validation Precision:   {best_metrics['precision']:.4f}")
    print(f"  Validation Recall:      {best_metrics['recall']:.4f}")
    print(f"  Validation Specificity: {best_metrics['specificity']:.4f}")
    print(f"  Validation Accuracy:    {best_metrics['accuracy']:.4f}")
    print(f"  Confusion Matrix:       TN={best_metrics['confusion_matrix']['tn']}, "
          f"FP={best_metrics['confusion_matrix']['fp']}, "
          f"FN={best_metrics['confusion_matrix']['fn']}, "
          f"TP={best_metrics['confusion_matrix']['tp']}")

    # Attach predictions to validation dataframe
    val_out_df = val_df.copy()
    val_out_df["fusion_probability_val"] = np.round(val_probs, 8)
    val_out_df["fusion_prediction_val"] = (val_probs >= best_th).astype(int)
    val_out_df.to_csv(OUTPUT_VAL_PREDS_PATH, index=False)
    print(f"Saved validation fusion predictions to: {OUTPUT_VAL_PREDS_PATH}")

    val_summary = {
        "threshold_selection_protocol": "Grid search 0.01 to 0.99 step 0.01; Maximize F1 subject to Specificity >= 0.90",
        "total_thresholds_evaluated": len(threshold_grid),
        "candidate_thresholds_satisfying_constraint": len(candidates),
        "selected_threshold": best_th,
        "metrics_at_selected_threshold": best_metrics,
        "validation_samples": len(y_val),
        "validation_identities": int(val_df["record_id"].nunique()),
    }

    return best_th, val_summary, val_out_df


def step7_save_configuration(
    train_metadata: Dict[str, Any],
    diagnostics: Dict[str, Any],
    val_summary: Dict[str, Any],
) -> None:
    print("\n" + "=" * 80)
    print("STEP 7: SAVING CORRECTED FUSION CONFIGURATION & PROVENANCE")
    print("=" * 80)

    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 1B-B Corrected Fusion Training + Validation Threshold Selection",
        "model_architecture": "LogisticRegression",
        "feature_order": ["cnn_probability_oof", "m3_probability_oof"],
        "target_column": "final_label",
        "hyperparameters": train_metadata["hyperparameters"],
        "learned_parameters": train_metadata["learned_parameters"],
        "sklearn_version": train_metadata["sklearn_version"],
        "training_data_provenance": {
            "source_type": "Genuine 5-fold GroupKFold OOF predictions (100% held-out)",
            "aligned_oof_artifact": str(OUTPUT_OOF_ALIGNED_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "aligned_oof_sha256": sha256_file(OUTPUT_OOF_ALIGNED_PATH),
            "cnn_oof_source": str(CNN_OOF_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "cnn_oof_sha256": sha256_file(CNN_OOF_PATH),
            "m3_oof_source": str(M3_OOF_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "m3_oof_sha256": sha256_file(M3_OOF_PATH),
            "canonical_train_split": str(TRAIN_SPLIT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "canonical_train_sha256": sha256_file(TRAIN_SPLIT_PATH),
            "total_training_rows": 7700,
            "total_training_record_ids": 700,
        },
        "oof_training_diagnostics": diagnostics,
        "validation_threshold_selection": {
            "canonical_val_split": str(VAL_SPLIT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "canonical_val_sha256": sha256_file(VAL_SPLIT_PATH),
            "val_upstream_predictions_artifact": str(OUTPUT_VAL_UPSTREAM_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "val_upstream_predictions_sha256": sha256_file(OUTPUT_VAL_UPSTREAM_PATH),
            "threshold_selection_protocol": val_summary["threshold_selection_protocol"],
            "candidate_thresholds_satisfying_constraint": val_summary["candidate_thresholds_satisfying_constraint"],
            "selected_threshold": val_summary["selected_threshold"],
            "metrics_at_selected_threshold": val_summary["metrics_at_selected_threshold"],
        },
        "test_set_access_status": "STRICTLY UNACCESSED — Zero test rows loaded, fitted, or evaluated",
        "historical_artifacts_preserved": {
            "train_oof_alignment_csv": "outputs/fusion/train_oof_alignment.csv",
            "historical_fusion_model_joblib": "outputs/fusion/final_fusion_freeze/final_fusion_model.joblib",
            "historical_fusion_config_json": "outputs/fusion/final_fusion_freeze/final_fusion_config.json",
        },
    }

    with open(CORRECTED_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to: {CORRECTED_CONFIG_PATH}")

    OUTPUT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Mirrored configuration to: {OUTPUT_CONFIG_PATH}")


def main() -> None:
    print("Starting CrossVerify Phase 1B-B Corrected Fusion Pipeline...\n")
    oof_df = step1_align_oof_data()
    model, train_metadata = step2_train_corrected_fusion_model(oof_df)
    diagnostics = step3_evaluate_oof_diagnostics(model, oof_df)
    val_upstream_df = step4_generate_val_upstream_predictions()
    best_th, val_summary, val_out_df = step5_and_6_validation_inference_and_threshold_selection(model, val_upstream_df)
    step7_save_configuration(train_metadata, diagnostics, val_summary)
    print("\nPhase 1B-B Corrected Fusion Pipeline finished successfully.")


if __name__ == "__main__":
    main()
