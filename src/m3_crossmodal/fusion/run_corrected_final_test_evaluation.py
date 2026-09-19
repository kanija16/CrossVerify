"""
CrossVerify Phase 1B-C: Locked Corrected Fusion Final Test Evaluation Engine.

Executes:
1. Canonical test split validation and identity disjointness verification against train/val.
2. Verification of frozen Member 2 CNN upstream predictions (models/cnn/test_predictions.csv).
   Exports aligned raw working copy to outputs/fusion/cnn_probability_test.csv.
3. Clean scikit-learn 1.6.1 inference of frozen Member 3 model (final_m3_model.joblib)
   on outputs/features/test_features.csv using exact 15-feature schema.
4. Corrected late-fusion inference using frozen LogisticRegression model
   (models/fusion/corrected_fusion_model.joblib) with frozen threshold t = 0.82.
5. Official metrics generation:
   - Overall metrics & confusion matrix
   - Attack-wise performance breakdown across all canonical tamper types
   - Family-wise performance breakdown (Family A vs Family B)
6. Export of official corrected final test artifacts with cryptographic provenance.

Zero retraining, zero threshold tuning, zero model changes.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import sklearn
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

# Input paths
TEST_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
TRAIN_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"

M2_TEST_PREDS_PATH = PROJECT_ROOT / "models" / "cnn" / "test_predictions.csv"
M3_MODEL_PATH = PROJECT_ROOT / "outputs" / "final_m3_model_freeze" / "final_m3_model.joblib"
M3_SCHEMA_PATH = PROJECT_ROOT / "outputs" / "final_m3_model_freeze" / "final_feature_schema.json"
M3_FEATURES_PATH = PROJECT_ROOT / "outputs" / "features" / "test_features.csv"

FUSION_MODEL_PATH = PROJECT_ROOT / "models" / "fusion" / "corrected_fusion_model.joblib"
FUSION_CONFIG_PATH = PROJECT_ROOT / "models" / "fusion" / "corrected_fusion_config.json"

# Output paths
DIR_FUSION = PROJECT_ROOT / "outputs" / "fusion"
OUT_CNN_PROB_PATH = DIR_FUSION / "cnn_probability_test.csv"
OUT_TEST_PREDS_PATH = DIR_FUSION / "corrected_fusion_test_predictions.csv"
OUT_TEST_METRICS_PATH = DIR_FUSION / "corrected_fusion_test_metrics.json"
OUT_TEST_ATTACK_PATH = DIR_FUSION / "corrected_fusion_test_attack_wise.csv"
OUT_TEST_FAMILY_PATH = DIR_FUSION / "corrected_fusion_test_family_wise.csv"

# Expected invariant hashes
EXPECTED_CNN_TEST_HASH = "fdaafab5e7b2477164b2b64709af2d44b17d0f23ff33c9f8c8e4877338b5e77b"
EXPECTED_M3_MODEL_HASH = "a0d3dd8a4c544ecd307e0926addda7b0b4d3a187138c13307cd582d6500d55f3"
EXPECTED_FUSION_MODEL_HASH = "73931e2fa90b84f83b8d61348aeed9729a802bf52c0e80704f62f5f13a5281e2"
FROZEN_THRESHOLD = 0.82


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_locked_test_evaluation() -> None:
    print("=" * 80)
    print("CROSSVERIFY PHASE 1B-C: LOCKED CORRECTED FUSION FINAL TEST EVALUATION")
    print("=" * 80)
    start_time = datetime.now(timezone.utc)
    print(f"Execution started at: {start_time.isoformat()}")
    print(f"Python version: {sys.version}")
    print(f"scikit-learn version: {sklearn.__version__}")

    # =========================================================================
    # STEP 1: CANONICAL DATA INTEGRITY & IDENTITY ISOLATION ASSERTIONS
    # =========================================================================
    print("\n--- Step 1: Verifying Canonical Test Split & Identity Disjointness ---")
    assert TEST_SPLIT_PATH.exists(), f"Missing canonical test split: {TEST_SPLIT_PATH}"
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    n_test = len(test_df)
    assert n_test == 1650, f"Expected 1,650 test rows, got {n_test}"

    unique_test_ids = int(test_df["id"].nunique())
    assert unique_test_ids == 1650, f"Duplicate sample IDs in test.csv: {unique_test_ids}"

    unique_test_records = int(test_df["record_id"].nunique())
    assert unique_test_records == 150, f"Expected 150 unique record IDs, got {unique_test_records}"
    assert (test_df["record_id"].value_counts() == 11).all(), "Every record_id must have exactly 11 samples"

    # Verify identity disjointness with train and val
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    val_df = pd.read_csv(VAL_SPLIT_PATH)
    s_train = set(train_df["record_id"])
    s_val = set(val_df["record_id"])
    s_test = set(test_df["record_id"])

    train_overlap = s_train.intersection(s_test)
    val_overlap = s_val.intersection(s_test)
    assert len(train_overlap) == 0, f"CRITICAL LEAKAGE: {len(train_overlap)} record_ids overlap with train.csv!"
    assert len(val_overlap) == 0, f"CRITICAL LEAKAGE: {len(val_overlap)} record_ids overlap with val.csv!"
    print("[PASS] Verified test split: 1,650 rows, 150 unique record_ids, 0 train/val overlap.")

    # Ground truth targets
    y_true = (test_df["final_label"] == "forged").astype(int).values
    n_genuine = int(np.sum(y_true == 0))
    n_forged = int(np.sum(y_true == 1))
    assert n_genuine == 150, f"Expected 150 genuine test samples, got {n_genuine}"
    assert n_forged == 1500, f"Expected 1,500 forged test samples, got {n_forged}"
    print(f"Target distribution: Genuine = {n_genuine}, Forged = {n_forged}")

    # =========================================================================
    # STEP 2: VERIFY & INGEST CNN UPSTREAM TEST PROBABILITIES
    # =========================================================================
    print("\n--- Step 2: Verifying CNN Upstream Test Predictions ---")
    assert M2_TEST_PREDS_PATH.exists(), f"Missing CNN test predictions: {M2_TEST_PREDS_PATH}"
    actual_cnn_hash = sha256_file(M2_TEST_PREDS_PATH)
    assert actual_cnn_hash == EXPECTED_CNN_TEST_HASH, f"CNN test predictions hash mismatch: {actual_cnn_hash}"
    print(f"[PASS] CNN test predictions hash verified: {actual_cnn_hash}")

    cnn_test_df = pd.read_csv(M2_TEST_PREDS_PATH)
    assert len(cnn_test_df) == 1650, f"Expected 1,650 CNN rows, got {len(cnn_test_df)}"
    assert (cnn_test_df["id"] == test_df["id"]).all(), "1:1 ID alignment mismatch between test.csv and CNN predictions!"
    assert (cnn_test_df["record_id"] == test_df["record_id"]).all(), "record_id mismatch with CNN predictions!"
    assert cnn_test_df["cnn_probability"].isnull().sum() == 0, "Missing values in cnn_probability!"

    p_cnn = cnn_test_df["cnn_probability"].values
    assert (p_cnn >= 0.0).all() and (p_cnn <= 1.0).all(), "CNN probabilities outside [0, 1]!"
    print(f"CNN probability range: [{p_cnn.min():.8f}, {p_cnn.max():.8f}], mean = {p_cnn.mean():.4f}")

    # Save working copy to outputs/fusion/cnn_probability_test.csv without modifying original
    DIR_FUSION.mkdir(parents=True, exist_ok=True)
    cnn_test_df.to_csv(OUT_CNN_PROB_PATH, index=False)
    print(f"Saved working raw CNN probabilities to: {OUT_CNN_PROB_PATH}")

    # =========================================================================
    # STEP 3: M3 INFERENCE USING CLEAN SKLEARN 1.6.1 ENVIRONMENT
    # =========================================================================
    print("\n--- Step 3: Running Frozen M3 Inference ---")
    assert M3_MODEL_PATH.exists(), f"Missing M3 model: {M3_MODEL_PATH}"
    actual_m3_hash = sha256_file(M3_MODEL_PATH)
    assert actual_m3_hash == EXPECTED_M3_MODEL_HASH, f"M3 model hash mismatch: {actual_m3_hash}"
    print(f"[PASS] M3 model hash verified: {actual_m3_hash}")

    with open(M3_SCHEMA_PATH) as f:
        m3_schema = json.load(f)
    feature_names = m3_schema["feature_order"]
    assert len(feature_names) == 15, f"Expected 15 features, got {len(feature_names)}"
    print(f"M3 15-feature schema: {feature_names}")

    assert M3_FEATURES_PATH.exists(), f"Missing test features: {M3_FEATURES_PATH}"
    test_features_df = pd.read_csv(M3_FEATURES_PATH)
    assert len(test_features_df) == 1650, f"Expected 1,650 test feature rows, got {len(test_features_df)}"
    assert (test_features_df["id"] == test_df["id"]).all(), "1:1 ID mismatch with test_features.csv!"
    assert (test_features_df["record_id"] == test_df["record_id"]).all(), "record_id mismatch with test_features.csv!"

    m3_model = joblib.load(M3_MODEL_PATH)
    X_m3 = test_features_df[feature_names]
    p_m3 = m3_model.predict_proba(X_m3)[:, 1]
    assert not np.isnan(p_m3).any(), "NaN in M3 predicted probabilities!"
    assert (p_m3 >= 0.0).all() and (p_m3 <= 1.0).all(), "M3 probabilities outside [0, 1]!"
    print(f"M3 probability range: [{p_m3.min():.8f}, {p_m3.max():.8f}], mean = {p_m3.mean():.4f}")

    # =========================================================================
    # STEP 4: CORRECTED FUSION INFERENCE (FROZEN WEIGHTS & THRESHOLD)
    # =========================================================================
    print("\n--- Step 4: Computing Corrected Fusion Probabilities & Predictions ---")
    assert FUSION_MODEL_PATH.exists(), f"Missing corrected fusion model: {FUSION_MODEL_PATH}"
    actual_fusion_hash = sha256_file(FUSION_MODEL_PATH)
    assert actual_fusion_hash == EXPECTED_FUSION_MODEL_HASH, f"Fusion model hash mismatch: {actual_fusion_hash}"
    print(f"[PASS] Corrected fusion model hash verified: {actual_fusion_hash}")

    fusion_model = joblib.load(FUSION_MODEL_PATH)
    assert fusion_model.C == 1.0
    assert fusion_model.solver == "lbfgs"
    assert fusion_model.random_state == 42

    # Scikit-learn 1.6.1 compatibility shim for models saved with sklearn >= 1.7
    if not hasattr(fusion_model, "multi_class"):
        fusion_model.multi_class = "auto"

    coef_cnn = float(fusion_model.coef_[0][0])
    coef_m3 = float(fusion_model.coef_[0][1])
    intercept = float(fusion_model.intercept_[0])
    print(f"Loaded Fusion Weights: CNN = {coef_cnn:.8f}, M3 = {coef_m3:.8f}, Intercept = {intercept:.8f}")

    # Input feature vector
    X_fusion = np.column_stack([p_cnn, p_m3])
    p_fusion = fusion_model.predict_proba(X_fusion)[:, 1]

    # Verify formula equivalence: p = sigmoid(w1*x1 + w2*x2 + b)
    z = coef_cnn * p_cnn + coef_m3 * p_m3 + intercept
    p_formula = 1.0 / (1.0 + np.exp(-z))
    assert np.allclose(p_fusion, p_formula, atol=1e-7), "Discrepancy between model and formula probabilities!"

    # Binary decision rule at frozen threshold = 0.82
    final_pred = (p_fusion >= FROZEN_THRESHOLD).astype(int)
    print(f"Applied frozen threshold t = {FROZEN_THRESHOLD:.2f}: "
          f"Predicted {int(np.sum(final_pred == 1))} forged, {int(np.sum(final_pred == 0))} genuine.")

    # =========================================================================
    # STEP 5: COMPUTE OFFICIAL EVALUATION METRICS
    # =========================================================================
    print("\n--- Step 5: Calculating Official Test Metrics ---")
    acc = float(accuracy_score(y_true, final_pred))
    prec = float(precision_score(y_true, final_pred, zero_division=0))
    rec = float(recall_score(y_true, final_pred, zero_division=0))
    f1 = float(f1_score(y_true, final_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, p_fusion))
    pr_auc = float(average_precision_score(y_true, p_fusion))

    cm = confusion_matrix(y_true, final_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print("=" * 60)
    print("FINAL TEST EVALUATION METRICS (OVERALL)")
    print("=" * 60)
    print(f"Total Samples: {n_test} (Genuine: {n_genuine}, Forged: {n_forged})")
    print(f"Operating Threshold: {FROZEN_THRESHOLD:.2f}")
    print(f"Accuracy:    {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision:   {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall:      {rec:.4f} ({rec * 100:.2f}%)")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Specificity: {spec:.4f} ({spec * 100:.2f}%)")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN = {tn:4d}  |  FP = {fp:4d}")
    print(f"  FN = {fn:4d}  |  TP = {tp:4d}")
    print("=" * 60)

    # Attack-wise breakdown
    test_out_df = pd.DataFrame({
        "id": test_df["id"],
        "document_family": test_df["document_family"],
        "record_id": test_df["record_id"],
        "template_variant": test_df["template_variant"],
        "tamper_type": test_df["tamper_type"],
        "final_label": test_df["final_label"],
        "cnn_label": test_df["cnn_label"],
        "cnn_probability": np.round(p_cnn, 8),
        "m3_probability": np.round(p_m3, 8),
        "fusion_probability": np.round(p_fusion, 8),
        "final_prediction": final_pred,
        "threshold": FROZEN_THRESHOLD,
    })

    attack_rows = []
    print("\nAttack-Wise Results Breakdown:")
    print(f"{'Tamper Type':<28} {'Role':<12} {'Metric':<12} {'Samples':<8} {'Correct':<8} {'Rate':<8}")
    print("-" * 76)
    for tt, grp in test_out_df.groupby("tamper_type"):
        is_gen = (tt == "genuine")
        target_val = 0 if is_gen else 1
        metric_name = "specificity" if is_gen else "recall"
        correct = int((grp["final_prediction"] == target_val).sum())
        total_s = len(grp)
        rate = float(correct / total_s)
        mean_prob = float(grp["fusion_probability"].mean())
        attack_rows.append({
            "tamper_type": tt,
            "role": "genuine" if is_gen else "forged_attack",
            "metric_name": metric_name,
            "total_samples": total_s,
            "correct_predictions": correct,
            "rate": round(rate, 4),
            "mean_fusion_probability": round(mean_prob, 6),
        })
        print(f"{tt:<28} {'genuine' if is_gen else 'forged':<12} {metric_name:<12} {total_s:<8} {correct:<8} {rate * 100:.2f}%")

    df_attack = pd.DataFrame(attack_rows)

    # Family-wise breakdown
    family_rows = []
    print("\nFamily-Wise Results Breakdown:")
    for fam, grp in test_out_df.groupby("document_family"):
        fam_y = (grp["final_label"] == "forged").astype(int).values
        fam_pred = grp["final_prediction"].values
        fam_prob = grp["fusion_probability"].values
        fam_cm = confusion_matrix(fam_y, fam_pred)
        f_tn, f_fp, f_fn, f_tp = [int(v) for v in fam_cm.ravel()]
        f_spec = float(f_tn / (f_tn + f_fp)) if (f_tn + f_fp) > 0 else 0.0
        f_acc = float(accuracy_score(fam_y, fam_pred))
        f_prec = float(precision_score(fam_y, fam_pred, zero_division=0))
        f_rec = float(recall_score(fam_y, fam_pred, zero_division=0))
        f_f1 = float(f1_score(fam_y, fam_pred, zero_division=0))
        family_rows.append({
            "document_family": fam,
            "samples": len(grp),
            "genuine_samples": int(np.sum(fam_y == 0)),
            "forged_samples": int(np.sum(fam_y == 1)),
            "accuracy": round(f_acc, 4),
            "precision": round(f_prec, 4),
            "recall": round(f_rec, 4),
            "specificity": round(f_spec, 4),
            "f1_score": round(f_f1, 4),
            "confusion_matrix": {"tn": f_tn, "fp": f_fp, "fn": f_fn, "tp": f_tp},
        })
        print(f"Family: {fam} (N={len(grp)}) | Acc: {f_acc:.4f} | F1: {f_f1:.4f} | Rec: {f_rec:.4f} | Spec: {f_spec:.4f}")

    df_family = pd.DataFrame(family_rows)

    # =========================================================================
    # STEP 6: EXPORT OFFICIAL ARTIFACTS
    # =========================================================================
    print("\n--- Step 6: Exporting Official Corrected Final Test Artifacts ---")

    # 1. Predictions CSV
    test_out_df.to_csv(OUT_TEST_PREDS_PATH, index=False)
    print(f"Saved predictions CSV: {OUT_TEST_PREDS_PATH}")

    # 2. Attack-wise CSV
    df_attack.to_csv(OUT_TEST_ATTACK_PATH, index=False)
    print(f"Saved attack-wise CSV: {OUT_TEST_ATTACK_PATH}")

    # 3. Family-wise CSV
    df_family.to_csv(OUT_TEST_FAMILY_PATH, index=False)
    print(f"Saved family-wise CSV: {OUT_TEST_FAMILY_PATH}")

    # 4. Metrics JSON
    test_metrics_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 1B-C Locked Corrected Fusion Final Test Evaluation",
        "sample_counts": {
            "total": n_test,
            "genuine": n_genuine,
            "forged": n_forged,
            "unique_record_ids": unique_test_records,
        },
        "operating_threshold": FROZEN_THRESHOLD,
        "classification_metrics": {
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
            "specificity": round(spec, 6),
        },
        "discrimination_metrics": {
            "roc_auc": round(roc_auc, 6),
            "pr_auc": round(pr_auc, 6),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "model_provenance": {
            "m2_cnn_test_predictions_sha256": actual_cnn_hash,
            "m3_model_sha256": actual_m3_hash,
            "corrected_fusion_model_sha256": actual_fusion_hash,
            "test_split_sha256": sha256_file(TEST_SPLIT_PATH),
        },
        "historical_comparison_note": "Corrected fusion model trained strictly on 7,700 genuine 5-fold OOF predictions. Historical in-sample fusion result remains available for methodological comparison only.",
    }

    with open(OUT_TEST_METRICS_PATH, "w") as f:
        json.dump(test_metrics_data, f, indent=2)
    print(f"Saved metrics JSON: {OUT_TEST_METRICS_PATH}")

    print("\nPhase 1B-C Locked Test Evaluation completed successfully.")


if __name__ == "__main__":
    run_locked_test_evaluation()
