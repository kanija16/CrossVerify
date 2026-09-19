"""
src/baseline/generate_c7_handoff.py

Member 2 Phase C7 — CNN Handoff Package Generation & Verification Script.

Executes:
1. Validation set inference using models/cnn/best_model.pt on val.csv.
   Generates models/cnn/val_predictions.csv and calculates official validation metrics.
2. Verification of test set metrics from existing models/cnn/test_predictions.csv.
   Asserts exact match with C6 reported metrics.
3. Attack-wise performance breakdown by tamper_type.
   Generates models/cnn/attack_wise_results.csv.
4. Family-wise performance breakdown by document family (Family A vs Family B).
   Generates models/cnn/family_wise_results.csv.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

sys.stdout.reconfigure(encoding='utf-8')

def log(msg: str = ""):
    print(msg, flush=True)

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baseline.dataset import VisualForensicsDataset, CNN_LABEL_MAP
from src.baseline.transforms import get_eval_transforms
from src.baseline.cnn import (
    ResNet18VisualForensics,
    CNNBaselineConfig,
    compute_pos_weight_from_csv,
    set_seed
)


def run_c7_handoff_generation(dataset_root: str) -> Dict[str, Any]:
    log("=" * 70)
    log("PHASE C7 — CNN BASELINE HANDOFF PACKAGE GENERATION")
    log("=" * 70)
    
    start_time = time.time()
    root = Path(dataset_root)
    config = CNNBaselineConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device used: {device}")

    ckpt_path = Path("models/cnn/best_model.pt")
    val_csv = root / "val.csv"
    train_csv = root / "train.csv"
    test_pred_csv = Path("models/cnn/test_predictions.csv")

    assert ckpt_path.exists(), f"Missing checkpoint file: {ckpt_path}"
    assert val_csv.exists(), f"Missing val.csv: {val_csv}"
    assert test_pred_csv.exists(), f"Missing test_predictions.csv: {test_pred_csv}"

    # =========================================================================
    # 1. Validation Predictions & Metrics Generation
    # =========================================================================
    log("\n--- Step 1: Running Validation Set Inference (val.csv) ---")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    log(f"✓ Loaded best checkpoint: Epoch {checkpoint.get('epoch', 'N/A')}, Best Val AUC = {checkpoint.get('val_auc', 0.0):.4f}")

    model = ResNet18VisualForensics(pretrained=False, dropout_rate=config.dropout_rate)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Calculate loss criterion for val loss reporting
    pos_w = compute_pos_weight_from_csv(train_csv)
    pos_weight_tensor = torch.tensor([pos_w], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    val_ds = VisualForensicsDataset(
        csv_file=val_csv,
        dataset_root=root,
        transform=get_eval_transforms(target_size=config.target_image_size),
        return_metadata=True,
        validate=True
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    val_logits = []
    val_probs = []
    val_targets = []
    val_metadata_rows = []
    running_val_loss = 0.0

    with torch.no_grad():
        for images, targets, batch_meta in val_loader:
            images = images.to(device)
            targets_tensor = targets.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, targets_tensor)
            running_val_loss += loss.item() * images.size(0)

            logits_flat = logits.squeeze(1)
            probs_flat = torch.sigmoid(logits_flat)

            val_logits.extend(logits_flat.cpu().numpy().tolist())
            val_probs.extend(probs_flat.cpu().numpy().tolist())
            val_targets.extend(targets.numpy().tolist())

            batch_size_cur = images.size(0)
            for i in range(batch_size_cur):
                val_metadata_rows.append({
                    "id": batch_meta["id"][i],
                    "record_id": batch_meta["record_id"][i],
                    "image_path": batch_meta["image_path"][i],
                    "final_label": batch_meta["final_label"][i],
                    "tamper_type": batch_meta["tamper_type"][i],
                    "cnn_label": batch_meta["cnn_label"][i]
                })

    val_loss = running_val_loss / len(val_ds)
    val_y_true = np.array(val_targets, dtype=int)
    val_y_prob = np.array(val_probs, dtype=float)
    val_y_pred = (val_y_prob >= 0.5).astype(int)

    # Save models/cnn/val_predictions.csv
    val_pred_df = pd.DataFrame(val_metadata_rows)
    val_pred_df["cnn_probability"] = val_y_prob
    val_pred_df["cnn_prediction"] = val_y_pred
    col_order = ["id", "record_id", "image_path", "cnn_probability", "cnn_prediction", "cnn_label", "final_label", "tamper_type"]
    val_pred_df = val_pred_df[col_order]
    
    val_pred_csv_path = Path("models/cnn/val_predictions.csv")
    val_pred_df.to_csv(val_pred_csv_path, index=False)
    log(f"✓ Saved validation predictions to: '{val_pred_csv_path}' ({len(val_pred_df)} rows)")

    # Calculate validation metrics
    val_acc = accuracy_score(val_y_true, val_y_pred)
    val_prec = precision_score(val_y_true, val_y_pred, zero_division=0)
    val_rec = recall_score(val_y_true, val_y_pred, zero_division=0)
    val_f1 = f1_score(val_y_true, val_y_pred, zero_division=0)
    val_roc_auc = roc_auc_score(val_y_true, val_y_prob)
    val_pr_auc = average_precision_score(val_y_true, val_y_prob)
    val_cm = confusion_matrix(val_y_true, val_y_pred)
    val_n_genuine = int(np.sum(val_y_true == 0))
    val_n_forged = int(np.sum(val_y_true == 1))

    log(f"Validation Set Summary (1,650 samples: Genuine={val_n_genuine}, Forged={val_n_forged}):")
    log(f"  Val Loss:    {val_loss:.4f}")
    log(f"  Accuracy:    {val_acc:.4f} ({val_acc * 100:.2f}%)")
    log(f"  Precision:   {val_prec:.4f}")
    log(f"  Recall:      {val_rec:.4f}")
    log(f"  F1-Score:    {val_f1:.4f}")
    log(f"  ROC-AUC:     {val_roc_auc:.4f}")
    log(f"  PR-AUC:      {val_pr_auc:.4f}")
    log(f"  Confusion Matrix: TN={val_cm[0,0]}, FP={val_cm[0,1]}, FN={val_cm[1,0]}, TP={val_cm[1,1]}")

    # =========================================================================
    # 2. Test Set Metrics Verification
    # =========================================================================
    log("\n--- Step 2: Verifying Test Set Metrics from test_predictions.csv ---")
    test_df = pd.read_csv(test_pred_csv)
    test_y_true = test_df["cnn_label"].map(CNN_LABEL_MAP).values
    test_y_prob = test_df["cnn_probability"].values
    test_y_pred = test_df["cnn_prediction"].values

    test_acc = accuracy_score(test_y_true, test_y_pred)
    test_prec = precision_score(test_y_true, test_y_pred, zero_division=0)
    test_rec = recall_score(test_y_true, test_y_pred, zero_division=0)
    test_f1 = f1_score(test_y_true, test_y_pred, zero_division=0)
    test_roc_auc = roc_auc_score(test_y_true, test_y_prob)
    test_pr_auc = average_precision_score(test_y_true, test_y_prob)
    test_cm = confusion_matrix(test_y_true, test_y_pred)
    test_n_genuine = int(np.sum(test_y_true == 0))
    test_n_forged = int(np.sum(test_y_true == 1))

    log(f"Test Metrics Verified:")
    log(f"  Accuracy:    {test_acc:.4f} (Expected: 0.9467)")
    log(f"  Precision:   {test_prec:.4f} (Expected: 0.8803)")
    log(f"  Recall:      {test_rec:.4f} (Expected: 0.9311)")
    log(f"  F1-Score:    {test_f1:.4f} (Expected: 0.9050)")
    log(f"  ROC-AUC:     {test_roc_auc:.4f} (Expected: 0.9817)")
    log(f"  PR-AUC:      {test_pr_auc:.4f} (Expected: 0.9723)")
    log(f"  Confusion Matrix: TN={test_cm[0,0]} (Exp: 1143), FP={test_cm[0,1]} (Exp: 57), FN={test_cm[1,0]} (Exp: 31), TP={test_cm[1,1]} (Exp: 419)")

    # Assert exact match with C6
    assert abs(test_acc - 0.9467) < 1e-3, f"Accuracy discrepancy: {test_acc} != 0.9467"
    assert abs(test_prec - 0.8803) < 1e-3, f"Precision discrepancy: {test_prec} != 0.8803"
    assert abs(test_rec - 0.9311) < 1e-3, f"Recall discrepancy: {test_rec} != 0.9311"
    assert abs(test_f1 - 0.9050) < 1e-3, f"F1 discrepancy: {test_f1} != 0.9050"
    assert abs(test_roc_auc - 0.9817) < 1e-3, f"ROC-AUC discrepancy: {test_roc_auc} != 0.9817"
    assert abs(test_pr_auc - 0.9723) < 1e-3, f"PR-AUC discrepancy: {test_pr_auc} != 0.9723"
    assert test_cm[0,0] == 1143 and test_cm[0,1] == 57 and test_cm[1,0] == 31 and test_cm[1,1] == 419, "Confusion matrix mismatch"
    log("✓ 100% EXPLICIT MATCH WITH C6 REPORTED TEST METRICS!")

    # =========================================================================
    # 3. Attack-Wise Results Generation
    # =========================================================================
    log("\n--- Step 3: Generating Attack-Wise Results ---")
    attack_rows = []
    
    for tamper_type, group in test_df.groupby("tamper_type"):
        g_y_true = group["cnn_label"].map(CNN_LABEL_MAP).values
        g_y_prob = group["cnn_probability"].values
        g_y_pred = group["cnn_prediction"].values

        n_samples = len(group)
        n_gen = int(np.sum(g_y_true == 0))
        n_forg = int(np.sum(g_y_true == 1))

        mean_prob = float(np.mean(g_y_prob))
        median_prob = float(np.median(g_y_prob))

        # Accuracy is computed against cnn_label target
        acc_val = accuracy_score(g_y_true, g_y_pred)
        
        # Precision, Recall, F1, ROC-AUC are calculated where meaningful
        if len(np.unique(g_y_true)) > 1:
            prec_val = precision_score(g_y_true, g_y_pred, zero_division=0)
            rec_val = recall_score(g_y_true, g_y_pred, zero_division=0)
            f1_val = f1_score(g_y_true, g_y_pred, zero_division=0)
            roc_auc_val = roc_auc_score(g_y_true, g_y_prob)
        else:
            # Class distribution is single-class (e.g. all cnn_label = genuine)
            prec_val = np.nan
            rec_val = np.nan
            f1_val = np.nan
            roc_auc_val = np.nan

        # Target classification note
        if tamper_type == "visual_splice":
            target_note = "Target Forgery Class (cnn_label=forged)"
        else:
            target_note = "Non-Visual Forgery Class (cnn_label=genuine)"

        attack_rows.append({
            "tamper_type": tamper_type,
            "cnn_target_role": target_note,
            "n_samples": n_samples,
            "n_genuine": n_gen,
            "n_forged": n_forg,
            "accuracy": round(acc_val, 4),
            "precision": round(prec_val, 4) if not np.isnan(prec_val) else None,
            "recall": round(rec_val, 4) if not np.isnan(rec_val) else None,
            "f1": round(f1_val, 4) if not np.isnan(f1_val) else None,
            "roc_auc": round(roc_auc_val, 4) if not np.isnan(roc_auc_val) else None,
            "mean_cnn_probability": round(mean_prob, 4),
            "median_cnn_probability": round(median_prob, 4)
        })

    attack_df = pd.DataFrame(attack_rows)
    attack_csv_path = Path("models/cnn/attack_wise_results.csv")
    attack_df.to_csv(attack_csv_path, index=False)
    log(f"✓ Saved attack-wise results to: '{attack_csv_path}' ({len(attack_df)} rows)")

    # Print Attack-wise table
    log("\nAttack-Wise Performance Table:")
    for _, r in attack_df.iterrows():
        log(f"  Tamper Type: {r['tamper_type']:<26} | Samples: {r['n_samples']:<3} | cnn_label(G/F): {r['n_genuine']}/{r['n_forged']} | Accuracy: {r['accuracy']:.4f} | Mean Prob: {r['mean_cnn_probability']:.4f} | Median Prob: {r['median_cnn_probability']:.4f}")

    # =========================================================================
    # 4. Family-Wise Results Generation
    # =========================================================================
    log("\n--- Step 4: Generating Family-Wise Results ---")
    family_rows = []
    
    test_df["family"] = test_df["id"].apply(lambda x: "Family A" if x.startswith("family_a") else "Family B")

    for family_name, group in test_df.groupby("family"):
        fam_y_true = group["cnn_label"].map(CNN_LABEL_MAP).values
        fam_y_prob = group["cnn_probability"].values
        fam_y_pred = group["cnn_prediction"].values

        fam_n_samples = len(group)
        fam_n_gen = int(np.sum(fam_y_true == 0))
        fam_n_forg = int(np.sum(fam_y_true == 1))

        fam_acc = accuracy_score(fam_y_true, fam_y_pred)
        fam_prec = precision_score(fam_y_true, fam_y_pred, zero_division=0)
        fam_rec = recall_score(fam_y_true, fam_y_pred, zero_division=0)
        fam_f1 = f1_score(fam_y_true, fam_y_pred, zero_division=0)
        fam_roc_auc = roc_auc_score(fam_y_true, fam_y_prob)
        fam_pr_auc = average_precision_score(fam_y_true, fam_y_prob)
        fam_mean_prob = float(np.mean(fam_y_prob))

        family_rows.append({
            "family_name": family_name,
            "n_samples": fam_n_samples,
            "n_genuine": fam_n_gen,
            "n_forged": fam_n_forg,
            "accuracy": round(fam_acc, 4),
            "precision": round(fam_prec, 4),
            "recall": round(fam_rec, 4),
            "f1": round(fam_f1, 4),
            "roc_auc": round(fam_roc_auc, 4),
            "pr_auc": round(fam_pr_auc, 4),
            "mean_cnn_probability": round(fam_mean_prob, 4)
        })

    family_df = pd.DataFrame(family_rows)
    family_csv_path = Path("models/cnn/family_wise_results.csv")
    family_df.to_csv(family_csv_path, index=False)
    log(f"✓ Saved family-wise results to: '{family_csv_path}' ({len(family_df)} rows)")

    log("\nFamily-Wise Performance Table:")
    for _, r in family_df.iterrows():
        log(f"  {r['family_name']}: Samples={r['n_samples']} (Genuine={r['n_genuine']}, Forged={r['n_forged']}) | Accuracy={r['accuracy']:.4f} | Precision={r['precision']:.4f} | Recall={r['recall']:.4f} | F1={r['f1']:.4f} | ROC-AUC={r['roc_auc']:.4f} | PR-AUC={r['pr_auc']:.4f} | Mean Prob={r['mean_cnn_probability']:.4f}")

    elapsed = time.time() - start_time
    log("\n" + "=" * 70)
    log("PHASE C7 HANDOFF SCRIPT COMPLETE")
    log(f"Total Execution Time: {elapsed:.2f} seconds")
    log("=" * 70)

    return {
        "val_metrics": {
            "loss": val_loss,
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "f1": val_f1,
            "roc_auc": val_roc_auc,
            "pr_auc": val_pr_auc,
            "confusion_matrix": val_cm.tolist(),
            "n_samples": len(val_ds),
            "n_genuine": val_n_genuine,
            "n_forged": val_n_forged
        },
        "test_metrics": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1": test_f1,
            "roc_auc": test_roc_auc,
            "pr_auc": test_pr_auc,
            "confusion_matrix": test_cm.tolist()
        },
        "attack_wise_rows": len(attack_df),
        "family_wise_rows": len(family_df)
    }


if __name__ == "__main__":
    default_dataset_root = os.environ.get(
        "DATASET_ROOT",
        r"C:\Users\jayadharshini\OneDrive\Desktop\dataset_final_v2\dataset_final_v2"
    )
    run_c7_handoff_generation(default_dataset_root)
