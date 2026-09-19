"""
src/baseline/eval_cnn_test.py

Member 2 Phase C6 — CNN Baseline Model Final Test Evaluation Script.

Evaluates the approved ResNet18 CNN baseline model checkpoint (models/cnn/best_model.pt)
on the canonical test.csv split (1,650 samples).

Strictly enforces:
1. Checkpoint loading: Restores exact ResNet18 architecture from models/cnn/best_model.pt.
2. Canonical ground truth: Reads cnn_label directly (genuine -> 0, forged -> 1).
3. Image-only model input: RGB [3, 224, 224] image tensors ONLY. Zero metadata features.
4. Preprocessing: Aspect-preserving resize + center padding to 224x224, ToTensor, ImageNet normalization.
5. Primary metrics evaluated at threshold = 0.5 (NO threshold tuning on test data):
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
   - PR-AUC (Average Precision)
   - Confusion Matrix (TN, FP, FN, TP)
   - Total test sample count & class counts
6. Prediction CSV generation:
   Saves predictions to models/cnn/test_predictions.csv containing:
   [id, record_id, image_path, cnn_probability, cnn_prediction, cnn_label, final_label, tamper_type]
7. Test Set Isolation: Executed ONLY during final evaluation pass under model.eval() and torch.no_grad().
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
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
from torch.utils.data import DataLoader

from src.baseline.dataset import VisualForensicsDataset, CNN_LABEL_MAP
from src.baseline.transforms import get_eval_transforms
from src.baseline.cnn import ResNet18VisualForensics, CNNBaselineConfig, set_seed


def run_cnn_test_evaluation(
    dataset_root: str,
    checkpoint_path: str = "models/cnn/best_model.pt",
    output_prediction_csv: str = "models/cnn/test_predictions.csv"
) -> Dict[str, Any]:
    """
    Executes final CNN baseline evaluation on canonical test.csv.
    
    Args:
        dataset_root: Path to root dataset directory.
        checkpoint_path: Path to best model checkpoint.
        output_prediction_csv: Output CSV path for test predictions.

    Returns:
        Dict containing all calculated evaluation metrics and metadata summary.
    """
    log("=" * 70)
    log("PHASE C6 — CNN BASELINE MODEL FINAL TEST EVALUATION")
    log("=" * 70)
    
    start_time = time.time()
    
    # 1. Device & Reproducibility Setup
    config = CNNBaselineConfig()
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device used for inference: {device}")
    log(f"Target Checkpoint:        {checkpoint_path}")
    
    root = Path(dataset_root)
    test_csv = root / "test.csv"
    ckpt_file = Path(checkpoint_path)
    
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_file}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Canonical test.csv not found at: {test_csv}")

    # 2. Load Checkpoint and Restore Architecture
    log("\n--- Loading Checkpoint & Restoring Model ---")
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    log(f"✓ Checkpoint loaded successfully.")
    log(f"  Saved Checkpoint Epoch:   {checkpoint.get('epoch', 'N/A')}")
    log(f"  Best Val ROC-AUC Saved:   {checkpoint.get('val_auc', 0.0):.4f}")
    log(f"  Best Val Loss Saved:      {checkpoint.get('val_loss', 0.0):.4f}")
    log(f"  Best Val F1-Score Saved:  {checkpoint.get('val_f1', 0.0):.4f}")

    model = ResNet18VisualForensics(pretrained=False, dropout_rate=config.dropout_rate)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    log("✓ ResNet18VisualForensics architecture restored and set to eval() mode.")

    # 3. Instantiate Test DataLoader (return_metadata=True for prediction output joining)
    log("\n--- Instantiating Test DataLoader ---")
    test_ds = VisualForensicsDataset(
        csv_file=test_csv,
        dataset_root=root,
        transform=get_eval_transforms(target_size=config.target_image_size),
        return_metadata=True,
        validate=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    n_test_samples = len(test_ds)
    n_test_batches = len(test_loader)
    log(f"✓ Test Dataset Length: {n_test_samples} samples ({n_test_batches} batches)")

    # 4. Evaluation Loop (torch.no_grad, zero gradient computation)
    log("\n--- Running Inference Pass on Test Set ---")
    all_logits = []
    all_probs = []
    all_targets = []
    metadata_rows = []

    with torch.no_grad():
        for batch_idx, (images, targets, batch_meta) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images).squeeze(1)  # Shape [B]
            probs = torch.sigmoid(logits)

            all_logits.extend(logits.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())

            # Batch metadata extraction for predictions CSV
            batch_size_cur = images.size(0)
            for i in range(batch_size_cur):
                metadata_rows.append({
                    "id": batch_meta["id"][i],
                    "record_id": batch_meta["record_id"][i],
                    "image_path": batch_meta["image_path"][i],
                    "final_label": batch_meta["final_label"][i],
                    "tamper_type": batch_meta["tamper_type"][i],
                    "cnn_label": batch_meta["cnn_label"][i]
                })

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)  # Fixed primary threshold = 0.5

    # 5. Metric Calculation
    log("\n--- Calculating Final Test Metrics ---")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)  # [[TN, FP], [FN, TP]]

    n_genuine = int(np.sum(y_true == 0))
    n_forged = int(np.sum(y_true == 1))

    # 6. Generate Prediction CSV
    log("\n--- Generating Predictions CSV ---")
    pred_df = pd.DataFrame(metadata_rows)
    pred_df["cnn_probability"] = y_prob
    pred_df["cnn_prediction"] = y_pred

    # Ensure column ordering matches requirement
    col_order = [
        "id",
        "record_id",
        "image_path",
        "cnn_probability",
        "cnn_prediction",
        "cnn_label",
        "final_label",
        "tamper_type"
    ]
    pred_df = pred_df[col_order]

    os.makedirs(Path(output_prediction_csv).parent, exist_ok=True)
    pred_df.to_csv(output_prediction_csv, index=False)
    log(f"✓ Predictions CSV successfully saved to: '{output_prediction_csv}' ({len(pred_df)} rows)")

    elapsed_time = time.time() - start_time

    # 7. Summary Report Output
    log("\n" + "=" * 70)
    log("PHASE C6 CNN BASELINE TEST EVALUATION RESULTS")
    log("=" * 70)
    log(f"Target Checkpoint:       {checkpoint_path}")
    log(f"Total Test Samples:      {n_test_samples}")
    log(f"Class Distribution:      Genuine = {n_genuine}, Forged = {n_forged}")
    log(f"Classification Threshold: 0.50 (Fixed, No threshold tuning on test)")
    log("-" * 70)
    log(f"Accuracy:                {acc:.4f} ({acc * 100:.2f}%)")
    log(f"Precision:               {prec:.4f}")
    log(f"Recall:                  {rec:.4f}")
    log(f"F1-Score:                {f1:.4f}")
    log(f"ROC-AUC:                 {roc_auc:.4f}")
    log(f"PR-AUC (Avg Precision):  {pr_auc:.4f}")
    log("-" * 70)
    log("Confusion Matrix:")
    log(f"  TN (Genuine -> Genuine): {cm[0, 0]}")
    log(f"  FP (Genuine -> Forged):  {cm[0, 1]}")
    log(f"  FN (Forged -> Genuine):  {cm[1, 0]}")
    log(f"  TP (Forged -> Forged):   {cm[1, 1]}")
    log("=" * 70)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "n_test_samples": n_test_samples,
        "n_genuine": n_genuine,
        "n_forged": n_forged,
        "output_prediction_csv": str(output_prediction_csv),
        "execution_time_seconds": elapsed_time
    }


if __name__ == "__main__":
    default_dataset_root = os.environ.get(
        "DATASET_ROOT",
        r"C:\Users\jayadharshini\OneDrive\Desktop\dataset_final_v2\dataset_final_v2"
    )
    run_cnn_test_evaluation(default_dataset_root)
