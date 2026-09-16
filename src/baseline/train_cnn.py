"""
src/baseline/train_cnn.py

Member 2 Phase C5 — Full CNN Baseline Model Training Routine for CrossVerify Visual Forensics.

Executes the full 15-epoch training and validation loop for ResNet18 visual forensics baseline.
Supports seamless resumption from saved checkpoint (models/cnn/best_model.pt) if interrupted.

Strictly enforces:
- Canonical train.csv and val.csv only (test.csv is 100% isolated and never accessed).
- cnn_label target semantics (genuine -> 0, forged -> 1).
- ResNet18 (ImageNet pretrained) + Dropout(p=0.2) + Linear(512, 1) producing raw single logits.
- Preprocessing: Aspect-preserving resize + center padding to 224x224, ToTensor, ImageNet normalization.
- Weighted BCEWithLogitsLoss with pos_weight = 2.6666667 (genuine/forged ratio on train.csv).
- AdamW optimizer (lr=1e-4, weight_decay=1e-2).
- CosineAnnealingLR scheduler (T_max=15).
- Early stopping monitoring validation ROC-AUC (patience=5, mode='max', delta=1e-4).
- Best model saved to models/cnn/best_model.pt based on validation ROC-AUC.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

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

from src.baseline.dataset import VisualForensicsDataset
from src.baseline.transforms import get_train_transforms, get_eval_transforms
from src.baseline.cnn import (
    ResNet18VisualForensics,
    CNNBaselineConfig,
    EarlyStopping,
    compute_pos_weight_from_csv,
    set_seed
)


def run_full_cnn_training(dataset_root: str, resume_if_checkpoint_exists: bool = True) -> None:
    log("=" * 70)
    log("PHASE C5 — FULL CNN BASELINE MODEL TRAINING (15 EPOCHS MAX)")
    log("=" * 70)
    
    start_time = time.time()
    
    # 1. Configuration & Reproducibility
    config = CNNBaselineConfig(batch_size=32, epochs=15)
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device used: {device}")
    log(f"Configured batch size: {config.batch_size}")
    log(f"Configured max epochs: {config.epochs}")
    log(f"Optimizer: AdamW (lr={config.learning_rate}, weight_decay={config.weight_decay})")
    log(f"Scheduler: CosineAnnealingLR (T_max={config.epochs})")
    log(f"Early Stopping: Metric='{config.early_stopping_metric}', Mode='{config.early_stopping_mode}', Patience={config.early_stopping_patience}, Delta={config.early_stopping_delta}")

    root = Path(dataset_root)
    train_csv = root / "train.csv"
    val_csv = root / "val.csv"
    test_csv = root / "test.csv"
    
    # Verify test.csv is NOT accessed
    log(f"Training split CSV: {train_csv.name}")
    log(f"Validation split CSV: {val_csv.name}")
    log(f"Test split CSV: ISOLATED (Not loaded)")

    # 2. Dataset & DataLoader Instantiation
    log("\n--- Instantiating DataLoaders ---")
    train_ds = VisualForensicsDataset(
        csv_file=train_csv,
        dataset_root=root,
        transform=get_train_transforms(target_size=config.target_image_size),
        validate=True
    )
    val_ds = VisualForensicsDataset(
        csv_file=val_csv,
        dataset_root=root,
        transform=get_eval_transforms(target_size=config.target_image_size),
        validate=True
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    log(f"Train Dataset Length: {len(train_ds)} samples ({n_train_batches} batches)")
    log(f"Val Dataset Length: {len(val_ds)} samples ({n_val_batches} batches)")

    # 3. Model, Loss, Optimizer & Scheduler Initialization
    log("\n--- Initializing Model, Loss & Optimizer ---")
    model = ResNet18VisualForensics(pretrained=config.pretrained, dropout_rate=config.dropout_rate)
    model = model.to(device)

    # Compute pos_weight from train.csv cnn_label counts
    pos_w = compute_pos_weight_from_csv(train_csv)
    log(f"Calculated pos_weight from train.csv: {pos_w:.7f}")
    pos_weight_tensor = torch.tensor([pos_w], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs
    )

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        metric=config.early_stopping_metric,
        mode=config.early_stopping_mode,
        delta=config.early_stopping_delta
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = Path(config.checkpoint_dir) / config.best_checkpoint_filename

    best_epoch = 0
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    start_epoch = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    val_auc = 0.0
    val_f1 = 0.0

    # Resume capability from checkpoint
    if resume_if_checkpoint_exists and checkpoint_path.exists():
        log(f"\n[RESUME] Checkpoint found at '{checkpoint_path}'. Loading saved state...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        saved_epoch = checkpoint["epoch"]
        best_epoch = saved_epoch
        best_val_auc = checkpoint["val_auc"]
        best_val_loss = checkpoint["val_loss"]
        best_val_f1 = checkpoint["val_f1"]
        early_stopping.best_score = best_val_auc
        start_epoch = saved_epoch + 1
        log(f"[RESUME] Restored state from Epoch {saved_epoch}: Best Val AUC = {best_val_auc:.4f}, Val Loss = {best_val_loss:.4f}, Val F1 = {best_val_f1:.4f}")
        log(f"[RESUME] Continuing training from Epoch {start_epoch}/{config.epochs}...")

    log("\n--- Starting Training Loop ---")
    final_epoch_run = start_epoch - 1
    
    for epoch in range(start_epoch, config.epochs + 1):
        final_epoch_run = epoch
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # --- Training Pass ---
        model.train()
        running_train_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.float().unsqueeze(1).to(device)  # Shape [batch, 1]

            optimizer.zero_grad()
            logits = model(images)
            
            # Safety Check: output shape
            assert logits.shape == targets.shape, f"Logits shape {logits.shape} != targets shape {targets.shape}"
            
            loss = criterion(logits, targets)
            assert not (torch.isnan(loss) or torch.isinf(loss)), f"NaN/Inf loss encountered at Epoch {epoch}, Batch {batch_idx}"

            loss.backward()

            # Safety Check: finite gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()), \
                        f"NaN/Inf gradient encountered in parameter '{name}' at Epoch {epoch}"

            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == n_train_batches:
                log(f"  Epoch {epoch:02d}/{config.epochs:02d} | Batch [{batch_idx + 1}/{n_train_batches}] - Current Batch Loss: {loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_ds)

        # --- Validation Pass ---
        model.eval()
        running_val_loss = 0.0
        val_probs_list = []
        val_targets_list = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets_tensor = targets.float().unsqueeze(1).to(device)

                logits = model(images)
                loss = criterion(logits, targets_tensor)
                running_val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                val_probs_list.append(probs.cpu().numpy())
                val_targets_list.append(targets.numpy())

        epoch_val_loss = running_val_loss / len(val_ds)
        all_val_probs = np.vstack(val_probs_list).flatten()
        all_val_targets = np.concatenate(val_targets_list).flatten()

        val_auc = roc_auc_score(all_val_targets, all_val_probs)
        val_preds = (all_val_probs >= 0.5).astype(int)
        val_f1 = f1_score(all_val_targets, val_preds)

        # --- Checkpointing & Early Stopping ---
        is_best = early_stopping.step(val_auc)
        checkpoint_flag = ""

        if is_best:
            best_epoch = epoch
            best_val_auc = val_auc
            best_val_loss = epoch_val_loss
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_auc": val_auc,
                    "val_loss": epoch_val_loss,
                    "val_f1": val_f1,
                    "config": config
                },
                checkpoint_path
            )
            checkpoint_flag = f" [NEW BEST -> Saved '{checkpoint_path.name}']"

        epoch_time = time.time() - epoch_start
        
        # Per-epoch progress line requirement
        log(
            f"\n>>> Epoch {epoch:02d}/{config.epochs:02d} Summary | "
            f"train_loss: {epoch_train_loss:.4f} | "
            f"val_loss: {epoch_val_loss:.4f} | "
            f"val_auc: {val_auc:.4f} | "
            f"val_f1: {val_f1:.4f} | "
            f"learning_rate: {current_lr:.6f} | "
            f"time: {epoch_time:.1f}s{checkpoint_flag}\n"
        )

        scheduler.step()

        if early_stopping.early_stop:
            log(f"\n! Early stopping triggered at Epoch {epoch} (No improvement in validation ROC-AUC for {config.early_stopping_patience} consecutive epochs).")
            break

    elapsed_time = time.time() - start_time
    early_stopped = early_stopping.early_stop

    log("\n" + "=" * 70)
    log("PHASE C5 FULL CNN TRAINING SUMMARY")
    log("=" * 70)
    log(f"Device Used:                 {device}")
    log(f"Total Epochs Run:            {final_epoch_run}/{config.epochs}")
    log(f"Early Stopping Occurred:     {early_stopped}")
    log(f"Best Epoch:                  {best_epoch}")
    log(f"Best Validation ROC-AUC:     {best_val_auc:.4f}")
    log(f"Corresponding Val Loss:      {best_val_loss:.4f}")
    log(f"Corresponding Val F1-Score:  {best_val_f1:.4f}")
    log(f"Final Epoch Train Loss:      {epoch_train_loss:.4f}")
    log(f"Final Epoch Validation Loss: {epoch_val_loss:.4f}")
    log(f"Final Epoch Val ROC-AUC:     {val_auc:.4f}")
    log(f"Final Epoch Val F1-Score:    {val_f1:.4f}")
    log(f"Checkpoint Path:             {checkpoint_path}")
    log(f"Total Session Time:          {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    log("=" * 70)


if __name__ == "__main__":
    default_dataset_root = os.environ.get(
        "DATASET_ROOT",
        r"C:\Users\jayadharshini\OneDrive\Desktop\dataset_final_v2\dataset_final_v2"
    )
    run_full_cnn_training(default_dataset_root)
