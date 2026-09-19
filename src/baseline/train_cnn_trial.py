"""
src/baseline/train_cnn_trial.py

Member 2 Phase C4 — Tiny 1-Epoch CNN Pipeline Verification Trial.

Executes a 1-epoch training and validation pipeline check for the ResNet18 visual forensics baseline.
Strictly verifies:
- DataLoader batch creation and device tensor movement.
- Forward pass, BCEWithLogitsLoss evaluation, backpropagation, and AdamW optimizer step.
- CosineAnnealingLR scheduler single step.
- Validation pass, ROC-AUC, and F1 score computation.
- Early stopping & best checkpoint saving to models/cnn/best_model.pt based ONLY on validation ROC-AUC.
- Absolute isolation of test.csv (never loaded or evaluated).
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


def run_c4_training_trial(dataset_root: str) -> None:
    log("=" * 70)
    log("PHASE C4 — TINY CNN TRAINING PIPELINE TRIAL (1 EPOCH ONLY)")
    log("=" * 70)
    
    start_time = time.time()
    
    # 1. Configuration & Reproducibility
    config = CNNBaselineConfig(batch_size=32, epochs=15)  # T_max=15 tied to full budget
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device used: {device}")
    log(f"Configured batch size: {config.batch_size}")
    log(f"Configured total budget epochs (T_max): {config.epochs}")
    log(f"Trial execution limit: 1 EPOCH ONLY")

    root = Path(dataset_root)
    train_csv = root / "train.csv"
    val_csv = root / "val.csv"
    
    # Verify test.csv is NOT accessed
    log(f"Training split CSV: {train_csv.name}")
    log(f"Validation split CSV: {val_csv.name}")

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

    # Verify first batch shape
    first_images, first_targets = next(iter(train_loader))
    log(f"✓ DataLoader first batch image shape: {list(first_images.shape)}")
    log(f"✓ DataLoader first batch target shape: {list(first_targets.shape)}")
    assert first_images.shape == torch.Size([config.batch_size, 3, 224, 224])
    assert first_targets.shape == torch.Size([config.batch_size])

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

    # 4. Training Pass (Epoch 1 Only)
    log("\n--- Running Training Epoch 1/1 ---")
    model.train()
    running_train_loss = 0.0
    grads_verified = False

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.float().unsqueeze(1).to(device)  # Shape [batch, 1]

        optimizer.zero_grad()
        logits = model(images)
        
        # Verify output shape
        assert logits.shape == targets.shape, f"Logits shape {logits.shape} != targets shape {targets.shape}"
        
        loss = criterion(logits, targets)
        assert not torch.isnan(loss), f"NaN loss encountered at batch {batch_idx}"

        loss.backward()

        # Sanity check gradients exist after backward
        if not grads_verified:
            grad_norms = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
            assert len(grad_norms) > 0 and sum(grad_norms) > 0.0, "Gradients are zero or None after backward()"
            log(f"✓ Gradients verified after backward pass (Total param grad norm sum = {sum(grad_norms):.4f})")
            grads_verified = True

        optimizer.step()
        running_train_loss += loss.item() * images.size(0)

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == n_train_batches:
            log(f"  Batch [{batch_idx + 1}/{n_train_batches}] - Current Batch Loss: {loss.item():.4f}")

    epoch_train_loss = running_train_loss / len(train_ds)
    log(f"✓ Training Epoch 1 Complete - Mean Train Loss: {epoch_train_loss:.4f}")

    # 5. Validation Pass
    log("\n--- Running Validation Pass ---")
    model.eval()
    running_val_loss = 0.0
    val_probs_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
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

    log(f"✓ Validation Pass Complete:")
    log(f"  Mean Validation Loss: {epoch_val_loss:.4f}")
    log(f"  Validation ROC-AUC:   {val_auc:.4f}")
    log(f"  Validation F1-Score:  {val_f1:.4f}")

    # 6. Step Scheduler & Check LR
    scheduler.step()
    lr_after = scheduler.get_last_lr()[0]
    log(f"✓ CosineAnnealingLR Step 1 Applied - Learning Rate: {config.learning_rate:.6f} -> {lr_after:.6f}")

    # 7. Model Selection & Checkpointing
    checkpoint_saved = False
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = Path(config.checkpoint_dir) / config.best_checkpoint_filename

    is_best = early_stopping.step(val_auc)
    if is_best:
        torch.save(
            {
                "epoch": 1,
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
        checkpoint_saved = True
        checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        log(f"✓ Checkpoint Saved: Best validation ROC-AUC = {val_auc:.4f} -> Saved to '{checkpoint_path}' ({checkpoint_size_mb:.2f} MB)")

    elapsed_time = time.time() - start_time

    log("\n" + "=" * 70)
    log("PHASE C4 TINY CNN TRIAL SUMMARY")
    log("=" * 70)
    log(f"Device Used:             {device}")
    log(f"Batch Size:              {config.batch_size}")
    log(f"Number of Train Batches: {n_train_batches}")
    log(f"Number of Val Batches:   {n_val_batches}")
    log(f"Final Train Loss:        {epoch_train_loss:.4f}")
    log(f"Final Validation Loss:   {epoch_val_loss:.4f}")
    log(f"Validation ROC-AUC:      {val_auc:.4f}")
    log(f"Validation F1-Score:     {val_f1:.4f}")
    log(f"LR After Scheduler Step: {lr_after:.6f}")
    log(f"Checkpoint Saved:        {checkpoint_saved} ({checkpoint_path})")
    log(f"Total Execution Time:    {elapsed_time:.2f} seconds")
    log("=" * 70)


if __name__ == "__main__":
    default_dataset_root = os.environ.get(
        "DATASET_ROOT",
        r"C:\Users\jayadharshini\OneDrive\Desktop\dataset_final_v2\dataset_final_v2"
    )
    run_c4_training_trial(default_dataset_root)
