"""
src/baseline/train_cnn_oof.py

CrossVerify Phase 1B-A: 5-Fold TRUE Out-Of-Fold (OOF) Training & Inference Runner
for Member 2's CNN visual-forensics branch.

Strictly adheres to:
1. Canonical GroupKFold assignments from outputs/fusion/train_oof_folds.csv (140 record_ids / 1540 rows per fold).
2. Zero record_id overlap between training/validation pool and held-out fold.
3. Deterministic internal train/val split on the 4-fold pool (GroupShuffleSplit, test_size=0.2, random_state=42).
4. Zero leakage: Held-out fold is NEVER used for training, validation, checkpointing, early stopping, or threshold tuning.
5. Fresh CNN instantiation per fold; no weight sharing across folds; never use models/cnn/best_model.pt.
6. Fold checkpoints are completely isolated in models/cnn_oof/fold_{fold}/fold_{fold}_best_checkpoint.pt.
7. Only original decoded RGB images may be cached in RAM; Member 2 approved transforms are applied normally upon access.
8. Zero access to validation or test canonical splits.
9. Generates outputs/fusion/oof/cnn_oof_predictions.csv and outputs/fusion/oof/cnn_oof_provenance.json.
"""

import os
import sys
import time
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.baseline.dataset import VisualForensicsDataset, CNN_LABEL_MAP
from src.baseline.transforms import get_train_transforms, get_eval_transforms
from src.baseline.cnn import (
    ResNet18VisualForensics,
    CNNBaselineConfig,
    EarlyStopping,
    set_seed
)


def log(msg: str = ""):
    print(msg, flush=True)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class CachedVisualForensicsDataset(Dataset):
    """
    Dataset wrapping a pandas DataFrame of samples.
    Optionally caches original decoded RGB images in memory up to max_cache_size.
    Applies the existing Member 2 approved transform pipeline normally for each access.
    Does NOT cache stochastic or pre-transformed tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Path,
        transform: Optional[Any] = None,
        return_metadata: bool = False,
        decoded_cache: Optional[Dict[str, Image.Image]] = None,
        max_cache_size: int = 2500
    ):
        self.df = df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.transform = transform
        self.return_metadata = return_metadata
        self.decoded_cache = decoded_cache if decoded_cache is not None else {}
        self.max_cache_size = max_cache_size

    def __len__(self) -> int:
        return len(self.df)

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "id": str(row["id"]),
            "record_id": str(row["record_id"]),
            "image_path": str(row["image_path"]),
            "final_label": str(row["final_label"]),
            "tamper_type": str(row["tamper_type"]),
            "cnn_label": str(row["cnn_label"])
        }

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_img_path = row["image_path"]
        full_img_path = self.dataset_root / rel_img_path
        path_key = str(full_img_path)

        # 1. Load or retrieve original decoded RGB image
        if path_key in self.decoded_cache:
            image = self.decoded_cache[path_key]
        else:
            try:
                image = Image.open(full_img_path).convert("RGB")
                if len(self.decoded_cache) < self.max_cache_size:
                    self.decoded_cache[path_key] = image
            except Exception as e:
                raise RuntimeError(f"Corrupt or unreadable image at {full_img_path}: {e}") from e

        # 2. Apply existing Member 2 transform pipeline normally on each access
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        # 3. Label mapping: genuine -> 0, forged -> 1
        raw_label = row["cnn_label"]
        label_int = CNN_LABEL_MAP[raw_label]
        target = torch.tensor(label_int, dtype=torch.long)

        if self.return_metadata:
            metadata = self.get_metadata(idx)
            return image_tensor, target, metadata

        return image_tensor, target


def train_single_oof_fold(
    fold: int,
    df_train_canonical: pd.DataFrame,
    df_folds: pd.DataFrame,
    dataset_root: Path,
    config: CNNBaselineConfig,
    device: torch.device,
    checkpoint_dir: Path,
    partial_pred_dir: Path,
    shared_decoded_cache: Dict[str, Image.Image]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes training, model selection, and OOF inference for a single GroupKFold fold.
    """
    fold_start_time = time.time()
    log(f"\n{'=' * 75}")
    log(f"STARTING CNN OOF FOLD {fold}/4")
    log(f"{'=' * 75}")

    # Set seed deterministically for this fold
    set_seed(config.seed)

    # 1. Partition into Held-Out Fold and 4-Fold Pool
    held_out_mask = (df_folds["oof_fold"] == fold)
    df_held_out = df_train_canonical[held_out_mask].copy().reset_index(drop=True)
    df_pool = df_train_canonical[~held_out_mask].copy().reset_index(drop=True)

    # Assert held-out fold exact counts
    assert len(df_held_out) == 1540, f"Fold {fold}: Expected 1540 held-out rows, got {len(df_held_out)}"
    assert df_held_out["record_id"].nunique() == 140, f"Fold {fold}: Expected 140 held-out record_ids, got {df_held_out['record_id'].nunique()}"
    assert len(df_pool) == 6160, f"Fold {fold}: Expected 6160 pool rows, got {len(df_pool)}"
    assert df_pool["record_id"].nunique() == 560, f"Fold {fold}: Expected 560 pool record_ids, got {df_pool['record_id'].nunique()}"

    # 2. Deterministic Internal Train / Validation Split on Pool ONLY
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_pool, groups=df_pool["record_id"]))
    df_int_train = df_pool.iloc[train_idx].copy().reset_index(drop=True)
    df_int_val = df_pool.iloc[val_idx].copy().reset_index(drop=True)

    # Assert counts on internal splits
    assert len(df_int_train) == 4928, f"Fold {fold}: Expected 4928 internal train rows, got {len(df_int_train)}"
    assert df_int_train["record_id"].nunique() == 448, f"Fold {fold}: Expected 448 internal train record_ids, got {df_int_train['record_id'].nunique()}"
    assert len(df_int_val) == 1232, f"Fold {fold}: Expected 1232 internal val rows, got {len(df_int_val)}"
    assert df_int_val["record_id"].nunique() == 112, f"Fold {fold}: Expected 112 internal val record_ids, got {df_int_val['record_id'].nunique()}"

    # Strict Identity Isolation Assertions
    int_train_recs = set(df_int_train["record_id"])
    int_val_recs = set(df_int_val["record_id"])
    held_out_recs = set(df_held_out["record_id"])

    assert int_train_recs.isdisjoint(int_val_recs), f"Fold {fold}: Leakage between internal train and internal val!"
    assert int_train_recs.isdisjoint(held_out_recs), f"Fold {fold}: Leakage between internal train and held-out OOF fold!"
    assert int_val_recs.isdisjoint(held_out_recs), f"Fold {fold}: Leakage between internal val and held-out OOF fold!"
    log(f"[ISOLATION VERIFIED] Internal Train ({len(int_train_recs)} recs), Internal Val ({len(int_val_recs)} recs), Held-Out ({len(held_out_recs)} recs) are strictly disjoint.")

    # 3. Dynamic pos_weight on internal training subset
    n_gen = int((df_int_train["cnn_label"] == "genuine").sum())
    n_forg = int((df_int_train["cnn_label"] == "forged").sum())
    assert n_forg > 0, f"Fold {fold}: No forged samples in internal train!"
    pos_w = float(n_gen) / float(n_forg)
    log(f"Fold {fold} Internal Train Class Counts: genuine={n_gen}, forged={n_forg} -> pos_weight={pos_w:.7f}")

    # 4. Checkpoint Isolation Setup
    fold_ckpt_dir = checkpoint_dir / f"fold_{fold}"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_ckpt_dir / f"fold_{fold}_best_checkpoint.pt"
    partial_pred_path = partial_pred_dir / f"partial_fold_{fold}_predictions.csv"

    # Verify checkpoint path isolation
    assert f"fold_{fold}" in str(checkpoint_path), "Checkpoint path must explicitly identify the fold!"
    assert "models/cnn/best_model.pt" not in str(checkpoint_path).replace("\\", "/"), "Must NEVER use baseline best_model.pt!"

    # 5. Check if this fold's partial prediction already exists (resume capability)
    if partial_pred_path.exists() and checkpoint_path.exists():
        log(f"\n[RESUME DETECTED] Fold {fold} partial predictions already complete at '{partial_pred_path.name}'.")
        df_cached_preds = pd.read_csv(partial_pred_path)
        if len(df_cached_preds) == 1540 and df_cached_preds["cnn_probability_oof"].isnull().sum() == 0:
            log(f"[RESUME VERIFIED] Loaded {len(df_cached_preds)} valid predictions for Fold {fold}.")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            provenance = {
                "fold": fold,
                "status": "COMPLETED_RESUMED",
                "best_epoch": ckpt.get("epoch", 0),
                "val_auc": ckpt.get("val_auc", 0.0),
                "val_loss": ckpt.get("val_loss", 0.0),
                "val_f1": ckpt.get("val_f1", 0.0),
                "pos_weight": pos_w,
                "n_internal_train": len(df_int_train),
                "n_internal_val": len(df_int_val),
                "n_held_out": len(df_held_out),
                "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "elapsed_seconds": 0.0
            }
            return df_cached_preds, provenance

    # 6. Instantiate DataLoaders with Member 2 Transforms
    train_transform = get_train_transforms(target_size=config.target_image_size)
    eval_transform = get_eval_transforms(target_size=config.target_image_size)

    train_ds = CachedVisualForensicsDataset(
        df=df_int_train,
        dataset_root=dataset_root,
        transform=train_transform,
        decoded_cache=shared_decoded_cache,
        max_cache_size=3000
    )
    val_ds = CachedVisualForensicsDataset(
        df=df_int_val,
        dataset_root=dataset_root,
        transform=eval_transform,
        decoded_cache=shared_decoded_cache,
        max_cache_size=3000
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # 7. Instantiate FRESH ResNet-18 Model for this fold
    log(f"Instantiating FRESH ResNet18 model for Fold {fold} (Pretrained ImageNet-1K)...")
    model = ResNet18VisualForensics(pretrained=config.pretrained, dropout_rate=config.dropout_rate)
    model = model.to(device)

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

    best_epoch = 0
    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_val_f1 = 0.0

    # 8. Training & Internal Validation Loop
    n_train_batches = len(train_loader)
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Train pass ---
        model.train()
        running_train_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(images)
            assert logits.shape == targets.shape, f"Shape mismatch: {logits.shape} vs {targets.shape}"

            loss = criterion(logits, targets)
            assert not (torch.isnan(loss) or torch.isinf(loss)), f"NaN loss at Fold {fold}, Epoch {epoch}, Batch {batch_idx}"

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == n_train_batches:
                log(f"  Fold {fold} | Epoch {epoch:02d}/{config.epochs:02d} | Batch [{batch_idx + 1}/{n_train_batches}] - Loss: {loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_ds)

        # --- Validation pass on internal validation set ONLY ---
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

        val_auc = float(roc_auc_score(all_val_targets, all_val_probs))
        val_preds = (all_val_probs >= 0.5).astype(int)
        val_f1 = float(f1_score(all_val_targets, val_preds))

        is_best = early_stopping.step(val_auc)
        ckpt_note = ""

        if is_best:
            best_epoch = epoch
            best_val_auc = val_auc
            best_val_loss = epoch_val_loss
            best_val_f1 = val_f1
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_auc": val_auc,
                    "val_loss": epoch_val_loss,
                    "val_f1": val_f1,
                    "pos_weight": pos_w,
                    "config": config
                },
                checkpoint_path
            )
            ckpt_note = f" [NEW BEST FOLD CHECKPOINT -> {checkpoint_path.name}]"

        epoch_time = time.time() - epoch_start
        log(
            f">>> Fold {fold} | Epoch {epoch:02d}/{config.epochs:02d} | "
            f"train_loss: {epoch_train_loss:.4f} | "
            f"val_loss: {epoch_val_loss:.4f} | "
            f"val_auc: {val_auc:.4f} | "
            f"val_f1: {val_f1:.4f} | "
            f"lr: {current_lr:.6f} | "
            f"time: {epoch_time:.1f}s{ckpt_note}"
        )

        scheduler.step()

        if early_stopping.early_stop:
            log(f"! Early stopping triggered for Fold {fold} at Epoch {epoch} (Patience {config.early_stopping_patience} reached).")
            break

    # 9. Load Best Checkpoint for this fold
    log(f"\n--- Loading Best Checkpoint for Fold {fold} (Epoch {best_epoch}, Val AUC: {best_val_auc:.4f}) ---")
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert best_checkpoint["fold"] == fold, f"Checkpoint fold mismatch: expected {fold}, got {best_checkpoint['fold']}"
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # 10. Generate OOF Predictions on Held-Out Fold ONLY
    log(f"--- Generating OOF Predictions on Fold {fold} Held-Out Set ({len(df_held_out)} samples) ---")
    held_out_ds = CachedVisualForensicsDataset(
        df=df_held_out,
        dataset_root=dataset_root,
        transform=eval_transform,
        return_metadata=True,
        decoded_cache=shared_decoded_cache,
        max_cache_size=3000
    )
    held_out_loader = DataLoader(held_out_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    oof_probs_list = []
    with torch.no_grad():
        for images, targets, metadata in held_out_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            oof_probs_list.extend(probs)

    assert len(oof_probs_list) == len(df_held_out), f"Mismatch in predicted count: {len(oof_probs_list)} vs {len(df_held_out)}"

    # Assemble partial fold predictions DataFrame
    df_partial = pd.DataFrame({
        "id": df_held_out["id"].values,
        "document_family": df_held_out["document_family"].values,
        "record_id": df_held_out["record_id"].values,
        "template_variant": df_held_out["template_variant"].values,
        "tamper_type": df_held_out["tamper_type"].values,
        "final_label": df_held_out["final_label"].values,
        "cnn_label": df_held_out["cnn_label"].values,
        "fold": fold,
        "cnn_probability_oof": np.round(oof_probs_list, 6)
    })

    # Save partial prediction for fault tolerance
    df_partial.to_csv(partial_pred_path, index=False)
    log(f"Saved partial OOF predictions for Fold {fold} to '{partial_pred_path.name}'.")

    elapsed_seconds = time.time() - fold_start_time
    log(f"Fold {fold} finished in {elapsed_seconds:.1f}s ({elapsed_seconds / 60:.2f} min).")

    provenance = {
        "fold": fold,
        "status": "COMPLETED",
        "best_epoch": best_epoch,
        "val_auc": round(best_val_auc, 5),
        "val_loss": round(best_val_loss, 5),
        "val_f1": round(best_val_f1, 5),
        "pos_weight": round(pos_w, 7),
        "n_internal_train": len(df_int_train),
        "n_internal_val": len(df_int_val),
        "n_held_out": len(df_held_out),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "elapsed_seconds": round(elapsed_seconds, 2)
    }

    return df_partial, provenance


def run_5_fold_cnn_oof(dataset_root_str: str) -> None:
    start_time = time.time()
    dataset_root = Path(dataset_root_str)

    log("=" * 80)
    log("CROSSVERIFY PHASE 1B-A: 5-FOLD TRUE OOF GENERATION (MEMBER 2 CNN)")
    log(f"Start Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log("=" * 80)

    # 1. Setup & Environment
    device = torch.device("cpu")
    torch.set_num_threads(8)
    log(f"PyTorch Device: {device} (threads={torch.get_num_threads()})")

    config = CNNBaselineConfig(
        batch_size=32,
        epochs=15,
        learning_rate=1e-4,
        weight_decay=1e-2,
        seed=42,
        early_stopping_patience=5,
        early_stopping_metric="val_auc",
        early_stopping_mode="max",
        early_stopping_delta=1e-4
    )

    # Paths
    train_csv_path = dataset_root / "data" / "splits" / "train.csv"
    folds_csv_path = REPO_ROOT / "outputs" / "fusion" / "train_oof_folds.csv"
    oof_output_dir = REPO_ROOT / "outputs" / "fusion" / "oof"
    oof_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_base_dir = REPO_ROOT / "models" / "cnn_oof"
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    # Verify input files
    assert train_csv_path.exists(), f"Missing train.csv at: {train_csv_path}"
    assert folds_csv_path.exists(), f"Missing train_oof_folds.csv at: {folds_csv_path}"

    df_train = pd.read_csv(train_csv_path)
    df_folds = pd.read_csv(folds_csv_path)

    # Verify canonical train alignment
    assert len(df_train) == 7700, f"Expected 7700 rows in train.csv, got {len(df_train)}"
    assert len(df_folds) == 7700, f"Expected 7700 rows in train_oof_folds.csv, got {len(df_folds)}"
    assert (df_train["id"] == df_folds["id"]).all(), "train.csv and train_oof_folds.csv 'id' mismatch!"
    assert (df_train["record_id"] == df_folds["record_id"]).all(), "train.csv and train_oof_folds.csv 'record_id' mismatch!"

    train_csv_hash = compute_sha256(train_csv_path)
    folds_csv_hash = compute_sha256(folds_csv_path)
    log(f"Canonical train.csv SHA-256:       {train_csv_hash}")
    log(f"Canonical train_oof_folds SHA-256: {folds_csv_hash}")

    # Verify fold distributions
    fold_counts = df_folds["oof_fold"].value_counts().to_dict()
    assert set(fold_counts.keys()) == {0, 1, 2, 3, 4}, "Folds must be exactly {0, 1, 2, 3, 4}"
    assert all(c == 1540 for c in fold_counts.values()), f"Every fold must have 1540 rows: {fold_counts}"

    fold_records = df_folds.groupby("oof_fold")["record_id"].nunique().to_dict()
    assert all(r == 140 for r in fold_records.values()), f"Every fold must have 140 record_ids: {fold_records}"

    # Shared decoded RGB cache (capped to avoid memory pressure)
    shared_decoded_cache: Dict[str, Image.Image] = {}

    # 2. Iterate Over Folds 0, 1, 2, 3, 4
    fold_predictions: List[pd.DataFrame] = []
    provenance_list: List[Dict[str, Any]] = []

    for fold in range(5):
        df_partial, prov = train_single_oof_fold(
            fold=fold,
            df_train_canonical=df_train,
            df_folds=df_folds,
            dataset_root=dataset_root,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_base_dir,
            partial_pred_dir=oof_output_dir,
            shared_decoded_cache=shared_decoded_cache
        )
        fold_predictions.append(df_partial)
        provenance_list.append(prov)

    # 3. Assemble Final Unified OOF Predictions
    log("\n" + "=" * 80)
    log("ASSEMBLING FINAL CANONICAL CNN OOF PREDICTIONS TABLE")
    log("=" * 80)

    df_all_oof = pd.concat(fold_predictions, ignore_index=True)

    # Re-align with canonical train.csv order
    df_all_oof_indexed = df_all_oof.set_index("id").reindex(df_train["id"]).reset_index()

    # Verify alignment
    assert len(df_all_oof_indexed) == 7700, f"Expected 7700 assembled rows, got {len(df_all_oof_indexed)}"
    assert (df_all_oof_indexed["id"] == df_train["id"]).all(), "Final OOF rows do not match train.csv id order!"
    assert (df_all_oof_indexed["record_id"] == df_train["record_id"]).all(), "Final OOF record_ids mismatch!"
    assert df_all_oof_indexed["cnn_probability_oof"].isnull().sum() == 0, "Null values in cnn_probability_oof!"

    # Ensure required columns
    required_cols = [
        "id",
        "document_family",
        "record_id",
        "template_variant",
        "tamper_type",
        "final_label",
        "cnn_label",
        "fold",
        "cnn_probability_oof"
    ]
    df_final_oof = df_all_oof_indexed[required_cols]

    final_pred_path = oof_output_dir / "cnn_oof_predictions.csv"
    df_final_oof.to_csv(final_pred_path, index=False)
    log(f"[SAVED] Canonical CNN OOF predictions saved to: '{final_pred_path.name}'")

    # 4. Generate Comprehensive Provenance Report
    total_time = time.time() - start_time
    provenance_record = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_path": str(REPO_ROOT),
        "branch": "main",
        "canonical_train_csv_path": str(train_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "canonical_train_csv_hash": train_csv_hash,
        "groupkfold_assignment_source": str(folds_csv_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "groupkfold_assignment_hash": folds_csv_hash,
        "fold_counts": {str(k): int(v) for k, v in fold_counts.items()},
        "internal_train_validation_split_method": "GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) grouped by record_id",
        "internal_split_seed": 42,
        "per_fold_counts": {
            f"fold_{i}": {
                "internal_train_rows": 4928,
                "internal_train_records": 448,
                "internal_val_rows": 1232,
                "internal_val_records": 112,
                "held_out_oof_rows": 1540,
                "held_out_oof_records": 140
            } for i in range(5)
        },
        "cnn_configuration": {
            "model_architecture": "ResNet18",
            "weights": "ResNet18_Weights.DEFAULT (ImageNet-1K)",
            "classification_head": "nn.Sequential(Dropout(p=0.2), Linear(512, 1))",
            "loss": "BCEWithLogitsLoss(pos_weight=N_genuine / N_forged)",
            "optimizer": "AdamW(lr=1e-4, weight_decay=1e-2)",
            "scheduler": "CosineAnnealingLR(T_max=15)",
            "batch_size": config.batch_size,
            "max_epochs": config.epochs,
            "seed": config.seed,
            "early_stopping_metric": config.early_stopping_metric,
            "early_stopping_mode": config.early_stopping_mode,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_delta": config.early_stopping_delta
        },
        "transforms_configuration": {
            "target_image_size": [224, 224],
            "resize_method": "AspectPreservingResizeAndPad(fill=0)",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "augmentations": "NONE (deterministic, no flips, no crops)"
        },
        "per_fold_results": provenance_list,
        "per_fold_held_out_prediction_count": {str(i): 1540 for i in range(5)},
        "source_code_files_used": [
            "src/baseline/cnn.py",
            "src/baseline/dataset.py",
            "src/baseline/transforms.py",
            "src/baseline/train_cnn_oof.py"
        ],
        "deviations_or_errors": "NONE",
        "total_execution_seconds": round(total_time, 2)
    }

    final_provenance_path = oof_output_dir / "cnn_oof_provenance.json"
    with open(final_provenance_path, "w") as f:
        json.dump(provenance_record, f, indent=2)
    log(f"[SAVED] Provenance record saved to: '{final_provenance_path.name}'")

    log("\n" + "=" * 80)
    log(f"PHASE 1B-A 5-FOLD CNN OOF GENERATION COMPLETE IN {total_time / 60:.2f} MINUTES")
    log("=" * 80)


if __name__ == "__main__":
    run_5_fold_cnn_oof(str(REPO_ROOT))
