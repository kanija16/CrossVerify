"""
End-to-end image-only CNN baseline (paper revision).

Same ResNet-18 architecture, preprocessing, optimizer, schedule, seed and early-stopping
rule as the visual branch (src/baseline/train_cnn.py). The ONLY change is the target:

    visual branch (M2):   cnn_label   (visual splice = forged, everything else = genuine)
    this baseline:        final_label (genuine vs. ALL eight forgery mechanisms)

It answers the question "does a single image-only classifier trained on the full
forgery label make the cross-modal branch unnecessary?"

Protocol
  * train on data/splits/train.csv, select the checkpoint by validation ROC-AUC
  * pos_weight = N_negative / N_positive on the training split (700 / 7000 = 0.1)
  * operating threshold selected on validation with the fusion model's rule:
    maximise F1 subject to specificity >= 0.90 (grid 0.01..0.99)
  * the test split is scored ONCE, after training and threshold selection

Usage
  DATASET_ROOT=/path/to/dataset python -m src.baseline.train_cnn_e2e
  (DATASET_ROOT must contain images/family_a/*.png and images/family_b/*.png)

  Quick smoke test (tiny subset, 1 epoch, NOT a result):
  DATASET_ROOT=... python -m src.baseline.train_cnn_e2e --smoke

  30-epoch budget (the paper's headline run; the default 15 epochs matches M2's budget):
  DATASET_ROOT=... python -m src.baseline.train_cnn_e2e --epochs 30
  -> writes to models/cnn_e2e_ep30/ and outputs/baselines/cnn_e2e_ep30/

Outputs
  models/cnn_e2e/best_model.pt                         <- KEEP THIS FILE (gitignored)
  outputs/baselines/cnn_e2e/val_predictions.csv
  outputs/baselines/cnn_e2e/test_predictions.csv
  outputs/baselines/cnn_e2e/metrics.json               <- per-mechanism comparison with fusion
  outputs/baselines/cnn_e2e/training_log.json
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.baseline.cnn import CNNBaselineConfig, EarlyStopping, ResNet18VisualForensics, set_seed
from src.baseline.transforms import get_eval_transforms, get_train_transforms

SPLITS = REPO_ROOT / "data" / "splits"
FUSION_TEST = REPO_ROOT / "outputs" / "fusion" / "corrected_fusion_test_predictions.csv"
CKPT_DIR = REPO_ROOT / "models" / "cnn_e2e"
OUT_DIR = REPO_ROOT / "outputs" / "baselines" / "cnn_e2e"
LABEL = "final_label"
LABEL_MAP = {"genuine": 0, "forged": 1}


def log(msg=""):
    print(msg, flush=True)


class FinalLabelDataset(Dataset):
    """Image -> final_label. No metadata enters the model input."""

    def __init__(self, df, dataset_root, transform):
        self.df = df.reset_index(drop=True)
        self.root = Path(dataset_root)
        self.transform = transform
        missing = [p for p in self.df["image_path"] if not (self.root / p).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing under {self.root}, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["image_path"]).convert("RGB")
        return self.transform(img), torch.tensor(LABEL_MAP[row[LABEL]], dtype=torch.long)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for images, _ in loader:
        probs.append(torch.sigmoid(model(images.to(device))).cpu().numpy().ravel())
    return np.concatenate(probs)


def select_threshold(y, p, min_spec=0.90):
    """Max F1 subject to specificity >= min_spec (same rule as the fusion model)."""
    best = (-1.0, None)
    for th in np.round(np.linspace(0.01, 0.99, 99), 2):
        tn, fp, fn, tp = confusion_matrix(y, p >= th, labels=[0, 1]).ravel()
        if tn + fp and tn / (tn + fp) >= min_spec:
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best[0]:
                best = (f1, float(th))
    return best[1]


def summarize(y, p, th):
    tn, fp, fn, tp = confusion_matrix(y, p >= th, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "threshold": th,
        "recall": round(tp / (tp + fn), 4), "specificity": round(tn / (tn + fp), 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def subset(df, n_records, seed=0):
    keep = pd.Series(df["record_id"].unique()).sample(n=min(n_records, df["record_id"].nunique()), random_state=seed)
    return df[df["record_id"].isin(set(keep))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny subset + 1 epoch, for checking the setup only")
    ap.add_argument("--epochs", type=int, default=15,
                    help="max epochs (default 15 = same budget as the visual branch). Other values write to *_ep{N} folders")
    args = ap.parse_args()
    tag = "" if args.epochs == 15 or args.smoke else f"_ep{args.epochs}"
    ckpt_dir, out_dir = Path(f"{CKPT_DIR}{tag}"), Path(f"{OUT_DIR}{tag}")

    dataset_root = Path(os.environ.get("DATASET_ROOT", REPO_ROOT / "dataset"))
    config = CNNBaselineConfig(batch_size=32, epochs=1 if args.smoke else args.epochs)
    set_seed(config.seed)
    device = pick_device()
    t0 = time.time()

    train_df = pd.read_csv(SPLITS / "train.csv")
    val_df = pd.read_csv(SPLITS / "val.csv")
    test_df = pd.read_csv(SPLITS / "test.csv")
    if args.smoke:
        train_df, val_df, test_df = subset(train_df, 4), subset(val_df, 3), subset(test_df, 3)
        log("*** SMOKE TEST: tiny subset, 1 epoch. Results are meaningless. ***")
    assert not (set(train_df.record_id) & set(val_df.record_id) or set(train_df.record_id) & set(test_df.record_id)
                or set(val_df.record_id) & set(test_df.record_id)), "identity overlap between splits"

    n_pos = int((train_df[LABEL] == "forged").sum())
    n_neg = int((train_df[LABEL] == "genuine").sum())
    pos_weight = n_neg / n_pos
    log(f"device={device}  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  "
        f"pos_weight=N_neg/N_pos={n_neg}/{n_pos}={pos_weight:.4f}")

    size = config.target_image_size
    loaders = {
        "train": DataLoader(FinalLabelDataset(train_df, dataset_root, get_train_transforms(target_size=size)),
                            batch_size=config.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(FinalLabelDataset(val_df, dataset_root, get_eval_transforms(target_size=size)),
                          batch_size=config.batch_size, shuffle=False, num_workers=0),
    }
    y_val = (val_df[LABEL] == "forged").astype(int).values

    model = ResNet18VisualForensics(pretrained=config.pretrained, dropout_rate=config.dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    stopper = EarlyStopping(patience=config.early_stopping_patience, metric=config.early_stopping_metric,
                            mode=config.early_stopping_mode, delta=config.early_stopping_delta)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ("smoke_model.pt" if args.smoke else "best_model.pt")
    history, best_epoch = [], 0

    for epoch in range(1, config.epochs + 1):
        e0 = time.time()
        model.train()
        running = 0.0
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        p_val = predict(model, loaders["val"], device)
        val_auc = float(roc_auc_score(y_val, p_val))
        improved = stopper.step(val_auc)
        if improved:
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_auc": val_auc,
                        "target": LABEL, "pos_weight": pos_weight}, ckpt_path)
        history.append({"epoch": epoch, "train_loss": round(running / len(loaders["train"].dataset), 5),
                        "val_auc": round(val_auc, 5), "lr": optimizer.param_groups[0]["lr"],
                        "best": improved, "seconds": round(time.time() - e0, 1)})
        log(f"epoch {epoch:02d}/{config.epochs}  train_loss {history[-1]['train_loss']:.4f}  "
            f"val_auc {val_auc:.4f}{'  [best]' if improved else ''}  ({history[-1]['seconds']}s)")
        scheduler.step()
        if stopper.early_stop:
            log(f"early stopping at epoch {epoch}")
            break

    # Reload best checkpoint, select threshold on validation
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    p_val = predict(model, loaders["val"], device)
    threshold = select_threshold(y_val, p_val)
    if threshold is None:
        threshold = 0.5
        log("no threshold reached validation specificity >= 0.90; falling back to 0.5")
    log(f"best epoch {best_epoch}; validation-selected threshold {threshold}")

    # Score the test split once
    test_loader = DataLoader(FinalLabelDataset(test_df, dataset_root, get_eval_transforms(target_size=size)),
                             batch_size=config.batch_size, shuffle=False, num_workers=0)
    p_test = predict(model, test_loader, device)
    y_test = (test_df[LABEL] == "forged").astype(int).values

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    cols = ["id", "record_id", "document_family", "tamper_type", "final_label"]
    val_df.assign(e2e_probability=np.round(p_val, 8))[cols + ["e2e_probability"]].to_csv(
        out_dir / f"val_predictions{suffix}.csv", index=False)
    test_out = test_df.assign(e2e_probability=np.round(p_test, 8),
                              e2e_prediction=(p_test >= threshold).astype(int))[cols + ["e2e_probability", "e2e_prediction"]]
    test_out.to_csv(out_dir / f"test_predictions{suffix}.csv", index=False)

    per_type = test_out.groupby("tamper_type")["e2e_prediction"].mean().round(4).rename("cnn_e2e")
    if FUSION_TEST.exists() and not args.smoke:
        fusion = pd.read_csv(FUSION_TEST).set_index("id").loc[test_out["id"]]
        per_type = pd.concat([per_type, pd.Series(fusion["final_prediction"].values, index=test_out.index)
                              .groupby(test_out["tamper_type"]).mean().round(4).rename("fusion")], axis=1)
    metrics = {
        "target": LABEL, "best_epoch": best_epoch,
        "validation": summarize(y_val, p_val, threshold),
        "test": summarize(y_test, p_test, threshold),
        "test_flag_rate_by_tamper_type": per_type.reset_index().to_dict(orient="records"),
        "note": "Flag rate = recall for forgery types and false-positive rate for 'genuine'.",
    }
    (out_dir / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / f"training_log{suffix}.json").write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_test": args.smoke, "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "config": {k: str(v) for k, v in vars(config).items()},
        "pos_weight": pos_weight, "history": history, "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
        "total_seconds": round(time.time() - t0, 1),
        "test_access": "test split scored once, after checkpoint and threshold selection on validation",
    }, indent=2))

    log("\nTest results (end-to-end CNN):")
    log(json.dumps({k: metrics["test"][k] for k in ("roc_auc", "pr_auc", "recall", "specificity", "f1")}))
    log(per_type.to_string())
    log(f"\nsaved outputs to {out_dir.relative_to(REPO_ROOT)}; checkpoint {ckpt_path.relative_to(REPO_ROOT)}")
    if not args.smoke:
        log(f"KEEP {ckpt_path.relative_to(REPO_ROOT)} (it is gitignored) - back it up.")


if __name__ == "__main__":
    main()
