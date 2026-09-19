"""
Batch Inference Engine for Frozen M2 ResNet18 Checkpoint.
Runs deterministic evaluation transforms on canonical train images to generate:
- cnn_probability: P(cnn_label = forged) = P(visual_splice)
- cnn_prediction: thresholded at t = 0.50
"""

import math
from pathlib import Path
import sys
import time
import types
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as T

# Emulate checkpoint configuration module structure for safe unpickling
if "src.baseline" not in sys.modules:
    baseline_mod = types.ModuleType("src.baseline")
    cnn_mod = types.ModuleType("src.baseline.cnn")
    class CNNBaselineConfig:
        pass
    cnn_mod.CNNBaselineConfig = CNNBaselineConfig
    sys.modules["src.baseline"] = baseline_mod
    sys.modules["src.baseline.cnn"] = cnn_mod


# Deterministic Evaluation Preprocessing
def get_eval_transforms(target_size=(224, 224)):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess_image(img_path: Path, target_size=(224, 224)) -> torch.Tensor:
    """Aspect-ratio-preserving resize with symmetric center padding."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    
    pad_w = tw - nw
    pad_h = th - nh
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    
    padded = Image.new("RGB", target_size, (0, 0, 0))
    padded.paste(resized, (pad_left, pad_top))
    
    transform = get_eval_transforms(target_size)
    return transform(padded)


class ResNet18VisualForensics(nn.Module):
    def __init__(self, pretrained=False, dropout_rate=0.2):
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)


def load_m2_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = ResNet18VisualForensics()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_dataset(
    df: pd.DataFrame,
    image_base_dir: Path,
    checkpoint_path: Path,
    batch_size: int = 64,
    device_name: Optional[str] = None,
) -> pd.DataFrame:
    if device_name is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    
    print(f"Loading M2 model on {device}...")
    model = load_m2_model(checkpoint_path, device)
    
    n_samples = len(df)
    n_batches = math.ceil(n_samples / batch_size)
    probs = []
    
    print(f"Running inference on {n_samples} samples ({n_batches} batches)...")
    start_time = time.time()
    
    with torch.no_grad():
        for b in range(n_batches):
            b_start = b * batch_size
            b_end = min((b + 1) * batch_size, n_samples)
            batch_df = df.iloc[b_start:b_end]
            
            tensors = []
            for _, row in batch_df.iterrows():
                img_rel = row["image_path"]
                full_path = image_base_dir / img_rel
                tensors.append(preprocess_image(full_path))
            
            batch_tensor = torch.stack(tensors).to(device)
            logits = model(batch_tensor).squeeze(1)
            b_probs = torch.sigmoid(logits).cpu().numpy().tolist()
            probs.extend(b_probs)
            
            if (b + 1) % 10 == 0 or (b + 1) == n_batches:
                elapsed = time.time() - start_time
                print(f"Batch {b+1}/{n_batches} completed ({len(probs)}/{n_samples}) in {elapsed:.1f}s")
                
    df_out = df.copy()
    df_out["cnn_probability"] = probs
    df_out["cnn_prediction"] = (np.array(probs) >= 0.50).astype(int)
    return df_out
