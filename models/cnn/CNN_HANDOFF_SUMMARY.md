# CrossVerify Member 2 — CNN Baseline Model Handoff Summary Package

This document serves as the authoritative handoff package for **Member 3 (Fusion & Multimodal Integration)** detailing the completed **CNN Baseline Model** for document visual-forensics.

---

## A. Model & Checkpoint Overview

- **Checkpoint File**: `models/cnn/best_model.pt`
- **Backbone Architecture**: ResNet18 (initialized with ImageNet-1K pretrained weights `resnet18-f37072fd.pth`)
- **Classification Head**: Replaced final fully-connected layer with `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(512, 1))`
- **Output**: Single raw unscaled logit $z \in (-\infty, +\infty)$
- **Loss Function**: `BCEWithLogitsLoss(pos_weight=2.6666667)` (where `pos_weight = N_genuine / N_forged = 5600 / 2100` from `train.csv`)
- **Optimizer**: AdamW ($\text{lr} = 1\times 10^{-4}$, $\text{weight\_decay} = 1\times 10^{-2}$)
- **Scheduler**: `CosineAnnealingLR` ($T_{\max} = 15$)
- **Best Training Epoch**: **Epoch 15/15** (Validation ROC-AUC = `0.9634`)

---

## B. Target Definition & Class Mapping

The CNN model operates strictly on **`cnn_label`** read directly from the canonical dataset splits:

$$\text{cnn\_label} = \begin{cases} \text{genuine} & \rightarrow 0 \quad (\text{Genuine documents and non-visual forgery types}) \\ \text{forged} & \rightarrow 1 \quad (\text{Visual splice forgery types ONLY}) \end{cases}$$

- **Target Encoding**: `genuine = 0`, `forged = 1`
- `cnn_label` is read directly from the canonical dataset CSV columns. It is **NEVER** reconstructed from `final_label` or `tamper_type`.

---

## C. Probability Semantics

- **Probability Calculation**:

$$P(\text{cnn\_label} = \text{forged}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Semantics**: `cnn_probability` represents the model's estimated probability that an input image contains **pixel-level visual splice artifacts** ($P(\text{cnn\_label} = \text{forged})$).
- **Important Distinction**:
  - `cnn_probability = 0.90` indicates a **90% likelihood of visual splice manipulation**.
  - `cnn_probability` does **NOT** equal $P(\text{final\_label} = \text{forged})$.
  - `cnn_probability` does **NOT** reflect non-visual forgery mechanisms (e.g. metadata mismatches, QR code mismatches, checksum errors). Non-visual forgeries intentionally share the visual appearance of genuine documents.

---

## D. Classification Decision Threshold

- **Fixed Primary Threshold**: **`t = 0.50`**
- **Decision Rule**:

$$\text{cnn\_prediction} = \begin{cases} 1 \text{ (forged)} & \text{if } \text{cnn\_probability} \ge 0.50 \\ 0 \text{ (genuine)} & \text{if } \text{cnn\_probability} < 0.50 \end{cases}$$

- **Test Integrity**: Threshold $t=0.50$ was fixed prior to test evaluation. **Zero threshold tuning or optimization was performed on the test set.**

---

## E. Preprocessing Pipeline

Deterministic evaluation preprocessing applied to all validation and test images:

1. **RGB Conversion**: Input image converted to 3-channel RGB.
2. **Aspect-Ratio-Preserving Resize**: Image scaled preserving aspect ratio such that the max dimension fits target geometry.
3. **Center Padding**: Symmetric zero-padding applied to produce exact $[3, 224, 224]$ dimensions.
4. **Tensor Conversion**: `torchvision.transforms.ToTensor()` (scales pixel intensities to $[0.0, 1.0]$).
5. **ImageNet Normalization**:
   - $\text{Mean} = [0.485, 0.456, 0.406]$
   - $\text{Std} = [0.229, 0.224, 0.225]$
6. **Augmentation**: **NO random flip, rotation, or color jitter** applied during baseline training or evaluation.

### Document Family Original Aspect Ratios:
- **Family A**: $1000 \times 640$ RGB PNG
- **Family B**: $1150 \times 520$ RGB PNG

---

## F. Dataset Splits & Identity Isolation

Strict identity-grouped 70 / 15 / 15 split across document `record_id`s:

| Split | Samples Count | Unique `record_id`s | Genuine Count | Forged Count |
| :--- | :---: | :---: | :---: | :---: |
| **Train** (`train.csv`) | 7,700 | 700 | 5,600 | 2,100 |
| **Validation** (`val.csv`) | 1,650 | 150 | 1,200 | 450 |
| **Test** (`test.csv`) | 1,650 | 150 | 1,200 | 450 |
| **Total** | **11,000** | **1,000** | **8,000** | **3,000** |

- **Identity Disjointness**: **Zero `record_id` overlap** across train, validation, and test splits.

---

## G. Final Validation Set Metrics (Epoch 15 Best Checkpoint)

Evaluated via direct inference on canonical `val.csv` (1,650 samples):

| Metric | Measured Value |
| :--- | :---: |
| **Validation Loss** | `0.3806` |
| **Accuracy** | **`0.9376` (93.76%)** |
| **Precision** | `0.8684` |
| **Recall (Sensitivity)** | `0.9089` |
| **F1-Score** | **`0.8882`** |
| **ROC-AUC** | **`0.9634`** |
| **PR-AUC (Average Precision)** | `0.9483` |

### Validation Confusion Matrix ($t=0.50$):
- **TN (Genuine $\rightarrow$ Genuine)**: `1,138`
- **FP (Genuine $\rightarrow$ Forged)**: `62`
- **FN (Forged $\rightarrow$ Genuine)**: `41`
- **TP (Forged $\rightarrow$ Forged)**: `409`

---

## H. Final Test Set Metrics (`test.csv`)

Evaluated via direct inference on canonical `test.csv` (1,650 samples):

| Metric | Measured Value |
| :--- | :---: |
| **Accuracy** | **`0.9467` (94.67%)** |
| **Precision** | `0.8803` |
| **Recall (Sensitivity)** | `0.9311` |
| **F1-Score** | **`0.9050`** |
| **ROC-AUC** | **`0.9817`** |
| **PR-AUC (Average Precision)** | **`0.9723`** |

### Test Confusion Matrix ($t=0.50$):
- **TN (Genuine $\rightarrow$ Genuine)**: `1,143`
- **FP (Genuine $\rightarrow$ Forged)**: `57`
- **FN (Forged $\rightarrow$ Genuine)**: `31`
- **TP (Forged $\rightarrow$ Forged)**: `419`

---

## I. Attack-Wise Results Breakdown (`models/cnn/attack_wise_results.csv`)

Performance breakdown across the 9 canonical `tamper_type` categories in `test.csv`:

| Tamper Type | `cnn_label` Distribution | Samples | Accuracy | Mean $P(\text{forged})$ | Median $P(\text{forged})$ | Interpretation / Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`visual_splice`** | **0 Genuine / 450 Forged** | **450** | **`0.9311`** | **`0.9333`** | **`0.9999`** | **Primary Target Forgery Class (CNN Positive)** |
| `checksum_invalid` | 150 Genuine / 0 Forged | 150 | `0.9400` | `0.1019` | `0.0155` | Non-Visual Forgery (Visual Detector correctly outputs Genuine) |
| `coordinated_full_forgery` | 150 Genuine / 0 Forged | 150 | `0.9267` | `0.1055` | `0.0195` | Non-Visual Forgery (Visual Detector correctly outputs Genuine) |
| `field_missing` | 150 Genuine / 0 Forged | 150 | `1.0000` | `0.0035` | `0.0007` | Non-Visual Forgery (Visual Detector correctly outputs Genuine) |
| `fine_grained_edit` | 150 Genuine / 0 Forged | 150 | `0.9333` | `0.0976` | `0.0124` | Non-Visual Forgery (Visual Detector correctly outputs Genuine) |
| `format_invalid` | 150 Genuine / 0 Forged | 150 | `0.9733` | `0.0369` | `0.0030` | Non-Visual Forgery (Visual Detector correctly outputs Genuine) |
| `genuine` | 150 Genuine / 0 Forged | 150 | `0.9533` | `0.0849` | `0.0107` | Canonical Genuine Control Group |
| `qr_only_mismatch` | 150 Genuine / 0 Forged | 150 | `0.9333` | `0.1065` | `0.0176` | Non-Visual Forgery (Visual Detector correctly outputs Genuine) |
| `text_qr_mismatch` | 150 Genuine / 0 Forged | 150 | `0.9600` | `0.0841` | `0.0145` | Non-Visual Forgery (Visual Detector correctly outputs Genuine) |

### Key Attack-Wise Takeaways for Member 3:
1. **Target Detection**: The CNN model detects visual splice artifacts with **93.11% Recall** ($419/450$) and a median probability of **`0.9999`**.
2. **Non-Visual Forgery Behavior**: For non-visual forgery mechanisms (e.g. text/QR mismatch, checksum, missing fields), the CNN model correctly classifies them as visually genuine (mean probability $\approx 0.0035 - 0.1065$). This behavior is mathematically correct under the image-only problem formulation. Member 3 will combine these predictions with consistency/OCR vectors for complete document verification.

---

## J. Family-Wise Results Breakdown (`models/cnn/family_wise_results.csv`)

Performance breakdown across the two document layout families:

| Document Family | Samples | Genuine / Forged | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Mean $P(\text{forged})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Family A** ($1000 \times 640$) | 825 | 600 / 225 | **`0.9636`** | `0.9258` | `0.9422` | **`0.9339`** | **`0.9867`** | **`0.9802`** | `0.2945` |
| **Family B** ($1150 \times 520$) | 825 | 600 / 225 | **`0.9297`** | `0.8381` | `0.9200` | **`0.8771`** | **`0.9782`** | **`0.9646`** | `0.3275` |

- **Observation**: High performance generalizes across both document aspect ratios ($1000 \times 640$ vs $1150 \times 520$), demonstrating robust feature learning under aspect-preserving resize + center padding.

---

## K. Handoff Prediction Files & Artifact Registry

| Artifact File | Description | Row Count | Columns |
| :--- | :--- | :---: | :--- |
| `models/cnn/best_model.pt` | PyTorch model state dict & training state | N/A | `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `val_auc`, `val_loss`, `val_f1`, `config` |
| `models/cnn/val_predictions.csv` | Full validation set predictions | 1,650 | `id`, `record_id`, `image_path`, `cnn_probability`, `cnn_prediction`, `cnn_label`, `final_label`, `tamper_type` |
| `models/cnn/test_predictions.csv` | Full test set predictions | 1,650 | `id`, `record_id`, `image_path`, `cnn_probability`, `cnn_prediction`, `cnn_label`, `final_label`, `tamper_type` |
| `models/cnn/attack_wise_results.csv` | Tamper-type breakdown metrics | 9 | `tamper_type`, `cnn_target_role`, `n_samples`, `n_genuine`, `n_forged`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `mean_cnn_probability`, `median_cnn_probability` |
| `models/cnn/family_wise_results.csv` | Document family breakdown metrics | 2 | `family_name`, `n_samples`, `n_genuine`, `n_forged`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `mean_cnn_probability` |

---

## L. Inference & Reproduction Instructions for Member 3

### Environment Package Versions:
- **Python**: `3.9.13`
- **PyTorch**: `2.8.0+cpu`
- **Torchvision**: `0.23.0+cpu`
- **Pandas**: `2.3.3`
- **NumPy**: `1.23.5`
- **Scikit-Learn**: `1.6.1`
- **Pillow**: `11.3.0`

### Code snippet for loading model & generating probabilities:

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
from src.baseline.transforms import get_eval_transforms

# 1. Define Model Architecture
class ResNet18VisualForensics(nn.Module):
    def __init__(self, pretrained=False, dropout_rate=0.2):
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)

# 2. Instantiate and load checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18VisualForensics()
checkpoint = torch.load("models/cnn/best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. Preprocess image & run inference
transform = get_eval_transforms(target_size=(224, 224))
img = Image.open("path/to/document.png").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # Shape [1, 3, 224, 224]

with torch.no_grad():
    raw_logit = model(img_tensor).squeeze(1)
    cnn_probability = torch.sigmoid(raw_logit).item()  # P(cnn_label = forged)
    cnn_prediction = 1 if cnn_probability >= 0.50 else 0
```

---

## M. Leakage & Test Isolation Confirmation

1. **Test Set Isolation**: `test.csv` was 100% isolated during training and model selection. Inference was executed strictly under `model.eval()` and `torch.no_grad()`.
2. **Zero Threshold Tuning**: Classification threshold $t=0.50$ was fixed prior to test evaluation.
3. **Zero Feature Leakage**: Inputs to model consist exclusively of `[3, 224, 224]` RGB image tensors.
4. **Canonical Preservation**: Canonical CSV files (`train.csv`, `val.csv`, `test.csv`, `record_id_split_map.csv`) and model checkpoint remain 100% unchanged.
