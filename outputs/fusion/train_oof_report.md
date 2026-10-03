# CrossVerify M3 — Final Fusion Training Dataset Preparation Report (Frozen Upstream Models)

**Execution Context:** Train-Level OOF Dataset Preparation for Final Fusion Model Training  
**Test Set Status:** **STRICTLY LOCKED — ZERO TEST ACCESS**  
**Validation Set Status:** **UNTOUCHED & PRESERVED**  

---

> [!NOTE]
> **Methodological Provenance Clarification:**  
> The M2 and M3 upstream models were trained on the complete canonical training split; consequently, these predictions are not strictly out-of-fold with respect to upstream model fitting. GroupKFold assignments are retained for identity-grouped partitioning and audit purposes.


## 1. Summary Overview
This phase prepares the canonical, scientifically defensible **Train-Level Out-of-Fold (OOF)** prediction dataset required for training the final M2+M3 fusion model.

* **Dataset Size:** **7,700 samples** (exactly matching canonical `train.csv`).
* **Identity Grouping:** **700 unique record IDs**, each containing exactly **11 document variants** (700 genuine, 7,000 forged).
* **Cross-Validation Strategy:** Deterministic 5-fold `GroupKFold(groups=record_id)`.
  - Fold 0: 140 identities (1,540 samples)
  - Fold 1: 140 identities (1,540 samples)
  - Fold 2: 140 identities (1,540 samples)
  - Fold 3: 140 identities (1,540 samples)
  - Fold 4: 140 identities (1,540 samples)
  - **Zero identity overlap across folds:** $\forall i \neq j, \text{record\_ids}_i \cap \text{record\_ids}_j = \emptyset$.

---

## 2. Model Semantics & Inference Provenance

### A. Member 2 (CNN Visual Forensics)
* **Model Checkpoint:** `models/cnn/best_model.pt` (ResNet18 backbone, evaluated under `model.eval()` and `torch.no_grad()`).
* **Score Definition:** `cnn_probability` represents $P(\text{cnn\_label} = \text{forged}) = P(\text{visual\_splice})$.
* **Semantics Guarantee:** The model was frozen upstream; no parameters or weights were altered. Predictions reflect pixel-level visual splice artifacts.

### B. Member 3 (Cross-Modal Structural Consistency)
* **Models Used:**
  - Raw Model: `outputs/final_m3_model_freeze/final_m3_model.joblib`
  - Calibrated Model: `outputs/final_m3_model_freeze/final_m3_calibrated_model.joblib`
* **Features:** Exactly 15 canonical features defined in `outputs/final_m3_model_freeze/final_feature_schema.json`. Zero forbidden metadata columns entered the model.
* **Calibration Provenance:** 5-fold Sigmoid (Platt) calibration was fitted strictly on the canonical training split during the Phase 3 freeze. It was not fitted on validation or test.

---

## 3. Artifacts Generated
* `outputs/fusion/train_oof_alignment.csv` (7,700 rows, 12 columns)
* `outputs/fusion/train_oof_folds.csv` (7,700 rows, 3 columns)
* `outputs/fusion/train_oof_run_config.json`
* `outputs/fusion/train_oof_report.md`

---

## 4. Leakage & Isolation Audit
1. **Test Set Isolation:** Neither `test.csv`, `test_predictions.csv`, nor any test image was accessed, loaded, or inspected.
2. **Validation Integrity:** `outputs/fusion/val_alignment.csv` was not modified and remains strictly preserved as the validation benchmark.
3. **Feature Guard:** No metadata (`tamper_type`, `document_family`, `record_id`, `id`, `image_path`) is included in model training features.
