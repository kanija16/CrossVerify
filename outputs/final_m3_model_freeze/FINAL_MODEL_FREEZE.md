# Final M3 Model Freeze Manifest

**Model Status:** **FROZEN**  
**Timestamp:** `2026-09-14T08:13:32.001085+00:00`  
**Test Set Status:** **STRICTLY LOCKED DURING FREEZE — NOT YET ACCESSED**

---

## 1. Frozen Architecture
* **Classifier:** Canonical Random Forest with 5-fold Sigmoid (Platt) Calibration
* **Underlying Estimator:** `RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=42)`
* **Preprocessor:** `SimpleImputer(strategy='median', add_indicator=True)`
* **Calibration:** 5-fold Sigmoid scaling fitted strictly on `train`.
* **Feature Set:** Canonical A0 (15 features, common pooled classifier avoiding `document_family`).

## 2. Frozen Features
[
  "qr_readable",
  "text_qr_match_score",
  "checksum_valid",
  "format_valid",
  "missing_field_count",
  "ocr_field_presence_count",
  "ocr_field_presence_rate",
  "qr_field_count",
  "ocr_qr_key_overlap_count",
  "field_similarity__0",
  "field_similarity__1",
  "field_similarity__2",
  "field_similarity__3",
  "field_similarity__4",
  "checksum_computed"
]

## 3. Frozen Operating Threshold
* **Decision Threshold:** $\tau^* = 0.2500$
* **Selection Rationale:** Selected strictly on validation to enforce high genuine specificity ($89.33\%$) and high precision ($98.52\%$) while maximizing forged recall ($71.07\%$) under 10:1 class imbalance.

## 4. Frozen Validation Benchmark
* **PR-AUC:** **0.9729**
* **ROC-AUC:** **0.8357**
* **F1 Score:** **0.8257**
* **Precision:** **0.9852**
* **Recall:** **0.7107**
* **Specificity:** **0.8933**
* **Calibrated Brier Score:** **0.0709**
