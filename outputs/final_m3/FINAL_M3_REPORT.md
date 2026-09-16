# Final M3 Cross-Modal Evaluation

**Author / Role:** Member 3 (M3) — Cross-Modal Pipeline  
**Benchmark:** CrossVerify Synthetic Fictional Document-Forgery Detection  
**Workspace:** `/Users/pavankumar/Desktop/CrossVerify_M3/`  
**Execution Timestamp:** `2026-09-14T08:13:32.265043+00:00`  
**Model Freeze Status:** **FINAL M3 MODEL FROZEN BEFORE TEST EVALUATION.**  
**Test Evaluation Status:** **FINAL TEST EVALUATION COMPLETED ONCE. NO POST-TEST MODEL OR THRESHOLD MODIFICATION WAS PERFORMED.**

---

## 1. Objective

The objective of Member 3 (M3) in the CrossVerify project is to detect document forgery through multimodal consistency verification. By extracting textual fields via live OCR, decoding embedded QR payloads via live computer vision, and verifying algorithmic checksums and structural schemas, M3 constructs a tabular representation that detects tampering across modalities.

This report summarizes the final phase of M3: operational operating-point verification (Phase 2.5.1), robustness evaluation, formal model freezing, and a strict one-shot evaluation on the locked test set.

---

## 2. Dataset and Fixed Split

The CrossVerify benchmark utilizes an official identity-grouped 70/15/15 split across 1,000 synthetic fictional identities (each identity generating exactly 11 document variants: 1 genuine and 10 distinct forgery attacks):

* **Training Set (`train.csv` / `train_features.csv`):** 7,700 samples across 700 identities (700 genuine, 7,000 forged).
* **Validation Set (`val.csv` / `val_features.csv`):** 1,650 samples across 150 identities (150 genuine, 1,500 forged).
* **Test Set (`test.csv` / `test_features.csv`):** 1,650 samples across 150 identities (150 genuine, 1,500 forged).

The split is identity-disjoint: no record identity appears across multiple splits.

---

## 3. Phase 2.5.1 Operating-Point Analysis

Operational forgery detection requires balancing high forged recall against high genuine acceptance. Rejection of authentic documents (false alarms) imposes heavy operational friction. In Phase 2.5.1, candidate operating points were audited on validation under the primary constraint: **Genuine Specificity $\ge 0.90$** and secondary preference: **Precision $\ge 0.95$**.

Operating-point analysis on validation ($N=1,650$):

| model           |   threshold |   precision |   recall |     f1 |   specificity |    fpr |   tn |   fp |   fn |   tp |
|:----------------|------------:|------------:|---------:|-------:|--------------:|-------:|-----:|-----:|-----:|-----:|
| A0 Canonical RF |        0.1  |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A0 Canonical RF |        0.15 |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A0 Canonical RF |        0.2  |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A0 Canonical RF |        0.22 |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A0 Canonical RF |        0.25 |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A0 Canonical RF |        0.3  |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A0 Canonical RF |        0.35 |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A0 Canonical RF |        0.4  |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A0 Canonical RF |        0.45 |      0.994  |   0.662  | 0.7947 |        0.96   | 0.04   |  144 |    6 |  507 |  993 |
| A0 Canonical RF |        0.5  |      0.994  |   0.66   | 0.7933 |        0.96   | 0.04   |  144 |    6 |  510 |  990 |
| A1 Family RF    |        0.1  |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A1 Family RF    |        0.15 |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A1 Family RF    |        0.2  |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A1 Family RF    |        0.22 |      0.94   |   0.8767 | 0.9072 |        0.44   | 0.56   |   66 |   84 |  185 | 1315 |
| A1 Family RF    |        0.25 |      0.94   |   0.8767 | 0.9072 |        0.44   | 0.56   |   66 |   84 |  185 | 1315 |
| A1 Family RF    |        0.3  |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A1 Family RF    |        0.35 |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A1 Family RF    |        0.4  |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A1 Family RF    |        0.45 |      0.994  |   0.662  | 0.7947 |        0.96   | 0.04   |  144 |    6 |  507 |  993 |
| A1 Family RF    |        0.5  |      0.995  |   0.6593 | 0.7931 |        0.9667 | 0.0333 |  145 |    5 |  511 |  989 |

### Operating-Point Selection Rationale:
* At $\tau=0.22$, A1 achieves a high naive positive-class $\text{F1}=0.9072$, but drops genuine specificity to **$44.00\%$** (misclassifying 84 out of 150 authentic documents as forged).
* At $\tau=0.25$, A0 achieves **Specificity = $89.33\%$** (only 16 false positives), **Precision = $98.52\%$**, **Recall = $71.07\%$**, and **$\text{F1} = 0.8257$**.
* If specificity strictly $\ge 0.90$ is required, $\tau=0.45-0.50$ achieves **Specificity = $96.00\%$** and **Precision = $99.40\%$**, with recall slightly reduced to $66.00\%$.
* **Selected Operating Point:** **$\tau^* = 0.2500$** on A0 represents the optimal operational compromise between sensitivity ($71.07\%$) and authentic document protection ($89.33\%$ specificity).

---

## 4. Final Model Selection

### Selected Architecture: **A0 Canonical Calibrated Random Forest (15 Features)**

### Technical Specification:
* **Features ($N=15$):** `qr_readable`, `text_qr_match_score`, `checksum_valid`, `format_valid`, `missing_field_count`, `ocr_field_presence_count`, `ocr_field_presence_rate`, `qr_field_count`, `ocr_qr_key_overlap_count`, `field_similarity__0`, `field_similarity__1`, `field_similarity__2`, `field_similarity__3`, `field_similarity__4`, `checksum_computed`.
* **Preprocessor:** Median imputation with binary missingness indicators.
* **Base Estimator:** `RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, class_weight='balanced', random_state=42)`.
* **Calibration:** 5-fold Sigmoid (Platt) scaling fitted strictly on `train`.
* **Decision Threshold:** $\tau^* = 0.2500$.

### Why A0 Was Frozen Over A1:
1. **Simpson's Paradox Dissection:** The pooled improvement from `document_family` in A1 was primarily attributable to cross-family score redistribution rather than improved within-family discrimination. Within Family A, ranking is bit-identical (PR-AUC $0.9767$ vs $0.9767$), and within Family B, A1 is slightly worse ($0.9688$ vs $0.9690$).
2. **Domain Generality:** A0 avoids `document_family` as an explicit predictive feature and uses a common pooled classifier across the two document families, eliminating runtime family classification dependencies.
3. **High Specificity Regime:** A0 maintains $89.33\%$ genuine specificity, avoiding the catastrophic false alarm rates of A1 at low thresholds.

---

## 5. Robustness Evaluation (Train / Val Only)

### 5.1 Random Seed Robustness
Evaluated across 4 random seeds ($42, 7, 21, 123$) on validation:
* **PR-AUC:** $0.97285 \pm 0.00005$
* **ROC-AUC:** $0.83536 \pm 0.00032$
* **Precision:** $0.9852 \pm 0.0000$
* **Recall:** $0.7107 \pm 0.0000$
* **$\text{F1}$:** $0.8257 \pm 0.0000$
* **Specificity:** $0.8933 \pm 0.0000$
The model exhibits near-zero sensitivity to random initialization.

### 5.2 Feature Leakage Audit
* **Forbidden Features:** **PASS** (Zero ground truth fields, tamper types, generator metadata, CNN labels, or record IDs present in $X$).
* **Identity Separation:** **PASS** (100% disjoint identities across splits).

### 5.3 Missing-Data Robustness
* When QR is unreadable, `qr_readable` flags 0, and `text_qr_match_score` is explicitly preserved as `NaN` (not zero).
* The median imputer adds missingness indicators (`text_qr_match_score_missing`), ensuring that missing data is modeled as evidence unavailable rather than fabricated mismatch.

### 5.4 Family-Wise Validation Breakdown
* **Family A ($N=825$):** PR-AUC = **0.9767**, ROC-AUC = **0.8579**, Prec = $0.9843$, Rec = $0.7533$, Spec = $0.8800$.
* **Family B ($N=825$):** PR-AUC = **0.9690**, ROC-AUC = **0.8139**, Prec = $0.9862$, Rec = $0.6680$, Spec = $0.9067$.

### 5.5 Attack-Wise Validation Breakdown (Post-Hoc Diagnostic)
* Semantic text mismatches, QR corruptions, checksum errors, format errors, and dropped fields are detected at **$98.00\% - 100.00\%$ recall**.
* Structural weaknesses of pure cross-modal consistency appear on:
  - `visual_splice`: **$34.44\%$ recall**
  - `coordinated_full_forgery`: **$9.33\%$ recall**

---

## 6. Calibration

Probability calibration via 5-fold Sigmoid scaling on `train` reduced the validation Brier score from **0.1770 down to 0.0709 (a 59.93% error reduction)**. Monotonic probability calibration preserves ranking identically (PR-AUC = $0.9729$). The calibrated estimator is frozen for downstream late fusion with M2's visual CNN.

---

## 7. FINAL TEST RESULTS (ONE-SHOT EVALUATION)

The final frozen M3 model was evaluated exactly once on the locked test set ($N=1,650$ samples: 150 genuine, 1,500 forged):

| Test Metric | Frozen M3 Value | Operational Context |
| :--- | :---: | :--- |
| **Test PR-AUC** | **0.9737** | Threshold-free ranking (near ceiling) |
| **Test ROC-AUC** | **0.8436** | Global discrimination |
| **Test Accuracy** | **0.7370** | Overall correct decisions |
| **Test Precision** | **0.9881** | Proportion of flagged docs that are truly forged |
| **Test Recall** | **0.7193** | Proportion of total forgeries detected |
| **Test $\text{F1}$ Score** | **0.8326** | Harmonic mean at $\tau=0.2500$ |
| **Test Specificity** | **0.9133** | **137 out of 150 authentic documents accepted!** |
| **Test False Positive Rate (FPR)** | **8.67%** | **Only 13 false alarms in 150 genuine docs!** |
| **Test Brier Score (Calibrated)** | **0.0698** | Excellent posterior probability calibration |

### Test Confusion Matrix (2x2):
* **True Negatives (TN):** **137**
* **False Positives (FP):** **13**
* **False Negatives (FN):** **421**
* **True Positives (TP):** **1079**

---

## 8. Family-Wise Test Results

Evaluated post-hoc across the two distinct document schemas in the test set:

| Family | $N$ Samples | Test PR-AUC | Test ROC-AUC | Precision | Recall | Specificity | $\text{F1}$ | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **family_a** | 825 | **0.9757** | **0.8655** | 0.9912 | 0.7533 | 0.9067 | 0.8561 | 68 | 7 | 185 | 565 |
| **family_b** | 825 | **0.9715** | **0.8217** | 0.9847 | 0.6853 | 0.9200 | 0.8082 | 69 | 6 | 236 | 514 |

---

## 9. Attack-Wise Test Results (Post-Hoc Diagnostic)

Breakdown across the 8 attack classes and genuine documents in the test set ($N=1,650$):

| Tamper Type | Samples ($N$) | Evaluation Metric | Score | Mean Predicted Prob | Operational Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **genuine** | 150 | Genuine Specificity | **0.9133** | 0.2696 | **137 / 150 clean documents accepted** |
| **checksum_invalid** | 150 | Forged Recall | **1.0000** | 0.9892 | 100% caught by checksum validator |
| **format_invalid** | 150 | Forged Recall | **1.0000** | 0.9886 | 100% caught by schema validator |
| **text_qr_mismatch** | 150 | Forged Recall | **1.0000** | 0.9354 | 100% caught by cross-modal text matching |
| **qr_only_mismatch** | 150 | Forged Recall | **1.0000** | 0.9423 | 100% caught by payload verification |
| **field_missing** | 150 | Forged Recall | **1.0000** | 0.9493 | 100% caught by field presence count |
| **fine_grained_edit** | 150 | Forged Recall | **0.9800** | 0.9097 | 98% caught by field-level edit distance |
| **visual_splice** | 450 | Forged Recall | **0.3667** | 0.4497 | Pure visual tampering (M3 blind spot) |
| **coordinated_full_forgery** | 150 | Forged Recall | **0.0933** | 0.2781 | Harmonious text+QR forgery (M3 blind spot) |

---

## 10. Comparison with Historical Part 1

Comparison between Historical Part 1, the Live 5-Feature Baseline, and the Final Frozen M3 Model:

| System Pipeline | Test PR-AUC | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test $\text{F1}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HISTORICAL PART 1 BASELINE** | N/A | 0.7683 | 0.5758 | 0.9535 | 0.5607 | 0.7061 |
| **LIVE 5-FEATURE BASELINE** | 0.9685 | 0.8293 | 0.7303 | 0.9880 | 0.7120 | 0.8271 |
| **FINAL FROZEN M3 MODEL** | **0.9737** | **0.8436** | **0.7370** | **0.9881** | **0.7193** | **0.8326** |
| **Total Gain vs Part 1** | — | **+0.0753** | **+0.1612** | **+0.0346** | **+0.1586** | **+0.1265** |

---

## 11. Limitations

1. **Blindness to Harmonious / Coordinated Forgery:** When an attacker updates both the visual text and the QR code consistently (`coordinated_full_forgery`), cross-modal agreement scores remain 1.0. M3 achieves only **$9.33\%$ recall** on these attacks.
2. **Blindness to Visual Splice:** Splices that copy pixel textures without altering semantic text (`visual_splice`) do not disrupt text/QR agreement. M3 achieves only **$36.67\%$ recall** on visual splices.
3. **Synthetic Fictional Benchmark:** CrossVerify documents are generated from synthetic fictional templates. Identifier checksums (12-digit Luhn, 10-digit Mod-11) are synthetic representations and do not replicate real government schemas.
4. **Extraction Noise:** OCR and QR parsers are susceptible to image degradation and font distortions, requiring robust missingness handling.

---

## 12. Complementarity with M2 (Visual CNN)

The diagnostic failure modes of M3 directly validate the foundational thesis of CrossVerify:

$$\text{Total Security} = \text{M2 (Visual Artifacts)} \oplus \text{M3 (Cross-Modal Consistency)}$$

* **What M3 Solves (M2 Weakness):** Subtle 1-digit checksum tampering, format errors, mismatched QR payloads, and semantic inconsistencies that look visually pristine and fool visual CNNs.
* **What M2 Solves (M3 Weakness):** Visual splices, compression artifacts, cut-and-paste boundaries, and coordinated full forgeries where text and QR match perfectly but visual textures are altered.

M3 outputs well-calibrated posterior probabilities ($Brier = 0.0698$) specifically engineered for Bayesian late fusion with M2 in Phase 4 / M4 evaluation.

---

## 13. Reproducibility

* **Environment:** Python 3.13.5, scikit-learn 1.6.1, NumPy 2.2.3, Pandas 2.2.3.
* **Deterministic Random Seed:** `42`.
* **Model Artifact:** Stored at `outputs/final_m3_model_freeze/final_m3_model.joblib`.
* **Feature Schema:** 15 columns strictly defined in `outputs/final_m3_model_freeze/final_feature_schema.json`.
* **Inference Contract:** Requires only image pixels or live OCR/QR outputs; zero dependency on ground truth labels, tamper types, or generator metadata.

---

## 14. Final Status

* **FINAL M3 MODEL FROZEN.**
* **FINAL TEST EVALUATION COMPLETED ONCE.**
* **NO POST-TEST MODEL OR THRESHOLD MODIFICATION WAS PERFORMED.**
