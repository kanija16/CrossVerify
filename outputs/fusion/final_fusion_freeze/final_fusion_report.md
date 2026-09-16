# CrossVerify M3 — Final Fusion Model Card & Freeze Report

**Freeze Status:** **FROZEN**  
**Selected Model Architecture:** Logistic Fusion (`LogisticRegression(C=1.0)`)  
**Input Features:** `cnn_probability`, `m3_probability`  
**Test Set Access Status:** **STRICTLY LOCKED — ZERO TEST ACCESS**  

---

## 1. Frozen Mathematical Formulation

$$\text{fusion\_logit} = 3.4710 \cdot \text{cnn\_probability} + 7.2098 \cdot \text{m3\_probability} - 1.8087$$

$$\text{fusion\_probability} = \sigma(\text{fusion\_logit}) = \frac{1}{1 + e^{-\text{fusion\_logit}}}$$

$$\text{final\_prediction} = \begin{cases} 1 \text{ (forged)} & \text{if } \text{fusion\_probability} \ge 0.7800 \\ 0 \text{ (genuine)} & \text{if } \text{fusion\_probability} < 0.7800 \end{cases}$$

---

## 2. Objective Selection Rationale
* **Predefined Selection Rule:** Maximizing validation F1 subject to the operational constraint $\text{Specificity} \ge 0.90$.
* **Measured Validation Superiority:**
  - **PR-AUC:** **0.9907** (Highest among all evaluated candidates, exceeding M3 alone by $+0.0178$).
  - **ROC-AUC:** **0.9228** (Highest among all evaluated candidates, exceeding M3 alone by $+0.0871$).
  - **Brier Score:** **0.0558** (Superior probability calibration, error reduced by $21.3\%$ over calibrated M3 alone).
  - **F1 Score:** **0.9297** (At $\tau = 0.7800$, $\text{Specificity} = 0.9000$, $\text{Recall} = 0.8773$).
* **Resolving Modality Blind Spots:**
  - `visual_splice` recall increased from **$34.4\%$** (M3 alone) to **$94.0\%$** ($423/450$).
  - Semantic and format forgeries maintained at **$93.3\% - 100\%$** recall.
* **Methodological Parsimony:**
  - Logistic regression on two legitimate model outputs is strictly monotonic, robust against overfitting, and requires no manual heuristic hyperparameter search.

---

## 3. Benchmark Comparison on Validation Split ($N=1,650$)

| Model Configuration | PR-AUC | ROC-AUC | Brier Score | Operating Threshold | Precision | Recall | F1 Score | Specificity | Visual Splice Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M3 Alone (Raw)** | 0.9729 | 0.8357 | 0.1770 | 0.25 | 0.9852 | 0.7107 | 0.8257 | 0.8933 | 0.3444 |
| **M3 Alone (Calibrated)** | 0.9729 | 0.8358 | 0.0709 | 0.89 | 0.9931 | 0.6693 | 0.7997 | 0.9533 | 0.2822 |
| **M2 Diagnostic (Raw)** | 0.9455 | 0.5855 | 0.5841 | 0.26 | 0.9718 | 0.3447 | 0.5089 | 0.9000 | 0.9089 |
| **Weighted Blend ($\alpha=0.05$)** | 0.9902 | 0.9127 | 0.0706 | 0.89 | 0.9931 | 0.6693 | 0.7997 | 0.9533 | 0.2822 |
| **Weighted Blend ($\alpha=0.70$)** | 0.9892 | 0.9164 | 0.0740 | 0.89 | 0.9893 | 0.8673 | 0.9250 | 0.9200 | 0.9267 |
| **FINAL FROZEN FUSION (LR)** | **0.9907** | **0.9228** | **0.0558** | **0.78** | **0.9887** | **0.8773** | **0.9297** | **0.9000** | **0.9400** |

*(All metrics labeled: VALIDATION BENCHMARK RESULTS)*

---

## 4. Attack-Wise Performance of Final Frozen Fusion

| Tamper Type | Ground Truth Role | Metric | Total Samples | Correct | Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **genuine** | Genuine | Specificity | 150 | 135 | **90.00%** |
| **visual_splice** | Forged Attack | Recall | 450 | 423 | **94.00%** |
| **checksum_invalid** | Forged Attack | Recall | 150 | 150 | **100.00%** |
| **format_invalid** | Forged Attack | Recall | 150 | 150 | **100.00%** |
| **field_missing** | Forged Attack | Recall | 150 | 150 | **100.00%** |
| **qr_only_mismatch** | Forged Attack | Recall | 150 | 144 | **96.00%** |
| **text_qr_mismatch** | Forged Attack | Recall | 150 | 140 | **93.33%** |
| **fine_grained_edit** | Forged Attack | Recall | 150 | 140 | **93.33%** |
| **coordinated_full_forgery** | Forged Attack | Recall | 150 | 19 | **12.67%** |

---

## 5. Methodological Provenance & Isolation
* **Training Predictions:** Trained on 7,700 frozen-upstream training predictions from M2 and M3. The upstream models were previously trained on the canonical training split; these predictions are not described as OOF.
* **Upstream Protection:** Upstream models M2 (`models/cnn/best_model.pt`) and M3 (`outputs/final_m3_model_freeze/`) remain 100% frozen and unaltered.
* **Validation Integrity:** Validation benchmark (`outputs/fusion/val_alignment.csv`) was read-only and remains unaltered.
* **Test Set Lock:** The test set was NOT accessed in any manner. All test-derived files remain locked until final evaluation.
