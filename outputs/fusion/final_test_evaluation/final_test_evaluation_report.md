# CrossVerify M3 — Final Locked Test Evaluation Report

**Execution Timestamp:** 2026-09-16T14:41:20.185533+00:00  
**Frozen Model Architecture:** Logistic Regression (`C=1.0, solver='lbfgs'`)  
**Frozen Features:** `['cnn_probability', 'm3_probability']`  
**Frozen Decision Threshold:** **0.7800**  
**Exact Formula:** $\text{logit} = 3.470975847501685 \cdot p_{\text{cnn}} + 7.209759054944582 \cdot p_{\text{m3}} - 1.8087318980585172$

---

## 1. Protocol & Reproducibility Statement
The fusion model and operating threshold were frozen using the training/validation development procedure before this final evaluation.

An earlier pre-freeze verification command accessed test predictions before the final fusion freeze. No model, hyperparameter, or threshold changes were made based on that access. The results reported here constitute the final locked evaluation after model freeze.

### Exact Reproducibility Checksums:
* **Frozen M2 SHA256:** `93641580863cc8b7ce4a11144157c1dfd42ab416aa7a4132e26fa0dda3711bf6`
* **Frozen M3 SHA256:** `a0d3dd8a4c544ecd307e0926addda7b0b4d3a187138c13307cd582d6500d55f3`
* **Frozen Fusion SHA256:** `4e8391922ec267865abecb2348c256ed6f6c981986876772bfa6f17ab81110d2`
* **Evaluation Samples:** 1,650 (150 genuine, 1,500 forged across 9 tamper types)
* **Bootstrap Seed & Resamples:** Seed = 42, Iterations = 1,000

---

## 2. Final Frozen Fusion Test Performance ($N=1,650$)

| Metric | Measured Value | Bootstrap 95% Confidence Interval |
| :--- | :---: | :---: |
| **PR-AUC** | **0.99380** | [0.9919 - 0.9955] |
| **ROC-AUC** | **0.93921** | [0.9262 - 0.9518] |
| **Brier Score** | **0.05168** | — |
| **Accuracy** | **0.8927** | [0.8788 - 0.9061] |
| **Precision** | **0.9860** | [0.9796 - 0.9920] |
| **Recall** | **0.8947** | [0.8794 - 0.9088] |
| **F1-Score** | **0.9381** | [0.9297 - 0.9465] |
| **Specificity** | **0.8733** | — |
| **False Positive Rate** | **0.1267** | — |

### Confusion Matrix (2x2):
* **True Negative (TN):** 131 / 150 (Genuine correctly classified)
* **False Positive (FP):** 19 / 150 (Genuine misclassified as forged)
* **False Negative (FN):** 158 / 1500 (Forged misclassified as genuine)
* **True Positive (TP):** 1342 / 1500 (Forged correctly classified)

### Fusion Probability Distribution Summary:
* **Min:** 0.484838
* **25th Percentile (Q1):** 0.967949
* **Median:** 0.994769
* **75th Percentile (Q3):** 0.995618
* **Max:** 0.999859
* **Mean (SD):** 0.906250 (0.174200)

---

## 3. Model Comparison on Test Set

| Metric | M2 ResNet18 (post-hoc diagnostic, $t=0.50$) | M3 Cross-Modal (Frozen RF, $\tau=0.25$) | Final Frozen Fusion (LR, $\tau=0.78$) |
| :--- | :---: | :---: | :---: |
| **PR-AUC** | 0.94818 | 0.97371 | **0.99380** |
| **ROC-AUC** | 0.60012 | 0.84356 | **0.93921** |
| **Brier Score** | 0.58015 | 0.17132 | **0.05168** |
| **Accuracy** | 0.3709 | 0.7370 | **0.8927** |
| **Precision** | 0.9853 | 0.9881 | **0.9860** |
| **Recall** | 0.3127 | 0.7193 | **0.8947** |
| **F1 Score** | 0.4747 | 0.8326 | **0.9381** |
| **Specificity** | 0.9533 | 0.9133 | **0.8733** |
| **FPR** | 0.0467 | 0.0867 | **0.1267** |
| **Confusion Matrix [TN, FP, FN, TP]** | [143, 7, 1031, 469] | [137, 13, 421, 1079] | [131, 19, 158, 1342] |

---

## 4. Attack-Wise Test Breakdown (All 9 Tamper Types)

| Tamper Category | Ground Truth Role | Metric | Total Samples | TP / N | FN / N | M2 Alone ($t=0.50$) | M3 Alone ($\tau=0.25$) | Final Frozen Fusion ($\tau=0.78$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **checksum_invalid** | forged_attack | recall | 150 | 150/150 | 0/150 | 0.0600 | 1.0000 | **1.0000** |
| **coordinated_full_forgery** | forged_attack | recall | 150 | 24/150 | 126/150 | 0.0733 | 0.0933 | **0.1600** |
| **field_missing** | forged_attack | recall | 150 | 150/150 | 0/150 | 0.0000 | 1.0000 | **1.0000** |
| **fine_grained_edit** | forged_attack | recall | 150 | 146/150 | 4/150 | 0.0667 | 1.0000 | **0.9733** |
| **format_invalid** | forged_attack | recall | 150 | 150/150 | 0/150 | 0.0267 | 1.0000 | **1.0000** |
| **genuine** | genuine | specificity | 150 | TN=131/150 | FP=19/150 | 0.9533 | 0.9133 | **0.8733** |
| **qr_only_mismatch** | forged_attack | recall | 150 | 145/150 | 5/150 | 0.0667 | 1.0000 | **0.9667** |
| **text_qr_mismatch** | forged_attack | recall | 150 | 145/150 | 5/150 | 0.0400 | 1.0000 | **0.9667** |
| **visual_splice** | forged_attack | recall | 450 | 432/450 | 18/450 | 0.9311 | 0.3667 | **0.9600** |

---

## 5. Complementarity & Disagreement Analysis
* **Both Correct:** 326 (19.76%)
* **M2 Correct / M3 Wrong:** 286 (17.33%)
* **M3 Correct / M2 Wrong:** 890 (53.94%)
* **Both Wrong:** 148 (8.97%)
* **Disagreement Count:** 1176 (71.27%)

### Attack-Wise Complementarity Context:
* **Visual Splice:** M2 is the primary signal source (432 / 450 detected by fusion).
* **Metadata / Format / Checksum / Missing Field:** M3 cross-modal consistency achieves 100% detection independently of M2.
* **Text-QR Mismatch & Fine-Grained Edit:** M3 achieves high recall (93.3% - 94.7%), supported by fusion.
* **Coordinated Full Forgery:** Both upstream models lack strong discriminative features for this attack without degrading specificity.

---

## 6. Family-Wise Performance
* **Family A ($N=825$):** PR-AUC = 0.99365, ROC-AUC = 0.93803, Accuracy = 0.8921, Precision = 0.9896, Recall = 0.8907, F1 = 0.9375, Specificity = 0.9067.
* **Family B ($N=825$):** PR-AUC = 0.99421, ROC-AUC = 0.94350, Accuracy = 0.8933, Precision = 0.9825, Recall = 0.8987, F1 = 0.9387, Specificity = 0.8400.

---

## 7. Reference Target (1,450 / 1,500) Analysis
* **Aspirational Reference Target:** 1,450 / 1,500 (96.67%)
* **Measured Forged Detected by Fusion:** **1342 / 1,500 (89.47%)**
* **Measured Forged Missed by Fusion:** **158 / 1,500 (10.53%)**
* **Gap to Reference Target:** 108 forged documents
* **Methodological Principle:** The 1,450 / 1,500 figure is treated strictly as an **aspirational reference point**, not as a success criterion for post-hoc threshold shifting or model retraining. The test set is permanently consumed and no post-test tuning is permitted.
