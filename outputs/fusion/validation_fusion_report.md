# CrossVerify M3 — Fusion Validation Experiments Report

**Execution Context:** Real Validation Run (Validation Development Only)  
**Test Set Status:** **STRICTLY LOCKED — NOT ACCESSED**  
**Validation Samples:** 1,650 (150 genuine, 1,500 forged across 150 unique record IDs)  

---

## 1. Experimental Objective & Methodology
This experiment evaluates late fusion between **Member 2 (CNN Visual Forensics)** and **Member 3 (Cross-Modal Structural Consistency)** on the validation split.

* **Primary Metric:** **PR-AUC** (Average Precision) under 10:1 class imbalance.
* **Secondary Metric:** **ROC-AUC**.
* **Operating Point Selection:** Maximizing F1 subject to **Specificity $\ge 0.90$** on validation.

> [!WARNING]
> **VALIDATION DEVELOPMENT RESULT NOTICE:**  
> All reported metrics are **Validation Development Results**. The validation set is utilized here for development, parameter sweeps, and diagnostic comparison. These results must NOT be described as unbiased final test estimates.

---

## 2. Summary Candidate Metrics (Validation Development)

| Candidate Name | PR-AUC | ROC-AUC | Brier | Threshold | Accuracy | Precision | Recall | F1 | Specificity | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M3 Alone (Calibrated)** | **0.9729** | **0.8358** | 0.0709 | 0.89 | 0.6952 | 0.9931 | 0.6693 | 0.7997 | 0.9533 | 0.0467 |
| **M3 Alone (Raw, tau=0.25)** | **0.9729** | **0.8357** | 0.1770 | 0.25 | 0.7273 | 0.9852 | 0.7107 | 0.8257 | 0.8933 | 0.1067 |
| **M2 Diagnostic (Raw CNN, t=0.50)** | **0.9455** | **0.5855** | 0.5841 | 0.50 | 0.3679 | 0.9851 | 0.3093 | 0.4708 | 0.9533 | 0.0467 |
| **Semantics-Naive Average (0.5 M2 + 0.5 M3_cal)** | **0.9820** | **0.8711** | 0.1815 | 0.53 | 0.4527 | 0.9761 | 0.4080 | 0.5755 | 0.9000 | 0.1000 |
| **M2 Transformed: g(cnn_probability)** | **0.9451** | **0.5917** | 0.0808 | 0.50 | 0.9091 | 0.9091 | 1.0000 | 0.9524 | 0.0000 | 1.0000 |
| **Semantics-Corrected Weighted Fusion (alpha=0.05)** | **0.9902** | **0.9124** | 0.0706 | 0.89 | 0.6952 | 0.9931 | 0.6693 | 0.7997 | 0.9533 | 0.0467 |
| **Semantics-Naive Blend (0.5 M2 + 0.5 M3_raw)** | **0.9883** | **0.9084** | 0.2416 | 0.30 | 0.8709 | 0.9894 | 0.8673 | 0.9243 | 0.9067 | 0.0933 |
| **Exploratory Validation Stacker (Logistic Regression OOF)** | **0.9873** | **0.9056** | 0.0611 | 0.87 | 0.8782 | 0.9894 | 0.8753 | 0.9289 | 0.9067 | 0.0933 |

---

## 3. Key Findings

### A. Complementarity Between Modalities
* **Visual Splice:** M2 alone detects visual splices with mean probability **0.9090**, while M3 alone struggled on clean visual splices (mean prob 0.4496, recall 34.4%).
* **Semantic & Checksum Forgeries:** M3 alone achieves near **100% recall** on text/QR mismatches, checksum errors, missing fields, and fine edits, where M2's mean score is $\approx 0.08$.
* **Coordinated Full Forgery:** Both models experience their lowest individual sensitivity (mean prob $\approx 0.09$ for M2 and $0.28$ for M3).

### B. Impact of Semantics-Corrected Transformation
* Transforming M2's visual splice probability into an estimated general forgery probability $g(cnn)$ via grouped 5-fold cross-validation resolved score scale misalignment.
* The Semantics-Corrected Weighted Fusion at **$\alpha = 0.05$** achieves **Val PR-AUC = 0.9902** and **Val ROC-AUC = 0.9124**, representing an improvement of **+0.0173 PR-AUC** and **+0.0767 ROC-AUC** over M3 alone.

### C. Exploratory Stacker
* The 5-fold Grouped CV Logistic Regression stacker achieved **Val PR-AUC = 0.9873** and **Val ROC-AUC = 0.9056**, assigning positive weights to both visual CNN ($w = 2.65$) and cross-modal consistency ($w = 8.92$).
* As documented, this result is exploratory because train-level OOF predictions were unavailable.

---

## 4. Attack-Wise Validation Performance (Specificity / Recall)

| Tamper Category | Role | Metric | Total | M3 Calibrated | Semantics-Naive | Semantics-Corrected Fusion | Exploratory Stacker |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **genuine** | genuine | specificity | 150 | 0.9533 | 0.9000 | 0.9533 | 0.9067 |
| **visual_splice** | forged_attack | recall | 450 | 0.2822 | 0.9244 | 0.2822 | 0.9311 |
| **coordinated_full_forgery** | forged_attack | recall | 150 | 0.0533 | 0.1000 | 0.0533 | 0.1200 |
| **text_qr_mismatch** | forged_attack | recall | 150 | 0.9267 | 0.2533 | 0.9267 | 0.9400 |
| **qr_only_mismatch** | forged_attack | recall | 150 | 0.9467 | 0.3000 | 0.9467 | 0.9600 |
| **checksum_invalid** | forged_attack | recall | 150 | 1.0000 | 0.2867 | 1.0000 | 1.0000 |
| **format_invalid** | forged_attack | recall | 150 | 1.0000 | 0.0800 | 1.0000 | 1.0000 |
| **field_missing** | forged_attack | recall | 150 | 1.0000 | 0.0200 | 1.0000 | 1.0000 |
| **fine_grained_edit** | forged_attack | recall | 150 | 0.9200 | 0.2667 | 0.9200 | 0.9400 |

---

## 5. Artifacts Generated
* `outputs/fusion/val_alignment.csv`
* `outputs/fusion/val_alignment_metrics.json`
* `outputs/fusion/fusion_candidate_metrics.csv`
* `outputs/fusion/alpha_sweep.csv`
* `outputs/fusion/threshold_sweep.csv`
* `outputs/fusion/complementarity.csv`
* `outputs/fusion/attack_wise_validation.csv`
* `outputs/fusion/score_distributions.csv`
* `outputs/fusion/fusion_run_config.json`
* `outputs/fusion/validation_fusion_results.json`
* Plots: `alpha_sweep.png`, `threshold_tradeoff.png`, `pr_curves.png`, `complementarity_breakdown.png`
