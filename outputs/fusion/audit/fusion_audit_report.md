# CrossVerify M3 — Fusion Methodological Audit Report

**Status:** Methodological Pre-Freeze Audit  
**Date:** 2026-09-16  
**Test Set Status:** **STRICTLY LOCKED — AUDITED ZERO ACCESS**

---

## 1. Executive Summary & Core Audit Conclusions

1. **Exact Reproduction of Core Statistics:**
   * All previously generated metrics, counts, and probability distributions reproduce with $100\%$ precision from [`outputs/fusion/val_alignment.csv`](file:///Users/pavankumar/Desktop/CrossVerify_M3/outputs/fusion/val_alignment.csv).
   * Complementarity counts are verified exactly:
     - **Overall:** Both Correct = 320, M2 Correct / M3 Wrong = 287, M3 Correct / M2 Wrong = 880, Both Wrong = 163.
     - **Visual Splice:** Both Correct = 143, M2 Correct / M3 Wrong = 266, M3 Correct / M2 Wrong = 12, Both Wrong = 29.
     - **Other Forged:** Both Correct = 47, M2 Correct / M3 Wrong = 1, M3 Correct / M2 Wrong = 850, Both Wrong = 2.
     - **Coordinated:** Both Correct = 2, M2 Correct / M3 Wrong = 5, M3 Correct / M2 Wrong = 12, Both Wrong = 131.

2. **Flagged Unsupported Claim for Immediate Removal:**
   * **Claim:** *"Fusion resolves 70.72% of samples where an individual modality was blind."*
   * **Audit Finding:** **UNSUPPORTED / CONFUSED TERMINOLOGY.**  
     The calculation $(287 + 880) / 1650 = 1167 / 1650 = 70.72\%$ represents the fraction of **the entire validation set** where exactly one model was correct and the other was wrong (disagreement rate). It is **not** the resolution rate of blind spots by fusion. This claim has been excised from methodological documentation.

3. **Nature and Cause of $g(	ext{cnn})$ Range Compression:**
   * $g(	ext{cnn})$ is strictly compressed into $[0.8743, 0.9782]$ across all 5 cross-validation folds.
   * **Root Cause:** In the validation set (and overall dataset), the base rate of forgery is **10:1** ($90.9\%$ forged). The intercept of logistic regression fitted to this target is $eta_0 pprox +2.0$ to $+2.2$. Even when $cnn\_probability 	o 0$, $\sigma(eta_0) pprox \sigma(2.0) pprox 0.88$.
   * **Consequence:** This compression is mathematically expected given the base rate, but it causes weighted blend $lpha \cdot g(	ext{cnn}) + (1-lpha) \cdot p_m3_cal$ to behave non-linearly, shifting the probability mass into a narrow upper band $[0.76, 0.99]$.

4. **Stacker Methodological Status:**
   * The stacker is verified to use strict 5-fold GroupKFold by `record_id` with zero same-identity leakage across folds.
   * However, because train-level OOF predictions were unavailable, the stacker weights ($w_{	ext{cnn}} pprox 2.65, w_{	ext{m3}} pprox 8.92$) were fitted on validation data. Furthermore, operating thresholds were selected on the same OOF predictions.
   * **Verdict:** The stacker cannot be certified as an unbiased generalization estimate and should be treated as an **Exploratory Architecture**, not frozen without train OOF support.

---

## 2. Alpha Sweep & Operating-Point Analysis

| Alpha ($lpha$) | Best PR-AUC | Best ROC-AUC | Brier Score | Operating Point ($	ext{Spec} \ge 0.90$) | Precision | Recall | F1 Score | Specificity | Visual Splice Recall | Other Forged Recall | Coordinated Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** (M3 Alone) | 0.9729 | 0.8358 | 0.0709 | $	ext{th}=0.89$ | 0.9931 | 0.6693 | 0.7997 | 0.9533 | 0.2822 | 0.9656 | 0.0533 |
| **0.05** | **0.9903** | 0.9124 | 0.0706 | $	ext{th}=0.89$ | 0.9931 | 0.6693 | 0.7997 | 0.9533 | 0.2822 | 0.9656 | 0.0533 |
| **0.25** | 0.9882 | 0.9064 | 0.0700 | $	ext{th}=0.89$ | 0.9933 | 0.6893 | 0.8139 | 0.9533 | 0.3378 | 0.9689 | 0.0667 |
| **0.50** | 0.9880 | 0.9063 | 0.0713 | $	ext{th}=0.88$ | 0.9860 | 0.7067 | 0.8233 | 0.9000 | 0.3422 | 0.9911 | 0.0933 |
| **0.70** | 0.9898 | **0.9197** | 0.0740 | $	ext{th}=0.88$ | **0.9888** | **0.8813** | **0.9320** | **0.9000** | **0.9378** | **0.9778** | **0.1333** |
| **0.75** | 0.9873 | 0.9051 | 0.0749 | $	ext{th}=0.89$ | 0.9902 | 0.8713 | 0.9270 | 0.9133 | 0.9333 | 0.9656 | 0.1200 |
| **1.00** (Transformed M2) | 0.9451 | 0.5917 | 0.0808 | $	ext{th}=0.97$ | 0.9839 | 0.2033 | 0.3370 | 0.9667 | 0.6711 | 0.0033 | 0.0133 |

### Crucial Insight on Alpha Selection:
* While **$lpha = 0.05$** achieves the mathematically highest PR-AUC ($0.9903$), at the operational operating threshold ($	ext{Spec} \ge 0.90$), its visual splice recall is only **$28.22\%$** (virtually identical to M3 alone).
* In contrast, **$lpha = 0.70$** achieves the best ROC-AUC ($0.9197$) and second-highest PR-AUC ($0.9898$), while operationally unlocking **$93.78\%$ visual splice recall** and **$97.78\%$ other forged recall** at **$90.00\%$ specificity** ($	ext{F1} = 0.9320$).

---

## 3. Attack-Wise Specialization Verification

| Tamper Category | Role | Samples | M2 Alone ($t=0.50$) | M3 Raw ($	au=0.25$) | M3 Cal ($	ext{th}=0.89$) | Fusion $lpha=0.05$ | Fusion $lpha=0.70$ | Stacker OOF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **genuine** | Genuine | 150 | 0.9533 | 0.8933 | 0.9533 | 0.9533 | 0.9000 | 0.9067 |
| **visual_splice** | Forged | 450 | **0.9089** | 0.3444 | 0.2822 | 0.2822 | **0.9378** | **0.9311** |
| **coordinated_full_forgery** | Forged | 150 | 0.0467 | 0.1133 | 0.0533 | 0.0533 | 0.1333 | 0.1200 |
| **text_qr_mismatch** | Forged | 150 | 0.1067 | 0.9933 | 0.9267 | 0.9267 | 0.9867 | 0.9400 |
| **qr_only_mismatch** | Forged | 150 | 0.0733 | 1.0000 | 0.9467 | 0.9467 | 0.9867 | 0.9600 |
| **checksum_invalid** | Forged | 150 | 0.0333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **format_invalid** | Forged | 150 | 0.0267 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **field_missing** | Forged | 150 | 0.0133 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **fine_grained_edit** | Forged | 150 | 0.0667 | 0.9867 | 0.9200 | 0.9200 | 0.9733 | 0.9400 |

* **Empirical Pattern Confirmed:**
  - M2 is singularly strong on `visual_splice` ($90.89\%$) and near chance on all semantic/format attacks ($1-10\%$).
  - M3 is near-perfect on all structural attacks ($92-100\%$) and weak on `visual_splice` ($28-34\%$).
  - Both struggle on `coordinated_full_forgery` ($5-13\%$).
  - Balanced fusion ($lpha=0.70$) simultaneously captures the strengths of both systems.

---

## 4. Test Set Access Audit

* Scanned all code in `src/m3_crossmodal/fusion/`, `tests/test_fusion/`, and scripts.
* **Findings:**
  - ZERO references to `test.csv`, `test_predictions.csv`, or test images.
  - No test directories or files were accessed or created.
  - The test split remains 100% untouched and locked.

---

## 5. Final Recommendation on Model Selection Before Freeze

* **Architecture Decision:**
  1. **Do NOT freeze the Stacker as the final paper model:** Because train-level OOF predictions do not exist, the stacker weights were fitted on validation data.
  2. **Do NOT freeze $lpha = 0.05$ solely due to PR-AUC:** Despite achieving maximum PR-AUC ($0.9903$), at the operational operating point it fails to detect visual splices ($28.22\%$ recall), defeating the primary engineering purpose of fusion.
  3. **Candidate for Freeze:** **Semantics-Corrected Weighted Fusion with $lpha pprox 0.70$ (or Naive Blend with raw scores):** Provides genuine complementary protection ($93.78\%$ visual splice recall, $97.78\%$ semantic recall, $90.00\%$ genuine specificity, $	ext{F1} = 0.9320$).
