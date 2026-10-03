# CrossVerify M3 — Stage 2 Three-Way Learned Fusion Report

**Status:** Validation Complete (TRAIN + VALIDATION ONLY)
**Test Split Access:** ZERO (Strictly Locked & Unloaded)

---

## 1. Candidate Model Descriptions & Stacker Weights

| candidate_id     | description                                                 | input_features                                                | weights                   |   intercept |
|:-----------------|:------------------------------------------------------------|:--------------------------------------------------------------|:--------------------------|------------:|
| B0_2Way_Baseline | Existing 2-way frozen fusion baseline [cnn_prob, m3_prob]   | ['cnn_probability', 'm3_probability']                         | [3.471, 7.2098]           |     -1.8087 |
| B1_3Way_RF       | 3-way learned fusion [cnn_prob, m3_prob, forensic_rf_prob]  | ['cnn_probability', 'm3_probability', 'forensic_probability'] | [2.8386, 2.2841, 16.9557] |    -13.0926 |
| B2_3Way_HGB      | 3-way learned fusion [cnn_prob, m3_prob, forensic_hgb_prob] | ['cnn_probability', 'm3_probability', 'forensic_probability'] | [2.7404, 2.7997, 14.5129] |    -10.9037 |
| B3_2Way_A2_RF    | 2-way direct fusion [cnn_prob, forensic_rf_prob]            | ['cnn_probability', 'forensic_probability']                   | [2.6986, 19.0408]         |    -13.9663 |
| B4_2Way_A2_HGB   | 2-way direct fusion [cnn_prob, forensic_hgb_prob]           | ['cnn_probability', 'forensic_probability']                   | [2.6023, 16.4374]         |    -11.4383 |

---

## 2. Validation Metrics Across Candidates ($N=1,650$)

| candidate_id     |   pr_auc |   roc_auc |   brier_score |   threshold |   accuracy |   precision |   recall |     f1 |   specificity |    fpr |
|:-----------------|---------:|----------:|--------------:|------------:|-----------:|------------:|---------:|-------:|--------------:|-------:|
| B0_2Way_Baseline |  0.99072 |   0.92276 |       0.05578 |        0.78 |     0.8794 |      0.9887 |   0.8773 | 0.9297 |        0.9    | 0.1    |
| B1_3Way_RF       |  0.98908 |   0.90955 |       0.06565 |        0.95 |     0.8448 |      0.9898 |   0.838  | 0.9076 |        0.9133 | 0.0867 |
| B2_3Way_HGB      |  0.98765 |   0.89528 |       0.07805 |        0.95 |     0.8418 |      0.9762 |   0.8467 | 0.9068 |        0.7933 | 0.2067 |
| B3_2Way_A2_RF    |  0.98796 |   0.90281 |       0.06753 |        0.95 |     0.8412 |      0.9897 |   0.834  | 0.9052 |        0.9133 | 0.0867 |
| B4_2Way_A2_HGB   |  0.98641 |   0.88605 |       0.08164 |        0.95 |     0.8491 |      0.9721 |   0.8587 | 0.9119 |        0.7533 | 0.2467 |

---

## 3. Attack-Wise Validation Breakdown

| tamper_type              |   B0_2Way_Baseline |   B1_3Way_RF |   B2_3Way_HGB |   B3_2Way_A2_RF |   B4_2Way_A2_HGB |
|:-------------------------|-------------------:|-------------:|--------------:|----------------:|-----------------:|
| checksum_invalid         |             1      |       0.9933 |        0.9933 |          0.9867 |           0.9933 |
| coordinated_full_forgery |             0.1267 |       0.1    |        0.1667 |          0.1    |           0.2467 |
| field_missing            |             1      |       0.9933 |        1      |          0.9933 |           1      |
| fine_grained_edit        |             0.9333 |       0.9533 |        0.9533 |          0.9533 |           0.96   |
| format_invalid           |             1      |       0.9933 |        1      |          0.9867 |           1      |
| genuine                  |             0.9    |       0.9133 |        0.7933 |          0.9133 |           0.7533 |
| qr_only_mismatch         |             0.96   |       0.96   |        0.9667 |          0.96   |           0.96   |
| text_qr_mismatch         |             0.9333 |       0.9733 |        0.98   |          0.9733 |           0.9867 |
| visual_splice            |             0.94   |       0.8044 |        0.8022 |          0.7956 |           0.8133 |

---

## 4. Coordinated Full Forgery Focus ($N=150$ Validation Samples)

| candidate_id     |   N |   correct |   errors |   metric_value |
|:-----------------|----:|----------:|---------:|---------------:|
| B0_2Way_Baseline | 150 |        19 |      131 |         0.1267 |
| B1_3Way_RF       | 150 |        15 |      135 |         0.1    |
| B2_3Way_HGB      | 150 |        25 |      125 |         0.1667 |
| B3_2Way_A2_RF    | 150 |        15 |      135 |         0.1    |
| B4_2Way_A2_HGB   | 150 |        37 |      113 |         0.2467 |

---

## 5. Hard-Attack & Complementarity Analysis

| tamper_type              |   N |   m2_correct |   m3_correct |   forensic_correct |   b0_2way_correct |   b1_3way_correct |   m3_wrong_and_forensic_right |
|:-------------------------|----:|-------------:|-------------:|-------------------:|------------------:|------------------:|------------------------------:|
| genuine                  | 150 |          143 |          134 |                  0 |               135 |               137 |                             0 |
| text_qr_mismatch         | 150 |           16 |          150 |                150 |               140 |               146 |                             0 |
| qr_only_mismatch         | 150 |           11 |          150 |                150 |               144 |               144 |                             0 |
| checksum_invalid         | 150 |            5 |          150 |                150 |               150 |               149 |                             0 |
| format_invalid           | 150 |            4 |          150 |                150 |               150 |               149 |                             0 |
| field_missing            | 150 |            2 |          150 |                150 |               150 |               149 |                             0 |
| visual_splice            | 450 |          409 |          155 |                450 |               423 |               362 |                           295 |
| fine_grained_edit        | 150 |           10 |          147 |                150 |               140 |               143 |                             3 |
| coordinated_full_forgery | 150 |            7 |           14 |                150 |                19 |                15 |                           136 |

---

## 6. Three-Way Stacker Feature Importance (B1 3-Way RF)

| feature              |   logistic_weight |   permutation_importance_mean |   permutation_importance_std |
|:---------------------|------------------:|------------------------------:|-----------------------------:|
| cnn_probability      |           2.83864 |                     0.0746609 |                   0.00511457 |
| m3_probability       |           2.28409 |                     0.0156436 |                   0.00388031 |
| forensic_probability |          16.9557  |                     0.185088  |                   0.0160045  |

---

