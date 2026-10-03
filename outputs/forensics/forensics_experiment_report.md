# CrossVerify M3 — Stage 1 Forensic Feature Engineering Report

**Status:** Validation Complete (TRAIN + VALIDATION ONLY)
**Test Split Access:** ZERO (Strictly Locked & Unloaded)

---

## 1. Overall Validation Metrics Across Ablations

| ablation_id         | description                      | model                |   n_features |   pr_auc |   roc_auc |   brier_score |   threshold |   accuracy |   precision |   recall |     f1 |   specificity |   tn |   fp |   fn |   tp |
|:--------------------|:---------------------------------|:---------------------|-------------:|---------:|----------:|--------------:|------------:|-----------:|------------:|---------:|-------:|--------------:|-----:|-----:|-----:|-----:|
| A0_RF_Baseline      | A0: 15 M3 Features Baseline      | RandomForest         |           15 |  0.97288 |   0.83539 |       0.0708  |        0.76 |     0.7267 |      0.9852 |   0.71   | 0.8253 |        0.8933 |  134 |   16 |  435 | 1065 |
| A0_HGB_Baseline     | A0: 15 M3 Features Baseline      | HistGradientBoosting |           15 |  0.97271 |   0.83438 |       0.07071 |        0.76 |     0.7261 |      0.9861 |   0.7087 | 0.8247 |        0.9    |  135 |   15 |  437 | 1063 |
| A1_RF_ForensicOnly  | A1: 16 Forensic-Only Features    | RandomForest         |           16 |  0.91604 |   0.50432 |       0.08389 |        0.5  |     0.9091 |      0.9091 |   1      | 0.9524 |        0      |    0 |  150 |    0 | 1500 |
| A1_HGB_ForensicOnly | A1: 16 Forensic-Only Features    | HistGradientBoosting |           16 |  0.9143  |   0.49897 |       0.08561 |        0.5  |     0.9085 |      0.91   |   0.998  | 0.952  |        0.0133 |    2 |  148 |    3 | 1497 |
| A2_RF_Combined      | A2: 15 M3 + 16 Forensic Combined | RandomForest         |           31 |  0.98112 |   0.83721 |       0.07129 |        0.85 |     0.7273 |      0.9808 |   0.714  | 0.8264 |        0.86   |  129 |   21 |  429 | 1071 |
| A2_HGB_Combined     | A2: 15 M3 + 16 Forensic Combined | HistGradientBoosting |           31 |  0.98238 |   0.83617 |       0.07516 |        0.9  |     0.7291 |      0.9808 |   0.716  | 0.8277 |        0.86   |  129 |   21 |  426 | 1074 |

---

## 2. Attack-Wise Recall Breakdown

| tamper_type              |   A0_HGB_Baseline |   A0_RF_Baseline |   A1_HGB_ForensicOnly |   A1_RF_ForensicOnly |   A2_HGB_Combined |   A2_RF_Combined |
|:-------------------------|------------------:|-----------------:|----------------------:|---------------------:|------------------:|-----------------:|
| checksum_invalid         |            1      |           1      |                1      |                    1 |            0.9933 |           0.9867 |
| coordinated_full_forgery |            0.0933 |           0.0933 |                1      |                    1 |            0.14   |           0.1267 |
| field_missing            |            1      |           1      |                1      |                    1 |            1      |           1      |
| fine_grained_edit        |            0.98   |           0.98   |                0.9933 |                    1 |            0.9333 |           0.96   |
| format_invalid           |            1      |           1      |                1      |                    1 |            0.9867 |           0.9933 |
| genuine                  |            0.9    |           0.8933 |                0.0133 |                    0 |            0.86   |           0.86   |
| qr_only_mismatch         |            0.9933 |           1      |                1      |                    1 |            0.9533 |           0.9733 |
| text_qr_mismatch         |            1      |           1      |                1      |                    1 |            0.9667 |           0.9867 |
| visual_splice            |            0.34   |           0.3422 |                0.9956 |                    1 |            0.3956 |           0.3711 |

---

## 3. Coordinated Full Forgery Focus ($N=150$ Validation Samples)

| ablation_id         |   N | metric   |   score |   tp_or_tn |   fn_or_fp |
|:--------------------|----:|:---------|--------:|-----------:|-----------:|
| A0_RF_Baseline      | 150 | recall   |  0.0933 |         14 |        136 |
| A0_HGB_Baseline     | 150 | recall   |  0.0933 |         14 |        136 |
| A1_RF_ForensicOnly  | 150 | recall   |  1      |        150 |          0 |
| A1_HGB_ForensicOnly | 150 | recall   |  1      |        150 |          0 |
| A2_RF_Combined      | 150 | recall   |  0.1267 |         19 |        131 |
| A2_HGB_Combined     | 150 | recall   |  0.14   |         21 |        129 |

---

## 4. Top 15 Feature Importances (A2 Combined RF)

| feature                            |   importance_mean |   importance_std | feature_group   |
|:-----------------------------------|------------------:|-----------------:|:----------------|
| text_qr_match_score                |       0.168128    |      0.0094958   | m3              |
| field_similarity__3                |       0.00779422  |      0.00446347  | m3              |
| format_valid                       |       0.00511244  |      0.00438851  | m3              |
| field_similarity__1                |       0.00486889  |      0.00370476  | m3              |
| field_similarity__2                |       0.00228444  |      0.00178582  | m3              |
| ocr_field_presence_rate            |       0.001944    |      0.00270869  | m3              |
| forensic_reg_lap_std_cv            |       0.00193689  |      0.00408713  | forensic        |
| forensic_reg_entropy_dispersion    |       0.001732    |      0.00228151  | forensic        |
| forensic_reg_grad_mag_max_z        |       0.00116356  |      0.00192932  | forensic        |
| forensic_reg_lap_std_max_z         |       0.001092    |      0.00206138  | forensic        |
| field_similarity__4                |       0.000892889 |      0.00210444  | m3              |
| forensic_reg_qr_vs_body_grad_ratio |       0.000721778 |      0.00149394  | forensic        |
| forensic_reg_grad_mag_mean         |       0.000242222 |      0.000796188 | forensic        |
| forensic_reg_edge_density_mean     |       0.000198667 |      0.00230333  | forensic        |
| checksum_valid                     |       4.48889e-05 |      0.00174514  | m3              |

---

