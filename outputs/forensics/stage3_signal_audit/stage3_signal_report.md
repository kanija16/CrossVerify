# CrossVerify M3 — Stage 3 Local Document Structure Signal Audit

**Execution Timestamp:** 2026-09-16T14:09:06.131718+00:00  
**Status:** Signal Audit Complete (NO NEW MODEL TRAINED, ZERO TEST ACCESS)  

---

## 1. Executive Summary & Verdict

- **Total Candidate Local Features Audited:** 62
- **Strong Signals (ROC-AUC >= 0.65 or |d| >= 0.5):** 0
- **Weak Signals (ROC-AUC >= 0.55 or |d| >= 0.2):** 0
- **No Detectable Signal (ROC-AUC < 0.55):** 62

### Verdict on M4 Local Structure Branch:
> [!CAUTION]
> **NEGATIVE VERDICT:** There is **NO statistically robust local visual structure signal** distinguishing `coordinated_full_forgery` from genuine documents. The synthetic generator rendered the altered text fields with identical font antialiasing, edge sharpness, and background blending as genuine templates. Building an M4 local visual branch is **scientifically ungrounded** and will not resolve coordinated full forgeries.

---

## 2. Top 15 Audited Local Document Features (Genuine vs Coordinated)

| feature                        |   genuine_mean |   genuine_std |   coord_mean |   coord_std |   visual_splice_mean |   fine_grained_mean |   cohens_d |   ks_statistic |   ks_pvalue |   single_feature_roc_auc |   single_feature_pr_auc | signal_classification   |
|:-------------------------------|---------------:|--------------:|-------------:|------------:|---------------------:|--------------------:|-----------:|---------------:|------------:|-------------------------:|------------------------:|:------------------------|
| within_doc_mean_abs_zscore     |         0.8017 |        0.0778 |       0.8151 |      0.0419 |               0.8047 |              0.8087 |     0.2145 |         0.1467 |   0.0793098 |                   0.5608 |                  0.5499 | NO_SIGNAL               |
| field_local_entropy_std        |         0.2866 |        0.1138 |       0.3114 |      0.1196 |               0.3023 |              0.2847 |     0.2129 |         0.14   |   0.105738  |                   0.56   |                  0.5423 | NO_SIGNAL               |
| field_edge_density_std         |         0.0288 |        0.0087 |       0.0308 |      0.0087 |               0.0304 |              0.0293 |     0.2386 |         0.12   |   0.230782  |                   0.5558 |                  0.5797 | NO_SIGNAL               |
| field_local_entropy_range      |         0.7802 |        0.3143 |       0.8333 |      0.317  |               0.8257 |              0.7693 |     0.1681 |         0.12   |   0.230782  |                   0.5513 |                  0.5332 | NO_SIGNAL               |
| field_gray_mean_std            |         9.1274 |        3.5773 |       9.8516 |      3.8152 |              10.0378 |              9.1642 |     0.1958 |         0.0933 |   0.532187  |                   0.5447 |                  0.5567 | NO_SIGNAL               |
| field_hf_residual_energy_std   |      3202.17   |     1240.33   |    3412.11   |   1263.89   |            3635.85   |           3272.65   |     0.1677 |         0.1067 |   0.361727  |                   0.5419 |                  0.5465 | NO_SIGNAL               |
| field_laplacian_var_std        |      3190.72   |     1238.74   |    3399.23   |   1262.65   |            3624.82   |           3260.88   |     0.1667 |         0.1133 |   0.291074  |                   0.5417 |                  0.5465 | NO_SIGNAL               |
| field_hf_residual_energy_range |      8751.09   |     3435.62   |    9260.18   |   3360.78   |            9989.19   |           8963.98   |     0.1498 |         0.12   |   0.230782  |                   0.5416 |                  0.5337 | NO_SIGNAL               |
| field_laplacian_var_range      |      8719.32   |     3429.76   |    9224.07   |   3356.04   |            9959      |           8932.36   |     0.1488 |         0.12   |   0.230782  |                   0.5412 |                  0.5336 | NO_SIGNAL               |
| field_grad_mag_mean_std        |        35.7212 |       13.8384 |      37.8127 |     13.8758 |              42.109  |             35.6331 |     0.1509 |         0.0733 |   0.816483  |                   0.5372 |                  0.5527 | NO_SIGNAL               |
| within_doc_zscore_dispersion   |         0.5513 |        0.069  |       0.5465 |      0.0554 |               0.5525 |              0.5514 |    -0.0759 |         0.1333 |   0.139062  |                   0.5367 |                  0.5335 | NO_SIGNAL               |
| field_edge_density_range       |         0.0799 |        0.024  |       0.0841 |      0.0239 |               0.0837 |              0.081  |     0.1753 |         0.0933 |   0.532187  |                   0.5365 |                  0.5501 | NO_SIGNAL               |
| field_edge_density_max         |         0.3326 |        0.0318 |       0.3366 |      0.0169 |               0.3331 |              0.3352 |     0.1545 |         0.0933 |   0.532187  |                   0.5356 |                  0.5308 | NO_SIGNAL               |
| field_gray_mean_range          |        25.5131 |        9.9368 |      27.1696 |     10.1901 |              27.933  |             25.4885 |     0.1646 |         0.1    |   0.442524  |                   0.5356 |                  0.5382 | NO_SIGNAL               |
| field_grad_mag_std_range       |        27.2517 |        9.2602 |      28.6538 |      9.7943 |              31.9557 |             27.669  |     0.1471 |         0.1    |   0.442524  |                   0.5353 |                  0.5398 | NO_SIGNAL               |

---

## 3. Within-Document Outlier Analysis

| within_doc_metric            |   genuine_mean |   genuine_std |   coord_mean |   coord_std |   cohens_d |   ks_pvalue |
|:-----------------------------|---------------:|--------------:|-------------:|------------:|-----------:|------------:|
| within_doc_max_field_zscore  |         1.9011 |        0.1736 |       1.9074 |      0.0848 |     0.0459 |   0.894321  |
| within_doc_mean_abs_zscore   |         0.8017 |        0.0778 |       0.8151 |      0.0419 |     0.2145 |   0.0793098 |
| within_doc_zscore_dispersion |         0.5513 |        0.069  |       0.5465 |      0.0554 |    -0.0759 |   0.139062  |
| num_fields_gt_2sd            |         0      |        0      |       0      |      0      |     0      |   1         |
| num_fields_gt_3sd            |         0      |        0      |       0      |      0      |     0      |   1         |

---

## 4. Family-Stratified Consistency Audit

| document_family   | feature                    |   genuine_mean |   coord_mean |   cohens_d |   ks_statistic |   ks_pvalue |   roc_auc |
|:------------------|:---------------------------|---------------:|-------------:|-----------:|---------------:|------------:|----------:|
| family_a          | num_fields_analyzed        |         5      |       5      |     0      |         0      |   1         |    0.5    |
| family_a          | field_gray_mean_mean       |       172.449  |     173.027  |     0.2128 |         0.1333 |   0.520434  |    0.5566 |
| family_a          | field_gray_mean_std        |        12.0011 |      12.2062 |     0.1028 |         0.1333 |   0.520434  |    0.5394 |
| family_a          | field_gray_mean_max        |       190.719  |     190.803  |     0.027  |         0.1067 |   0.790656  |    0.5152 |
| family_a          | field_gray_mean_range      |        33.0973 |      33.1121 |     0.0026 |         0.1333 |   0.520434  |    0.5049 |
| family_a          | field_gray_std_mean        |        98.4014 |      96.9722 |    -0.2621 |         0.1867 |   0.146863  |    0.5769 |
| family_a          | field_gray_std_std         |         3.4611 |       3.4289 |    -0.0366 |         0.08   |   0.971733  |    0.5084 |
| family_a          | field_gray_std_max         |       102.405  |     100.99   |    -0.2376 |         0.1867 |   0.146863  |    0.5655 |
| family_a          | field_gray_std_range       |         9.3233 |       9.1012 |    -0.0987 |         0.0933 |   0.902582  |    0.5202 |
| family_a          | field_local_contrast_mean  |         0.9993 |       0.9968 |    -0.4015 |         0.1333 |   0.520434  |    0.5526 |
| family_a          | field_local_contrast_std   |         0.0009 |       0.0019 |     0.2728 |         0.0933 |   0.902582  |    0.547  |
| family_a          | field_local_contrast_max   |         1      |       0.9984 |    -0.3783 |         0.08   |   0.971733  |    0.54   |
| family_a          | field_local_contrast_range |         0.0023 |       0.0049 |     0.2776 |         0.0933 |   0.902582  |    0.5479 |
| family_a          | field_grad_mag_mean_mean   |       460.609  |     452.595  |    -0.3019 |         0.2267 |   0.0421031 |    0.5847 |
| family_a          | field_grad_mag_mean_std    |        46.2922 |      46.9982 |     0.0778 |         0.1067 |   0.790656  |    0.5205 |
| family_a          | field_grad_mag_mean_max    |       513.27   |     506.918  |    -0.1915 |         0.1733 |   0.210674  |    0.5524 |
| family_a          | field_grad_mag_mean_range  |       124.452  |     125.102  |     0.0266 |         0.08   |   0.971733  |    0.5067 |
| family_a          | field_grad_mag_std_mean    |       369.673  |     362.663  |    -0.2312 |         0.2267 |   0.0421031 |    0.5682 |
| family_a          | field_grad_mag_std_std     |        11.6364 |      11.16   |    -0.1364 |         0.12   |   0.656176  |    0.5351 |
| family_a          | field_grad_mag_std_max     |       384.045  |     376.827  |    -0.2487 |         0.1867 |   0.146863  |    0.5701 |

---

## 5. Field Role Breakdown (Sample Roles)

| role        | feature                    |   genuine_N |   coord_N |   genuine_mean |   coord_mean |   cohens_d |   ks_pvalue |
|:------------|:---------------------------|------------:|----------:|---------------:|-------------:|-----------:|------------:|
| address     | local_contrast             |          75 |        75 |         0.9998 |       0.9979 |    -0.3457 |   0.902582  |
| address     | grad_mag_mean              |          75 |        75 |       390.271  |     384.288  |    -0.2392 |   0.397445  |
| address     | edge_density               |          75 |        75 |         0.2371 |       0.2377 |     0.0417 |   0.520434  |
| address     | laplacian_var              |          75 |        75 |     20922.8    |   19514.5    |    -0.2566 |   0.146863  |
| address     | text_bg_contrast           |          75 |        75 |         0.1998 |       0.2027 |     0.1306 |   0.520434  |
| address     | boundary_discontinuity     |          75 |        75 |         0.2968 |       0.2923 |    -0.0221 |   0.397445  |
| address     | field_to_bg_gradient_ratio |          75 |        75 |         1.618  |       1.631  |     0.1139 |   0.656176  |
| dob         | local_contrast             |          75 |        75 |         0.9997 |       0.9949 |    -0.5006 |   0.397445  |
| dob         | grad_mag_mean              |          75 |        75 |       476.256  |     471.283  |    -0.1498 |   0.397445  |
| dob         | edge_density               |          75 |        75 |         0.3081 |       0.3092 |     0.0626 |   0.971733  |
| dob         | laplacian_var              |          75 |        75 |     27831.9    |   26563      |    -0.1709 |   0.210674  |
| dob         | text_bg_contrast           |          75 |        75 |         0.3131 |       0.3123 |    -0.0396 |   0.902582  |
| dob         | boundary_discontinuity     |          75 |        75 |         2.3816 |       2.2741 |    -0.3465 |   0.0421031 |
| dob         | field_to_bg_gradient_ratio |          75 |        75 |         2.0704 |       2.0785 |     0.1541 |   0.210674  |
| entity_type | local_contrast             |          74 |        75 |         0.9378 |       0.9395 |     0.0314 |   0.995452  |
| entity_type | grad_mag_mean              |          74 |        75 |       441.215  |     444.135  |     0.1095 |   0.315141  |
| entity_type | edge_density               |          74 |        75 |         0.3118 |       0.3106 |    -0.1199 |   0.357193  |
| entity_type | laplacian_var              |          74 |        75 |     24799.5    |   26012.4    |     0.1673 |   0.223045  |
| entity_type | text_bg_contrast           |          74 |        75 |         0.303  |       0.3041 |     0.1195 |   0.877886  |
| entity_type | boundary_discontinuity     |          74 |        75 |         4.996  |       4.9184 |    -0.142  |   0.380705  |

---

