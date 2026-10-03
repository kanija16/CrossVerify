# CrossVerify M3 — Final Pre-Test Freeze Verification Report

**Execution Timestamp:** 2026-09-16T14:30:00Z  
**Verification Scope:** M2, M3, and M2+M3 Frozen 2-Way Fusion System  
**Test Data Access Status:** STRICTLY LOCKED — ZERO TEST SPLIT ACCESS  

---

## 1. Frozen Artifact Hashes, Sizes, and Timestamps

All canonical upstream and fusion artifacts were verified directly on the filesystem:

| Artifact Description | Canonical Relative Path | Size (Bytes) | SHA-256 Checksum | Freeze Timestamp (UTC) |
| :--- | :--- | :---: | :--- | :---: |
| **M2 ResNet18 Checkpoint** | `models/cnn/best_model.pt` | 134,256,709 | `93641580863cc8b7ce4a11144157c1dfd42ab416aa7a4132e26fa0dda3711bf6` | 2026-09-16 06:58:38 |
| **M3 Random Forest Model** | `outputs/final_m3_model_freeze/final_m3_model.joblib` | 609,922 | `a0d3dd8a4c544ecd307e0926addda7b0b4d3a187138c13307cd582d6500d55f3` | 2026-09-14 08:13:32 |
| **Frozen Fusion Model** | `outputs/fusion/final_fusion_freeze/final_fusion_model.joblib` | 879 | `4e8391922ec267865abecb2348c256ed6f6c981986876772bfa6f17ab81110d2` | 2026-09-16 10:14:23 |
| **Frozen Fusion Config** | `outputs/fusion/final_fusion_freeze/final_fusion_config.json` | 830 | `b9435eb7c6534663f335960735001587e51ffc044cc6e7177641deec72d0c03c` | 2026-09-16 10:14:23 |
| **Frozen Fusion Feature Schema** | `outputs/fusion/final_fusion_freeze/final_fusion_feature_schema.json` | 482 | `a2b602b87bbfafed24367edaf8d9a3b264de7b1be2cfbf43e3db76d3048e1842` | 2026-09-16 10:14:23 |
| **Frozen Validation Metrics** | `outputs/fusion/final_fusion_freeze/final_fusion_validation_metrics.json` | 1,997 | `948cfe117e2bc581fde1429117e38ba333d01b0477907ff9c3a17552a0d9440e` | 2026-09-16 10:14:23 |

**Artifact Unchanged Verification:**
- All 4 artifacts under `outputs/fusion/final_fusion_freeze/` carry identical filesystem timestamps (`2026-09-16 10:14:23 UTC`) corresponding to the original freeze event.
- Stage 1, Stage 2, and Stage 3 exploratory forensic runs wrote exclusively to `outputs/forensics/` and did NOT overwrite or touch `outputs/fusion/final_fusion_freeze/`.

---

## 2. Direct Serialized Model Inspection

Direct programmatic inspection of `outputs/fusion/final_fusion_freeze/final_fusion_model.joblib` confirms exact mathematical parameters:

- **Model Class:** `sklearn.linear_model.LogisticRegression`
- **`classes_`:** `[0, 1]` (binary classification: 0 = genuine, 1 = forged)
- **`coef_`:**
  $$\mathbf{w} = [[3.470975847501685, 7.209759054944582]]$$
- **`intercept_`:**
  $$w_0 = [-1.8087318980585172]$$
- **Features Required:** `["cnn_probability", "m3_probability"]` (in exact order)
- **Exact Decision Formula:**
  $$\text{logit} = 3.470975847501685 \cdot p_{\text{cnn}} + 7.209759054944582 \cdot p_{\text{m3}} - 1.8087318980585172$$
  $$p_{\text{fusion}} = \sigma(\text{logit})$$

---

## 3. Decision Threshold Confirmation

- **Operating Decision Threshold:** $\tau^* = \mathbf{0.78}$
- Confirmed in `final_fusion_config.json`: `"frozen_decision_threshold": 0.78`
- Confirmed in `final_fusion_validation_metrics.json`: `"operating_threshold": 0.78`

---

## 4. Validation Metrics Correspondence Confirmation

The canonical validation metrics in `final_fusion_validation_metrics.json` were matched against the canonical validation alignment:

- **PR-AUC:** `0.99072`
- **ROC-AUC:** `0.92276`
- **Brier Score:** `0.05578`
- **F1 Score:** `0.9297`
- **Precision:** `0.9887`
- **Recall:** `0.8773`
- **Specificity:** `0.9000` (FPR = 0.1000)
- **Accuracy:** `0.8794`
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 135 & 15 \\ 184 & 1316 \end{pmatrix}$$

---

## 5. Inspection of Final Test Evaluation Script

The execution script `src/m3_crossmodal/fusion/run_final_test_evaluation.py` was inspected line-by-line:

1. **Model Path:** Loads strictly from `outputs/fusion/final_fusion_freeze/final_fusion_model.joblib` (line 40).
2. **Features Loaded:** Stacked strictly as `[p_cnn, p_m3]` in accordance with `final_fusion_feature_schema.json` (lines 105–108).
3. **Threshold:** Hardcoded to `FROZEN_THRESHOLD = 0.78` (line 113).
4. **Retraining Check:** Zero `fit()` or `partial_fit()` calls exist. Pure `predict_proba()` inference only.
5. **Threshold Optimization Check:** Zero search, grid search, or threshold sweeps exist. Threshold is frozen at 0.78.
6. **Data Leakage Check:** Validation alignment is loaded strictly for identity disjointness assertion (`s_va.intersection(s_te) == 0`). No labels from validation or test are used to adjust the model.
7. **Pure Inference Operation:** Verified. The script executes a single forward pass, computes fixed-threshold classifications, and outputs test metrics.
8. **Candidate Model Selection:** Confirmed. No candidate models from Stages 1–3 are selected or used. The canonical frozen 2-way baseline is the sole model evaluated.

---

## 6. Verification Checklist Summary

| Verification Item | Requirement | Observed Status | Match? |
| :--- | :--- | :--- | :---: |
| 1. SHA256 Hashes | Exactly computed for all 6 artifacts | Verified and saved to `frozen_artifact_hashes.csv` | **YES** |
| 2. Model Parameters | Coefs: `[3.4709758..., 7.2097590...]`, Intercept: `-1.8087318...` | Matches to 16 decimal places | **YES** |
| 3. Operating Threshold | $\tau^* = 0.78$ | Specified in config and validation metrics | **YES** |
| 4. Validation Metrics | PR-AUC: 0.99072, ROC-AUC: 0.92276, Spec: 0.9000, CM: [135, 15, 184, 1316] | Exact match across all fields | **YES** |
| 5. Freeze Integrity | No modifications since freeze | Mtimes match freeze timestamp (2026-09-16 10:14:23 UTC) | **YES** |
| 6. Isolation from S1/S2/S3 | `outputs/fusion/final_fusion_freeze/` untouched | Zero modifications by Stages 1, 2, 3 | **YES** |
| 7. Test Script Model & Thresh | Canonical model path & threshold 0.78 | Verified in `run_final_test_evaluation.py` | **YES** |
| 8. Script Integrity | Pure inference, zero retraining, zero tuning | Verified line-by-line | **YES** |
| 9. Model Selection | Baseline maintained; no test-driven selection | Verified | **YES** |
| 10. Test Split Lock | No test images, test.csv, or predictions loaded | Strictly respected during audit | **YES** |

---

## 7. Final Verification Conclusion

All 10 verification criteria are fully satisfied with zero discrepancies.

**FINAL TEST READY**
