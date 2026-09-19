# Phase 2.5 Verification Audit

**Role:** Member 3 (M3) — Cross-Modal Pipeline  
**Benchmark:** CrossVerify Synthetic Fictional Document-Forgery Detection  
**Workspace:** `/Users/pavankumar/Desktop/CrossVerify_M3/`  
**Evaluation Scope:** Strictly Train ($N=7,700$) and Validation ($N=1,650$) Splits  
**Test Set Status:** **COMPLETELY LOCKED — ZERO TEST ACCESS, ZERO TEST EVALUATION**

---

## 1. Scope

* **Training Set:** $N=7,700$ samples across 700 unique identities (`train.csv` / `train_features.csv`).
* **Validation Set:** $N=1,650$ samples across 150 unique identities (`val.csv` / `val_features.csv`).
* **Test Set:** **LOCKED.** `test.csv` and `test_features.csv` were **NOT accessed, evaluated, scored, or loaded into memory**.

---

## 2. Reproduction Results

Reproduction of all Phase 2 candidate configurations on the validation set using deterministic random state (`random_state=42`):

| Candidate | Val PR-AUC | Val ROC-AUC | Val $\text{F1}$ | Val Precision | Val Recall | Val Accuracy | Operating $\tau$ | Val Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0 Canonical Pooled RF** | **0.9729** | **0.8357** | **0.8257** | **0.9852** | **0.7107** | **0.7273** | 0.2500 | 0.1770 |
| **A1 Family + Canonical RF** | 0.9781 | 0.8444 | 0.9072 | 0.9400 | 0.8767 | 0.8370 | 0.2200 | 0.1757 |
| **A2 Separate Family RFs** | 0.9781 | 0.8440 | 0.9072 | 0.9400 | 0.8767 | 0.8370 | 0.2200 | 0.1755 |
| **A3 Family Interactions RF** | 0.9781 | 0.8444 | 0.9072 | 0.9400 | 0.8767 | 0.8370 | 0.2300 | 0.1741 |
| **Ensemble 25RF/75HGB** | 0.9782 | 0.8447 | 0.9072 | 0.9400 | 0.8767 | 0.8370 | 0.2200 | 0.1781 |
| **Ensemble 50RF/50HGB** | 0.9782 | 0.8446 | 0.9072 | 0.9400 | 0.8767 | 0.8370 | 0.2200 | 0.1771 |
| **Ensemble 75RF/25HGB** | 0.9781 | 0.8444 | 0.9072 | 0.9400 | 0.8767 | 0.8370 | 0.2200 | 0.1763 |

---

## 3. Family Feature Investigation

### Why Pooled Metrics Change: Deconstructing Simpson's Paradox
In the pooled validation set, adding `document_family` increases PR-AUC from $0.9729$ to $0.9781$ (+0.0052) and ROC-AUC from $0.8357$ to $0.8444$ (+0.0087). However, an independent per-family evaluation reveals:

* **Family A Subset ($N=825$, 75 Genuine, 750 Forged):**
  - A0 (Agnostic) PR-AUC = **0.9767** | ROC-AUC = **0.8579**
  - A1 (With Family) PR-AUC = **0.9767** | ROC-AUC = **0.8579**
  - **Difference:** **0.0000** (Bit-identical intra-family ranking!)
* **Family B Subset ($N=825$, 75 Genuine, 750 Forged):**
  - A0 (Agnostic) PR-AUC = **0.9690** | ROC-AUC = **0.8139**
  - A1 (With Family) PR-AUC = **0.9688** | ROC-AUC = **0.8129**
  - **Difference:** **-0.0002 PR-AUC / -0.0010 ROC-AUC** (Slight degradation!)

### Statistical Distribution Analysis:
* **Family A:**
  - Genuine Mean Predicted Prob: A0 = $0.2696$ (Median $0.2424$) $\rightarrow$ A1 = $0.2458$ (Median $0.2183$)
  - Forged Mean Predicted Prob: A0 = $0.7622$ (Median $0.9700$) $\rightarrow$ A1 = $0.7610$ (Median $0.9823$)
* **Family B:**
  - Genuine Mean Predicted Prob: A0 = $0.2784$ (Median $0.2424$) $\rightarrow$ A1 = $0.2994$ (Median $0.2656$)
  - Forged Mean Predicted Prob: A0 = $0.7034$ (Median $0.9553$) $\rightarrow$ A1 = $0.7075$ (Median $0.9467$)

**Scientific Finding:**  
The pooled metric improvement is a classic **Simpson's Paradox aggregation effect**. Conditioning on `document_family` shifts the entire prediction distribution for Family A downward (genuine median drops to $0.2183$) while shifting Family B upward. This artificial probability displacement alters how samples from the two families interleave in the pooled ranking without providing any real discrimination gain within either family schema.

---

## 4. $\text{F1}$ Investigation ($0.8257 \rightarrow 0.9072$)

1. **Threshold Optimization Criterion:** The implementation in `find_optimal_threshold()` maximizes **macro-$\text{F1}$** across a grid of $\tau \in [0.10, 0.90]$ with 81 points. Under 10:1 class imbalance, binary $\text{F1}$ degenerates to $\tau=0.10$ ($	ext{F1}=0.9524$, $	ext{TN}=0$), whereas macro-$\text{F1}$ penalizes the collapse of the genuine class.
2. **Why $\text{F1}$ Jumps to $0.9072$:**
   - At $\tau=0.25$, A0 classifies Family A clean genuine documents with median probability $0.2424$ as Genuine ($	ext{TN}=134/150$, Specificity=$89.33\%$).
   - In A1, the downward probability shift causes the macro-$\text{F1}$ optimizer to select $\tau=0.2200$.
   - Lowering the threshold to $0.22$ sweeps $249$ forged samples (`visual_splice` and `coordinated_full_forgery`) above the decision threshold, increasing Recall from $0.7107$ to $0.8767$.
   - However, it also sweeps **66 clean genuine Family A documents** above the threshold, causing false positives to surge from $16$ to $84$.

---

## 5. Specificity / FPR Investigation

The $\text{F1}$ gain from $0.8257$ to $0.9072$ is achieved by trading away genuine document specificity:

| Metric | A0 Canonical RF ($\tau=0.25$) | A1 Family RF ($\tau=0.22$) | Impact of A1 |
| :--- | :---: | :---: | :--- |
| **True Negatives (TN)** | **134** / 150 | 66 / 150 | **-68 authentic documents** rejected |
| **False Positives (FP)** | **16** | 84 | **5.25x surge in false alarms** |
| **False Positive Rate (FPR)** | **10.67%** | **56.00%** | Unacceptable in production verification |
| **Genuine Specificity** | **89.33%** | **44.00%** | Massive loss of genuine acceptance |
| **Forged Recall** | 71.07% | 87.67% | +16.60% (swept into positive class) |
| **Precision** | 98.52% | 94.00% | -4.52% |

> [!CAUTION]
> **CRITICAL VERDICT ON A1:**  
> A1 is **NOT universally superior**. Its higher $\text{F1}$ is driven by setting a lower threshold ($\tau=0.22$) that misclassifies **$56\%$ of clean genuine documents as forged** (Specificity drops to $44\%$). A detector that rejects over half of all legitimate documents is unacceptable.

---

## 6. Threshold Audit Table

Exact operating point performance across thresholds on the validation set ($N=1,650$):

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
| A0 Canonical RF |        0.5  |      0.994  |   0.66   | 0.7933 |        0.96   | 0.04   |  144 |    6 |  510 |  990 |
| A1 Family RF    |        0.1  |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A1 Family RF    |        0.15 |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A1 Family RF    |        0.2  |      0.9091 |   1      | 0.9524 |        0      | 1      |    0 |  150 |    0 | 1500 |
| A1 Family RF    |        0.22 |      0.94   |   0.8767 | 0.9072 |        0.44   | 0.56   |   66 |   84 |  185 | 1315 |
| A1 Family RF    |        0.25 |      0.94   |   0.8767 | 0.9072 |        0.44   | 0.56   |   66 |   84 |  185 | 1315 |
| A1 Family RF    |        0.3  |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A1 Family RF    |        0.35 |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A1 Family RF    |        0.4  |      0.9852 |   0.7107 | 0.8257 |        0.8933 | 0.1067 |  134 |   16 |  434 | 1066 |
| A1 Family RF    |        0.5  |      0.995  |   0.6593 | 0.7931 |        0.9667 | 0.0333 |  145 |    5 |  511 |  989 |

---

## 7. Edit Distance Audit

* **Audit Status:** `IMPLEMENTED_BUT_NOT_INPUT_FEATURE`
* **Finding:** Edit distance implementation exists in `src/m3_crossmodal/part4_phase2.py` (functions `levenshtein_distance` and `normalized_edit_similarity`), but the Phase 2 ablation did not actually incorporate computed edit-distance features into the trained model; therefore no scientific conclusion about its predictive value can be made.
* **Feature Schema Verification:** All input feature vectors used in Phase 2 are derived strictly from `val_features.csv` which stores `difflib.SequenceMatcher` field similarities.

---

## 8. Semantic Alignment Audit

* **Audit Status:** `REDUNDANT_TRANSFORMATION_OF_EXISTING_FEATURES`
* **Exact Semantic Role Mappings:**
  - `sem_name_sim`: Slot 0 in both families (`name` in Fam A, `full_name` in Fam B).
  - `sem_identifier_sim`: Slot 3 (`id_number`) in Fam A, Slot 1 (`registry_id`) in Fam B.
  - `sem_date_sim`: Slot 1 (`dob`) in Fam A, Slot 4 (`registration_date`) in Fam B.
* **Finding:** The semantic features (`sem_min_field_sim`, `sem_name_sim`, etc.) are exact deterministic transformations of the existing 5 field similarity slots. Trees in A0 already learn these non-linear relationships directly, resulting in identical PR-AUC ($0.9729$).

---

## 9. Ensemble Audit

* **RF Alone:** Val PR-AUC = $0.9781$ | ROC-AUC = $0.8444$
* **HGB Alone:** Val PR-AUC = $0.9780$ | ROC-AUC = $0.8437$
* **Ensemble 25RF/75HGB:** Val PR-AUC = **0.9782** | ROC-AUC = **0.8447**
* **Ensemble 50RF/50HGB:** Val PR-AUC = **0.9782** | ROC-AUC = **0.8446**
* **Ensemble 75RF/25HGB:** Val PR-AUC = **0.9781** | ROC-AUC = **0.8444**
* **Delta vs RF Alone:** $\Delta \text{PR-AUC} = +0.0001$, $\Delta \text{ROC-AUC} = +0.0003$.
* **Audit Verdict:** The gain is **MARGINAL** ($+0.01\%$), representing random variance between tree implementations rather than a meaningful detection improvement.

---

## 10. Calibration Audit

* **Fitting Split:** `train` only (internal 5-fold cross-validation).
* **Validation Leakage:** **Zero.** Validation labels were never seen during calibration fitting.
* **Brier Score:** Uncalibrated = $0.1757 \rightarrow$ Calibrated = **0.0707** (**59.76% error reduction**).
* **Ranking Retention:**
  - Uncalibrated PR-AUC: $0.9781 \rightarrow$ Calibrated PR-AUC: **0.9781**
  - Uncalibrated ROC-AUC: $0.8444 \rightarrow$ Calibrated ROC-AUC: **0.8444**
* **Finding:** Sigmoid scaling is strictly monotonic, drastically improving probability reliability for downstream fusion without altering discrimination.

---

## 11. Null Test Audit

Audited all 5 training-label permutation trials:

* Trial 1: ROC-AUC = $0.5678$, PR-AUC = $0.9266$
* Trial 2: ROC-AUC = $0.5765$, PR-AUC = $0.9310$
* Trial 3: ROC-AUC = $0.4898$, PR-AUC = $0.9102$
* Trial 4: ROC-AUC = $0.5808$, PR-AUC = $0.9266$
* Trial 5: ROC-AUC = $0.4983$, PR-AUC = $0.9180$
* **Mean ROC-AUC:** **0.5426** (Close to random chance $0.5000$)
* **Mean PR-AUC:** **0.9225** (Close to validation positive base rate $0.9091$)

---

## 12. Leakage Audit

* **Forbidden Features:** **PASS** (Zero forbidden ground truth or metadata columns in $X$).
* **Identity Leakage:** **PASS** (700 train identities and 150 val identities are 100% disjoint).
* **Test Set Lock:** **PASS** (Zero test set access).
* **Family Negative Control:** **PASS** (`document_family` alone achieves exact chance ROC-AUC = $0.5000$).

---

## 13. Final Model Selection Recommendation

### Selected Model: **A0 Canonical Random Forest (15 Features)**

### Scientific Justification:
1. **Intra-Family Discrimination is Identical:** Decomposing Simpson's paradox proved that A1 provides **zero ranking gain** within Family A ($0.9767$ vs $0.9767$) and slightly degrades Family B ($0.9688$ vs $0.9690$). The apparent pooled gain ($0.9729 \rightarrow 0.9781$) is an aggregation artifact.
2. **Specificity Preservation:** A0 preserves **$89.33\%$ genuine specificity** at $\tau=0.25$ ($16$ false alarms), whereas A1 collapses specificity to **$44.00\%$** ($84$ false alarms) at $\tau=0.22$.
3. **Domain Agnosticism:** A0 is schema-independent and does not require document family classification at inference.
4. **Parsimony:** Rejecting A1, A2, A3, semantic features, and ensembles preserves an unbloated, robust 15-feature tree pipeline with zero risk of overfitting.

---

## 14. Confirmation of Test Set Lockdown

> [!IMPORTANT]
> **TEST SET WAS NOT LOADED, EVALUATED, SCORED, OR USED FOR ANY DECISION.**
