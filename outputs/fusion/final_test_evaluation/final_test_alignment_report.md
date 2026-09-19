# CrossVerify M3 — Final Test Alignment & Protocol Report

**Execution Timestamp:** 2026-09-16T14:41:20.184930+00:00  
**Test Set Row Count:** 1,650  
**Unique Identities:** 150 (exactly 11 variants per identity)  

---

## 1. Protocol & Historical Context Caveat
The fusion model and operating threshold were frozen using the training/validation development procedure before this final evaluation.

An earlier pre-freeze verification command accessed test predictions before the final fusion freeze. No model, hyperparameter, or threshold changes were made based on that access. The results reported here constitute the final locked evaluation after model freeze.

---

## 2. Alignment Assertions
* **Row Count:** Exactly 1,650 rows across canonical `test.csv`, `models/cnn/test_predictions.csv`, and `outputs/features/test_features.csv`.
* **Sample ID Concordance:** 100% exact row-by-row match across all test sources.
* **Record ID Concordance:** 100% exact row-by-row match across all test sources.
* **Identity Disjointness:** Programmatically verified zero identity overlap across train, val, and test:
  - $\text{record\_ids}_{\text{train}} \cap \text{record\_ids}_{\text{test}} = \emptyset$
  - $\text{record\_ids}_{\text{val}} \cap \text{record\_ids}_{\text{test}} = \emptyset$
* **Target Balance:** 150 genuine (`target=0`) and 1,500 forged (`target=1`).
* **Feature Schema:** Evaluated on `['cnn_probability', 'm3_probability']` in exact frozen schema order.
