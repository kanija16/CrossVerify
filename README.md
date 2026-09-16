# CrossVerify — Member 3 (M3) Phase 1: Part 1

**Cross-Modal Data Interface & Precomputed Consistency Baseline**

This package implements **Part 1** of Member 3's responsibilities in the CrossVerify synthetic fictional identity document forgery benchmark.

---

## 1. Purpose of Part 1

Part 1 provides:
1. A **read-only, leakage-free data interface** that joins Member 1's (M1) fixed split CSVs with the underlying label JSON metadata.
2. A deterministic **5-dimensional feature extractor** that pulls the precomputed consistency vector from each document's metadata.
3. A comprehensive **sanity check test bench** validating split integrity, sample counts, identity separation, and label validity.
4. Two benchmark baselines:
   - **Learned Baseline:** Scikit-learn Logistic Regression trained on the 5 precomputed consistency features.
   - **Rule-Based Baseline (Sanity Floor):** Cryptographic/format heuristic (`checksum_valid == False OR format_valid == False OR qr_readable == False`).

---

## 2. M1 Data Locations (Source of Truth)

All M1 dataset files and splits are treated as strictly immutable and read-only:

- **M1 Split CSVs & Manifest (Desktop):**
  - Manifest: `~/Desktop/manifest.json` (11,000 records)
  - Split Map: `~/Desktop/record_id_split_map.csv` (1,000 identities)
  - Train Set: `~/Desktop/train.csv` (7,700 rows, 700 identities)
  - Validation Set: `~/Desktop/val.csv` (1,650 rows, 150 identities)
  - Test Set: `~/Desktop/test.csv` (1,650 rows, 150 identities)
- **M1 Dataset Root (Downloads):**
  - Root: `~/Downloads/dataset_final_v2`
  - Images: `images/family_a/` (5,500 PNGs) and `images/family_b/` (5,500 PNGs)
  - Labels: `labels/` (11,000 JSONs)

---

## 3. Project Structure

```
CrossVerify_M3/
├── README.md                           # This documentation
├── requirements.txt                    # Minimal environment dependencies
├── src/
│   └── m3_crossmodal/
│       ├── __init__.py                 # Package initialization
│       ├── config.py                   # Paths, schemas, feature order, leakage guards
│       ├── data_interface.py           # Read-only loader joining CSVs and label JSONs
│       ├── features.py                 # 5D consistency feature matrix builder
│       ├── sanity_checks.py            # Comprehensive verification & integrity checks
│       ├── baseline.py                 # Logistic regression & rule-based baselines
│       └── run_part1.py                # Main executable entry point
├── tests/
│   └── test_part1.py                   # Automated unit & integration tests
├── outputs/
│   └── baseline_metrics.json           # Serialized baseline evaluation results
├── experiments/                        # Scratch experiment notebooks / configs
└── docs/                               # Phase specifications & audit documentation
```

---

## 4. Feature Extraction & Ordering

The Part 1 feature matrix $X \in \mathbb{R}^{N \times 5}$ extracts the following five metrics in **strict canonical order**:

| Index | Feature Name | Type | Description |
| :---: | :--- | :---: | :--- |
| `0` | `qr_readable` | Float (`0.0` or `1.0`) | 1.0 if QR barcode is decoded; 0.0 if missing/corrupt |
| `1` | `text_qr_match_score` | Float ($[0.0, 1.0]$) | Mean character-level similarity across text and QR fields |
| `2` | `checksum_valid` | Float (`0.0` or `1.0`) | 1.0 if ID checksum passes (Verhoeff for A, Mod-11 for B) |
| `3` | `format_valid` | Float (`0.0` or `1.0`) | 1.0 if ID adheres to length & starting-digit regex |
| `4` | `missing_field_count` | Float ($[0.0, 5.0]$) | Count of empty or omitted printed text fields |

### Missing Score Handling Policy:
If `text_qr_match_score` is `None` (e.g. when QR is completely unreadable and matching cannot be executed), it is explicitly imputed with `0.0` (indicating no match evidence). This preserves the exact 5-dimensional feature space without silently assuming forgery.

### Status of OCR and QR Dictionaries:
The raw dictionaries `ocr_extracted_fields` and `qr_decoded_fields` are parsed and retained inside `M3Record` for subsequent phases. However, in Part 1 they are **NOT flattened or modeled as numeric features**.

---

## 5. Strict Data Leakage Restrictions

To prevent artificial metric inflation and data leakage:
- **Forbidden Features:** The following fields are strictly prohibited from entering feature matrix $X$:
  `id`, `record_id`, `filename`, `image_path`, `label_path`, `tamper_type`, `final_label`, `cnn_label`, `ground_truth_fields`, `expected_consistency_vector`, `splice_*`, `fine_grained_edit_metadata`.
- **Filename / Path Safety:** File paths contain attack names (e.g. `_fine_grained_edit.png`). The loader uses paths solely to open files on disk and never tokenizes or passes them to models.
- **Identity Isolation:** `record_id` is retained strictly for identity grouping and split verification; it is never fed to classifiers.

---

## 6. How to Run

### Run the Complete Pipeline:
From the project root:
```bash
PYTHONPATH="src" python3 -m m3_crossmodal.run_part1
```
Or directly:
```bash
python3 src/m3_crossmodal/run_part1.py
```

### Run Unit Tests:
```bash
pytest tests/ -v
```

---

## 7. Expected Split Sizes & Verified Baseline Results

- **Train:** 7,700 samples (700 unique identities) — 700 genuine, 7,000 forged
- **Validation:** 1,650 samples (150 unique identities) — 150 genuine, 1,500 forged
- **Test:** 1,650 samples (150 unique identities) — 150 genuine, 1,500 forged
- **Total:** 11,000 samples (1,000 unique identities)

---

## 8. What is Intentionally NOT Implemented in Part 1

The following components belong to later project phases and are intentionally excluded from Part 1:
- **Live OCR Extraction:** Live text extraction via Tesseract/EasyOCR directly from PNGs.
- **Live QR Decoding:** Live barcode parsing via pyzbar/OpenCV from PNGs.
- **Active Checksum / Format Engine:** Live re-computation of Verhoeff and Mod-11 from OCR output.
- **Cross-Modal Token Embeddings:** Semantic and phonetic text-QR comparison.
- **M2 CNN Forensics / Feature Fusion:** Integration with M2's visual splice detector.

---

## 9. Next Steps (Part 2 and Beyond)

- **Part 2:** Implement standalone verification modules: Verhoeff validator, Weighted Mod-11 validator, format regex validators, and the live consistency vector calculation engine.
- **Part 3:** Live image-in inference pipeline (`image -> OCR -> QR -> Validators -> Consistency Engine`).
- **Part 4:** Advanced cross-modal learned classifier and attack-wise ablation benchmarks.
- **Phase 2:** Multi-modal fusion with Member 2's CNN forensic probability score.
