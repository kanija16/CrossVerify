# Member 3 (M3) Phase 1 — Part 2: Validators & Consistency Engine

## 1. Overview & Project Context

Part 2 implements the core domain validators and cross-modal consistency engine for Member 3 (M3) of the **CrossVerify** synthetic document-forgery detection benchmark.

> **CRITICAL DISCLAIMER (FICTIONAL BENCHMARK CONSTRUCTIONS):**  
> All document families, schemas, format rules, check digit algorithms, weights, and permutation tables implemented herein are **purely synthetic, fictional benchmark constructions** created specifically for research into multimodal document forgery detection. They do NOT reflect, model, reverse-engineer, or authenticate any real-world national identity systems, government registries, legal instruments, or financial credentials.

---

## 2. Validator Modules (`src/m3_crossmodal/validators.py`)

The validator module provides standalone, deterministic check-digit algorithms and format validators for both fictional document families.

### Family A: Verhoeff-based Checksum & Format
- **Identifier Field:** `id_number`
- **Format Constraints:**
  - Length: Exactly 12 characters.
  - Character Class: Numeric digits only (`0-9`).
  - Leading Digit: Must be in the range `[2, 9]` (leading `0` and `1` are format-invalid).
- **Checksum Algorithm:** **Verhoeff Checksum** (based on the dihedral group $D_5$ of order 10):
  - Uses the $10 \times 10$ non-commutative permutation/multiplication table ($D_5$).
  - Permutation table $P[i][j]$ applies position-dependent permutations.
  - Valid sequence satisfies: $\sum_{i=0}^{n-1} F(d_i) \equiv 0$ in $D_5$, where $d_0$ is the check digit at the rightmost position.
  - Generates check digit such that the full 12-digit string validates cleanly.

### Family B: Weighted Modulo-11 Checksum & Format
- **Identifier Field:** `registry_id`
- **Format Constraints:**
  - Length: Exactly 10 characters.
  - Character Class: Numeric digits only (`0-9`).
  - Leading Digit: Must be in the range `[1, 9]` (leading `0` is format-invalid).
- **Checksum Algorithm:** **Weighted Modulo-11 Checksum**:
  - Weight Vector: $W = [1, 2, 3, 4, 5, 6, 7, 8, 9]$ applied to the 9-digit body from left to right.
  - Checksum Formula: $S = \sum_{i=0}^{8} \text{body}[i] \times W[i]$.
  - Remainder: $R = S \pmod{11}$.
  - Valid check digit: $R$ when $R \in [0, 9]$. If $R = 10$, no single decimal digit check digit can represent it (rejected by benchmark generation rules).
  - Validation: Computes expected check digit over the first 9 digits and verifies it matches the 10th digit.

### Dispatcher Helpers
- `get_identifier_field(document_family: str) -> str`: Maps `"family_a"` $\to$ `"id_number"`, `"family_b"` $\to$ `"registry_id"`.
- `validate_format(document_family: str, identifier: str) -> bool`: Dispatches to `family_a_format_is_valid` or `family_b_format_is_valid`.
- `validate_checksum(document_family: str, identifier: str) -> bool`: Dispatches to `family_a_checksum_is_valid` or `family_b_checksum_is_valid`.

---

## 3. Cross-Modal Consistency Engine (`src/m3_crossmodal/consistency.py`)

The consistency engine receives extracted OCR fields, decoded QR fields, and the document family, producing the exact 5-feature consistency vector:

```python
{
    "qr_readable": bool,
    "text_qr_match_score": float | None,
    "checksum_valid": bool,
    "format_valid": bool,
    "missing_field_count": int,
}
```

### Key Semantics & Behavioral Specifications

1. **QR Unreadable Semantics:**
   - `qr_readable = False` indicates that QR data was unreadable or unavailable (empty or `None` `qr_fields`).
   - `qr_readable = False` **DOES NOT mean the document is forged**. It represents evidence unavailability.
   - When QR is unreadable, `text_qr_match_score` is explicitly set to `None` (NOT `0.0`), reflecting that no cross-modal comparison was possible.

2. **OCR Missing-Data Semantics:**
   - Missing or empty OCR fields indicate evidence unavailability, NOT an automatic forgery flag.
   - Any decision to flag forgery belongs solely to downstream classifiers, not to the feature extraction engine.
   - Internal metadata fields (such as `_raw_ocr_text`) are explicitly ignored when computing field similarity or counting missing fields.

3. **Format vs. Checksum Independence:**
   - As proven by audit against M1 ground truth (e.g. `family_a_000004_format_invalid.json`), `checksum_valid` is evaluated independently of `format_valid`.
   - An identifier may fail format constraints (e.g. initial digit `0` or `1`) while still mathematically satisfying the Verhoeff or Mod-11 algorithm across its digits.

4. **Text / QR Similarity Metric:**
   - Uses `difflib.SequenceMatcher(None, ocr_val, qr_val).ratio()` across common keys.
   - Text normalization strips leading/trailing whitespace without stripping punctuation or collapsing case, preserving exact string fidelity (e.g. dates, titles, and abbreviations).
   - The resulting mean score across common fields is rounded to 3 decimal places.

---

## 4. Verification Methodology & Full Dataset Audit (`src/m3_crossmodal/verification.py`)

### Verification Tooling
The `verification.py` diagnostic module compares freshly computed consistency vectors against M1's stored `consistency_vector` across all 11,000 label JSON files in `~/Downloads/dataset_final_v2/labels`.

- Strict read-only operation: Never alters, overwrites, or touches any M1 source files.
- Strict leakage guard: Never passes ground truth metadata (`ground_truth_fields`, `final_label`, `tamper_type`, `cnn_label`, `expected_consistency_vector`) to feature extraction.

### Full 11,000-Record Verification Results

Ran on all **11,000** ground-truth records:

| Feature / Metric | Matches / Total | Match Percentage | Mismatches |
| :--- | :--- | :--- | :--- |
| **`qr_readable`** | 11,000 / 11,000 | **100.00%** | 0 |
| **`checksum_valid`** | 11,000 / 11,000 | **100.00%** | 0 |
| **`format_valid`** | 11,000 / 11,000 | **100.00%** | 0 |
| **`missing_field_count`** | 10,986 / 11,000 | **99.87%** | 14 |
| **`text_qr_match_score` (exact)** | 10,178 / 11,000 | **92.53%** | 822 |
| **`text_qr_match_score` ($\le 0.05$)** | 10,963 / 11,000 | **99.66%** | 37 |
| **Full 5-Vector Exact Match** | 10,166 / 11,000 | **92.42%** | 834 |
| **Full 5-Vector Within Tolerance ($\le 0.05$)**| 10,949 / 11,000 | **99.54%** | 51 |

**Score Difference Summary:**
- Mean Absolute Error (all 11,000 records): **`0.00167`**
- Maximum Absolute Error: **`0.1000`**
- Within $\pm 0.001$: **10,179 records** (92.54%)
- Within $\pm 0.010$: **10,337 records** (93.97%)
- Within $\pm 0.050$: **10,963 records** (99.66%)

---

## 5. Root Cause Analysis of Remaining Mismatches

### A. Missing Field Count Mismatches (14 records / 0.13%)
- **Identified Cause:** In 14 records (e.g. `family_a_000720_field_missing.json`, `family_a_000498_checksum_invalid.json`), M1's synthetic OCR layout simulator lumped multiple lines into a single field (e.g., merging `dob` with `gender` and `id_number`).
- **Explanation:** M1 originally recorded the synthetic attack plan's missing field count (`expected_consistency_vector`), whereas the actual OCR output had an additional empty field due to the parsing collision. Our implementation strictly and correctly inspects `ocr_fields`, which is the only valid runtime input.

### B. Text / QR Match Score Mismatches (822 records / 7.47%)
- **Identified Causes:**
  1. **OCR Footer Watermark Ingestion (Family B):** In some Family B records (e.g. `family_b_004857_fine_grained_edit.json`), the OCR bounding box for `registration_date` ingested part of the synthetic benchmark footer disclaimer ("This is a fictional document generated for research purposes only"), lowering the difflib similarity ratio against the clean QR date.
  2. **Attack Generator Scoring Variance (30 fine_grained_edit records):** In 30 out of 1,000 fine-grained edit records, the stored score reflects an exact 0.10 penalty recorded by M1's attack generator rather than the field-average SequenceMatcher score.
  3. **High Overall Alignment:** For 92.53% of records, SequenceMatcher matches M1's stored score to 3 decimal places. The remaining 7.47% have a mean difference of only 0.022, confirming the underlying metric is difflib SequenceMatcher without speculative heuristic modifications.

---

## 6. Unit & Integration Test Suite

All **46 unit tests** pass cleanly:
```bash
pytest tests/ -v
```

- **`tests/test_part1.py` (9 tests):** Label mapping, feature ordering, leakage guard, path resolution, shape/finite checks, identity separation.
- **`tests/test_validators.py` (16 tests):** Family A Verhoeff valid/invalid, length/digit/first-digit format checks; Family B Mod-11 valid/invalid, length/digit/zero format checks; dispatchers and unknown family exceptions.
- **`tests/test_consistency.py` (21 tests):** QR readability, unreadable QR giving `None`, missing OCR field counts, internal key exclusion, normalization, similarity metric, dispatch, types, and format-invalid/checksum-valid independence.
