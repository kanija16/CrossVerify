# Member 3 (M3) Phase 1 — Part 3: Live OCR & Live QR

## 1. Overview & Architecture

Part 3 connects raw document image pixels to structured field dictionaries via live OCR and live QR detection, and hands those dictionaries directly to Part 2's existing consistency engine:

```
                    DOCUMENT IMAGE
                         |
              +----------+----------+
              |                     |
              v                     v
          LIVE OCR              LIVE QR
       (pytesseract)         (OpenCV QR)
              |                     |
              v                     v
       OCR field dict        QR field dict
              |                     |
              +----------+----------+
                         |
                         v
              Part 2 consistency engine
            (compute_consistency_vector)
                         |
                         v
                 5 consistency fields:
                 - qr_readable
                 - text_qr_match_score
                 - checksum_valid
                 - format_valid
                 - missing_field_count
```

> **CRITICAL DISCLAIMER (FICTIONAL BENCHMARK CONSTRUCTIONS):**  
> All document families, layouts, formats, check-digit algorithms, and QR payload encodings are **purely synthetic, fictional benchmark constructions** created specifically for multimodal forgery detection research. They do NOT reflect, model, reverse-engineer, or authenticate any real-world identity systems, registries, or credentials.

---

## 2. Module Implementations

### A. Shared Image Utilities (`src/m3_crossmodal/image_utils.py`)
- Loads image from path and returns RGB numpy array `(H, W, 3)`.
- Raises `ImageLoadError` uniformly for missing files, corrupt files, or unsupported image formats.

### B. Live OCR Module (`src/m3_crossmodal/live_ocr.py`)
- **Backend Architecture:** Pluggable `OCRBackend` interface with default `PytesseractOCRBackend` interfacing with local `tesseract` binary via `pytesseract`.
- **Low-level Execution:** `run_ocr(image_path) -> OCRResult(success, raw_text, error)`. Never raises exceptions on corrupt images or engine errors.
- **Family Field Schemas:**
  - **Family A:** `["name", "dob", "gender", "id_number", "address"]`
  - **Family B:** `["full_name", "registry_id", "entity_type", "jurisdiction_code", "registration_date"]`
- **Field Parsers:** Regex-based line parsers recognizing standard label alternatives (e.g., `DOB:`, `Date of Birth:`, `Registry ID:`).
- **Semantics:** Returns `{field: None}` on missing/unparseable lines or complete OCR failure. Raw OCR text is never mixed into clean output fields.
- **Leakage Isolation:** Operates strictly on image pixels. Never reads filenames, record IDs, label JSONs, or tamper metadata.

### C. Live QR Module (`src/m3_crossmodal/live_qr.py`)
- **Backend Architecture:** Pluggable `QRBackend` interface with default `OpenCVQRBackend` using `cv2.QRCodeDetector`.
- **Detection Flow:**
  1. Whole-image detection attempt first.
  2. Fallback to documented family corner region hint if whole-image fails:
     - **Family A:** 1000x640 canvas, QR located at **bottom-right**.
     - **Family B:** 1150x520 canvas, QR located at **top-right**.
     - Crop corner (40% width/height) and upscale (3x, 4x, 2x) to reliably decode small QR codes.
- **Payload Parsing:** Modular parser supporting JSON payloads (primary) and delimited key-value formats (`key:value;`, `key=value|`).
- **Semantics:** Returns `qr_readable=False` and `qr_decoded_fields={}` on failure without raising. Never classifies QR failure as forgery.

### D. Live Pipeline (`src/m3_crossmodal/live_pipeline.py`)
- **Entry Point:** `run_live_crossmodal(image_path, document_family, ...)`
- **Execution:**
  1. Runs `extract_live_modal_fields(image_path, document_family)` returning `ocr_extracted_fields` and `qr_decoded_fields`.
  2. Passes dictionaries directly to Part 2's `compute_consistency_vector(document_family, ocr_fields, qr_fields)`.
  3. When OCR produces no extracted text (all fields `None` on failure or non-text image), passes `{}` to Part 2 so `text_qr_match_score` is `None` (evidence unavailable, not 0.0).
  4. Returns:
     ```python
     {
         "ocr_extracted_fields": {...},
         "qr_decoded_fields": {...},
         "consistency_vector": {
             "qr_readable": bool,
             "text_qr_match_score": float | None,
             "checksum_valid": bool,
             "format_valid": bool,
             "missing_field_count": int,
         }
     }
     ```

### E. Diagnostic Runner (`src/m3_crossmodal/diagnostics.py`)
- Diagnostic batch evaluation tool for checking OCR/QR extraction against dataset CSV splits.
- Complies strictly with M1's `document_family` column schema.
- Reports OCR text rate, structured field availability, QR decoding success, and per-field presence rates.

---

## 3. Integration & Protection of Earlier Parts

- **Part 1 Protection:** All Part 1 baseline and data interface modules remain untouched and operational.
- **Part 2 Protection:** Part 2's consistency engine and validators remain the authoritative implementation. Part 3 adapts to Part 2 without modifying Part 2 files.
- **Leakage Guard:** Live extraction accepts only `image_path` and explicit `document_family`. No metadata enters the extraction process.

---

## 4. Dependencies & Setup

- **Python Packages:**
  - `pytesseract>=0.3.10`
  - `opencv-python-headless>=4.8`
  - `Pillow>=10.0`
  - `qrcode>=7.4` (test fixture generation)
  - `pytest>=7.0`
- **System Binary:** Local Tesseract OCR binary (installable via `brew install tesseract` on macOS or `apt-get install tesseract-ocr` on Ubuntu).
