"""
live_ocr.py
-----------
Live OCR extraction from document images.

Pipeline step:
    image -> raw OCR text -> best-effort structured field dictionary

Operates ONLY on image pixels. Never reads M1's precomputed fields,
ground truth, tamper_type, or filenames to decide field values.
OCR failure is represented explicitly as unavailable evidence and is
never treated as evidence of forgery.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .image_utils import ImageLoadError, load_image_as_array


class OCRBackendError(Exception):
    """Raised by an OCR backend when text extraction fails."""


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class OCRBackend:
    def extract_text(self, image_array: np.ndarray) -> str:
        raise NotImplementedError


class PytesseractOCRBackend(OCRBackend):
    """Default backend: local Tesseract via pytesseract."""

    def extract_text(self, image_array: np.ndarray) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise OCRBackendError(
                "pytesseract/Pillow not installed. Ensure pytesseract and Pillow are installed."
            ) from e

        try:
            return pytesseract.image_to_string(Image.fromarray(image_array))
        except Exception as e:
            raise OCRBackendError(f"Tesseract OCR failed: {e}") from e


_default_backend: Optional[OCRBackend] = None


def get_default_ocr_backend() -> OCRBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = PytesseractOCRBackend()
    return _default_backend


def set_default_ocr_backend(backend: OCRBackend) -> None:
    """Allows tests or callers to swap the backend."""
    global _default_backend
    _default_backend = backend


# ---------------------------------------------------------------------------
# Low-level OCR call
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    success: bool
    raw_text: Optional[str]
    error: Optional[str]


def run_ocr(image_path: str, backend: Optional[OCRBackend] = None) -> OCRResult:
    """
    Runs OCR on a single image and returns raw text or failure information.
    Never raises — missing files or engine errors return OCRResult(success=False, ...).
    """
    backend = backend or get_default_ocr_backend()

    try:
        image_array = load_image_as_array(image_path)
    except ImageLoadError as e:
        return OCRResult(success=False, raw_text=None, error=str(e))

    try:
        raw_text = backend.extract_text(image_array)
    except OCRBackendError as e:
        return OCRResult(success=False, raw_text=None, error=str(e))

    return OCRResult(success=True, raw_text=raw_text, error=None)


# ---------------------------------------------------------------------------
# Family field schemas + parsers
# ---------------------------------------------------------------------------

FAMILY_A_FIELDS = ["name", "dob", "gender", "id_number", "address"]
FAMILY_B_FIELDS = ["full_name", "registry_id", "entity_type", "jurisdiction_code", "registration_date"]

FAMILY_EXPECTED_FIELDS = {
    "family_a": FAMILY_A_FIELDS,
    "family_b": FAMILY_B_FIELDS,
}

_FAMILY_A_LABELS = {
    "name": ["name"],
    "dob": ["dob", "date of birth"],
    "gender": ["gender", "sex"],
    "id_number": ["id number", "id no", "id"],
    "address": ["address"],
}

_FAMILY_B_LABELS = {
    "full_name": ["registrant", "full name", "name"],
    "registry_id": ["registry id", "registry no", "registry number"],
    "entity_type": ["entity type", "type"],
    "jurisdiction_code": ["jurisdiction", "jurisdiction code"],
    "registration_date": ["registered on", "registration date", "reg date"],
}


def _empty_fields(field_names) -> Dict[str, Optional[str]]:
    return {name: None for name in field_names}


def _clean_label_candidate(text: str) -> str:
    """Strip trailing colons/hyphens and whitespace, return lowercased."""
    return re.sub(r"[:\-]+$", "", text).strip().lower()


def _parse_labeled_lines(
    raw_text: str,
    label_map: Dict[str, list],
    is_multiline_field: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Robust line parser supporting both:
    1. Same-line format: 'Label: Value' / 'Label - Value'
    2. Next-line format: 'Label:' on one line, value on subsequent line(s).
    """
    result: Dict[str, Optional[str]] = {field: None for field in label_map}
    if not raw_text:
        return result

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    
    # Build quick lookup sets for all label alternatives
    field_for_label = {}
    field_for_nospace = {}
    for canonical_field, label_alternatives in label_map.items():
        for alt in label_alternatives:
            field_for_label[alt.lower()] = canonical_field
            field_for_nospace[alt.replace(" ", "").lower()] = canonical_field

    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. Try same-line match: Label: Value
        same_line_match = re.match(r"^(.*?)[-:]\s*(.+)$", line)
        matched_field = None
        if same_line_match:
            lbl_raw = same_line_match.group(1).strip().lower()
            val_part = same_line_match.group(2).strip()
            # Match against known labels
            fld = field_for_label.get(lbl_raw) or field_for_nospace.get(lbl_raw.replace(" ", ""))
            if fld and result[fld] is None:
                result[fld] = val_part
                matched_field = fld

        # 2. Try next-line match: current line is a label only
        if not matched_field:
            candidate = _clean_label_candidate(line)
            fld = field_for_label.get(candidate) or field_for_nospace.get(candidate.replace(" ", ""))
            if fld and result[fld] is None:
                # Collect value line(s)
                val_lines = []
                j = i + 1
                while j < len(lines):
                    next_cand = _clean_label_candidate(lines[j])
                    is_next_label = (
                        next_cand in field_for_label
                        or next_cand.replace(" ", "") in field_for_nospace
                    )
                    if is_next_label:
                        break
                    val_lines.append(lines[j])
                    # Non-multiline fields only take 1 line
                    if fld != is_multiline_field:
                        break
                    j += 1

                if val_lines:
                    result[fld] = " ".join(val_lines)
                    i = j - 1

        i += 1

    return result


def parse_family_a_fields(raw_text: Optional[str]) -> Dict[str, Optional[str]]:
    """OCR text -> {name, dob, gender, id_number, address}."""
    if not raw_text:
        return _empty_fields(FAMILY_A_FIELDS)
    return _parse_labeled_lines(raw_text, _FAMILY_A_LABELS, is_multiline_field="address")


def parse_family_b_fields(raw_text: Optional[str]) -> Dict[str, Optional[str]]:
    """OCR text -> {full_name, registry_id, entity_type, jurisdiction_code, registration_date}."""
    if not raw_text:
        return _empty_fields(FAMILY_B_FIELDS)
    return _parse_labeled_lines(raw_text, _FAMILY_B_LABELS)


FAMILY_FIELD_PARSERS = {
    "family_a": parse_family_a_fields,
    "family_b": parse_family_b_fields,
}


def parse_ocr_fields(raw_text: Optional[str], document_family: str) -> Dict[str, Optional[str]]:
    try:
        parser = FAMILY_FIELD_PARSERS[document_family]
    except KeyError as e:
        raise ValueError(f"Unknown document_family: {document_family!r}") from e
    return parser(raw_text)


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def extract_ocr_fields(
    image_path: str,
    document_family: str,
    backend: Optional[OCRBackend] = None,
) -> Dict[str, Optional[str]]:
    """
    Runs OCR on image_path and returns structured fields for document_family.
    On failure, returns all-None fields (evidence unavailable).
    """
    if document_family not in FAMILY_EXPECTED_FIELDS:
        raise ValueError(f"Unknown document_family: {document_family!r}")

    ocr_result = run_ocr(image_path, backend=backend)
    if not ocr_result.success:
        return _empty_fields(FAMILY_EXPECTED_FIELDS[document_family])
    return parse_ocr_fields(ocr_result.raw_text, document_family)


def extract_ocr_debug(
    image_path: str,
    document_family: str,
    backend: Optional[OCRBackend] = None,
) -> Dict[str, object]:
    """
    Debug variant exposing raw OCR text alongside parsed fields.
    """
    if document_family not in FAMILY_EXPECTED_FIELDS:
        raise ValueError(f"Unknown document_family: {document_family!r}")

    ocr_result = run_ocr(image_path, backend=backend)
    if not ocr_result.success:
        fields = _empty_fields(FAMILY_EXPECTED_FIELDS[document_family])
    else:
        fields = parse_ocr_fields(ocr_result.raw_text, document_family)

    return {
        "fields": fields,
        "ocr_success": ocr_result.success,
        "raw_text": ocr_result.raw_text,
        "error": ocr_result.error,
    }
