"""
consistency.py
---------------
Cross-modal consistency engine for Member 3 (M3).

Takes OCR-extracted fields and QR-decoded fields (already decoded —
this module does NOT do OCR or QR decoding itself) plus the document
family, and returns exactly the 5 M3 consistency fields:

    qr_readable, text_qr_match_score, checksum_valid,
    format_valid, missing_field_count

Key semantics (per spec):
- qr_readable=False means QR evidence is unavailable, NOT that the
  document is forged.
- When QR is unreadable, text_qr_match_score is None (not 0) — there is
  nothing to compare, which is different from "compared and mismatched".
- Missing/empty OCR data is evidence unavailable, not automatic forgery.
  This engine only reports the state; any forgery decision belongs to a
  later classifier.
- Never reads ground_truth_fields, expected_consistency_vector,
  tamper_type, final_label, cnn_label, or any other metadata — this
  engine only ever looks at document_family, ocr_fields, and qr_fields.
"""

import difflib
from typing import Any, Dict, Optional

from . import validators

# Canonical expected field names per family (excluding internal metadata)
CANONICAL_FIELDS = {
    "family_a": ["name", "dob", "gender", "id_number", "address"],
    "family_b": ["full_name", "registry_id", "entity_type", "jurisdiction_code", "registration_date"],
}


# ---------------------------------------------------------------------------
# Text normalization / similarity
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """Basic whitespace trimming and string conversion."""
    if value is None:
        return ""
    return str(value).strip()


def field_similarity(a: str, b: str) -> float:
    """
    Compute similarity metric between two strings using difflib.SequenceMatcher.ratio().
    Returns 1.0 for two empty strings, or a float in [0.0, 1.0].
    """
    if a == "" and b == "":
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# QR readability
# ---------------------------------------------------------------------------

def is_qr_readable(qr_fields: Optional[Dict[str, Any]]) -> bool:
    """
    QR is considered readable if non-empty decoded fields are provided.
    This engine does not decode QR codes itself — an empty/None dict
    represents an unreadable QR code.
    """
    return bool(qr_fields)


# ---------------------------------------------------------------------------
# Missing OCR field count
# ---------------------------------------------------------------------------

def count_missing_ocr_fields(
    ocr_fields: Optional[Dict[str, Any]],
    document_family: Optional[str] = None,
) -> int:
    """
    Counts entries in ocr_fields whose value is missing or empty
    (None, or an empty/whitespace-only string).
    Internal fields (e.g. starting with '_') are ignored.
    If ocr_fields is empty or None, returns 0.
    """
    if not ocr_fields:
        return 0
    count = 0
    for key, value in ocr_fields.items():
        if key.startswith("_"):
            continue
        if value is None:
            count += 1
        elif isinstance(value, str) and value.strip() == "":
            count += 1
    return count


# ---------------------------------------------------------------------------
# OCR <-> QR text matching
# ---------------------------------------------------------------------------

def compute_text_qr_match_score(
    ocr_fields: Optional[Dict[str, Any]],
    qr_fields: Optional[Dict[str, Any]],
) -> Optional[float]:
    """
    Field-level comparison between OCR and QR data across common fields
    (excluding internal metadata keys starting with '_').
    Returns the average per-field similarity rounded to 3 decimal places,
    or None when:
    - QR is unreadable (qr_fields empty/None)
    - OCR is unavailable (ocr_fields empty/None)
    - No common fields exist between the two dicts
    """
    if not qr_fields or not ocr_fields:
        return None

    common_keys = sorted(k for k in ocr_fields.keys() if k in qr_fields and not k.startswith("_"))
    if not common_keys:
        return None

    scores = []
    for key in common_keys:
        ocr_val = normalize_text(ocr_fields.get(key))
        qr_val = normalize_text(qr_fields.get(key))
        scores.append(field_similarity(ocr_val, qr_val))

    mean_score = sum(scores) / len(scores)
    return round(mean_score, 3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_consistency_vector(
    document_family: str,
    ocr_fields: Optional[Dict[str, Any]],
    qr_fields: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes the 5-element consistency vector for a document:
        qr_readable         : bool
        text_qr_match_score : float | None
        checksum_valid      : bool
        format_valid        : bool
        missing_field_count : int

    document_family : "family_a" or "family_b"
    ocr_fields      : dict of OCR-extracted fields (may be None/empty)
    qr_fields       : dict of QR-decoded fields (may be None/empty)
    """
    ocr_fields = ocr_fields or {}
    qr_fields = qr_fields or {}

    qr_readable = is_qr_readable(qr_fields)
    text_qr_match_score = compute_text_qr_match_score(ocr_fields, qr_fields)

    identifier_field = validators.get_identifier_field(document_family)
    raw_identifier = ocr_fields.get(identifier_field)
    identifier = "" if raw_identifier is None else str(raw_identifier).strip()

    format_valid = validators.validate_format(document_family, identifier)
    # Checksum is evaluated independently of format validity:
    # An identifier can fail format constraints (e.g. invalid initial digit)
    # while still satisfying the checksum algorithm, as audited against M1 ground truth.
    checksum_valid = validators.validate_checksum(document_family, identifier)

    missing_field_count = count_missing_ocr_fields(ocr_fields, document_family)

    return {
        "qr_readable": qr_readable,
        "text_qr_match_score": text_qr_match_score,
        "checksum_valid": checksum_valid,
        "format_valid": format_valid,
        "missing_field_count": missing_field_count,
    }
