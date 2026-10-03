"""
part4_features.py
-----------------
Part 4 Feature Engineering for M3 Cross-Modal ML Classifier.

Extracts an expanded 16-feature tabular representation from live OCR, live QR,
and Part 2 consistency outputs.

Guarantees:
- Zero data leakage: forbidden generator/manifest/split columns strictly rejected.
- Deterministic feature ordering across both document families.
- Missing values preserved as NaN (non-comparable), NEVER converted to 0.0.
- Decoupled from M1 precomputed values (operates strictly on live pipeline outputs).
"""

from typing import Any, Dict, Iterable, List, Optional, Set
import numpy as np

from .consistency import field_similarity, normalize_text
from .validators import get_expected_fields, get_identifier_field


# ---------------------------------------------------------------------------
# Forbidden Features & Leakage Guards
# ---------------------------------------------------------------------------

FORBIDDEN_FEATURE_FIELDS: Set[str] = {
    "ground_truth_fields",
    "expected_consistency_vector",
    "tamper_type",
    "final_label",
    "cnn_label",
    "record_id",
    "manifest_id",
    "image_path",
    "label_path",
    "splice_metadata",
    "fine_grained_metadata",
    "target",
    "id",
}


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    """
    Asserts that no forbidden or leakage column exists in the given column list.
    Raises ValueError immediately if any forbidden column is found.
    """
    violating = [c for c in columns if c.lower() in FORBIDDEN_FEATURE_FIELDS or c.lower().endswith(("_path", "_label"))]
    if violating:
        raise ValueError(
            f"LEAKAGE VIOLATION: Forbidden columns found in feature matrix: {violating}. "
            f"Allowed features must be derived strictly from live cross-modal outputs."
        )


# ---------------------------------------------------------------------------
# Family Field Mappings (Deterministic 5 slots)
# ---------------------------------------------------------------------------

FAMILY_FIELD_SLOTS: Dict[str, List[str]] = {
    "family_a": ["name", "dob", "gender", "id_number", "address"],
    "family_b": ["full_name", "registry_id", "entity_type", "jurisdiction_code", "registration_date"],
}

# Canonical feature ordering
CORE_CONSISTENCY_FEATURES = [
    "qr_readable",
    "text_qr_match_score",
    "checksum_valid",
    "format_valid",
    "missing_field_count",
]

COVERAGE_FEATURES = [
    "ocr_field_presence_count",
    "ocr_field_presence_rate",
    "qr_field_count",
    "ocr_qr_key_overlap_count",
]

FIELD_SIMILARITY_FEATURES = [
    "field_similarity__0",
    "field_similarity__1",
    "field_similarity__2",
    "field_similarity__3",
    "field_similarity__4",
]

VALIDATOR_DETAIL_FEATURES = [
    "checksum_computed",
]

CONTEXT_FEATURES = [
    "document_family",
]

PART4_FEATURE_ORDER_NO_FAMILY = (
    CORE_CONSISTENCY_FEATURES
    + COVERAGE_FEATURES
    + FIELD_SIMILARITY_FEATURES
    + VALIDATOR_DETAIL_FEATURES
)

PART4_FEATURE_ORDER_WITH_FAMILY = PART4_FEATURE_ORDER_NO_FAMILY + CONTEXT_FEATURES


# ---------------------------------------------------------------------------
# Feature Extraction Functions
# ---------------------------------------------------------------------------

def compute_field_similarities(
    ocr_fields: Optional[Dict[str, Any]],
    qr_fields: Optional[Dict[str, Any]],
    document_family: str,
) -> Dict[str, float]:
    """
    Computes continuous [0, 1] similarity scores for each of the 5 canonical
    field slots of the given family.

    Critical Missingness Policy:
    If a field is missing in OCR, missing in QR, or QR is unreadable, the similarity
    is np.nan (NOT 0.0). Missingness must never be fabricated into false-forgery mismatch.
    """
    if document_family not in FAMILY_FIELD_SLOTS:
        raise ValueError(f"Unknown document_family: {document_family!r}")

    expected_fields = FAMILY_FIELD_SLOTS[document_family]
    ocr = ocr_fields or {}
    qr = qr_fields or {}

    sims: Dict[str, float] = {}
    for i, field_name in enumerate(expected_fields):
        col_name = f"field_similarity__{i}"
        val_ocr = ocr.get(field_name)
        val_qr = qr.get(field_name)

        if val_ocr is None or val_qr is None:
            sims[col_name] = np.nan
        else:
            norm_ocr = normalize_text(val_ocr)
            norm_qr = normalize_text(val_qr)
            if not norm_ocr or not norm_qr:
                sims[col_name] = np.nan
            else:
                score = field_similarity(norm_ocr, norm_qr)
                sims[col_name] = float(score)

    return sims


def extract_part4_features(
    ocr_fields: Optional[Dict[str, Any]],
    qr_fields: Optional[Dict[str, Any]],
    consistency_vector: Dict[str, Any],
    document_family: str,
) -> Dict[str, float]:
    """
    Extracts the full dictionary of 16 features from live OCR, live QR,
    consistency vector, and document family.

    Returns a dict containing all 16 feature values (with NaNs preserved).
    """
    if document_family not in FAMILY_FIELD_SLOTS:
        raise ValueError(f"Unknown document_family: {document_family!r}")

    expected_fields = FAMILY_FIELD_SLOTS[document_family]
    ocr = ocr_fields or {}
    qr = qr_fields or {}

    # A. Core 5 Consistency Features
    qr_readable = 1.0 if consistency_vector.get("qr_readable") else 0.0
    
    raw_match_score = consistency_vector.get("text_qr_match_score")
    # Preserve NaN if match score is None/unreadable
    text_qr_match_score = float(raw_match_score) if raw_match_score is not None else np.nan
    
    checksum_valid = 1.0 if consistency_vector.get("checksum_valid") else 0.0
    format_valid = 1.0 if consistency_vector.get("format_valid") else 0.0
    missing_field_count = float(consistency_vector.get("missing_field_count", 0.0))

    # B. Coverage Features
    ocr_present_keys = [
        k for k in expected_fields if ocr.get(k) is not None and str(ocr.get(k)).strip() != ""
    ]
    ocr_field_presence_count = float(len(ocr_present_keys))
    ocr_field_presence_rate = float(len(ocr_present_keys)) / float(len(expected_fields))

    qr_present_keys = [
        k for k in expected_fields if qr.get(k) is not None and str(qr.get(k)).strip() != ""
    ]
    qr_field_count = float(len(qr_present_keys))

    common_keys = set(ocr_present_keys).intersection(set(qr_present_keys))
    ocr_qr_key_overlap_count = float(len(common_keys))

    # C. Per-Field Similarities
    field_sims = compute_field_similarities(ocr, qr, document_family)

    # D. Validator Detail Feature
    identifier_field = get_identifier_field(document_family)
    raw_id = ocr.get(identifier_field)
    checksum_computed = 1.0 if raw_id is not None and str(raw_id).strip() != "" else 0.0

    # E. Optional Context Feature: document_family
    # 0.0 = family_a, 1.0 = family_b
    fam_numeric = 0.0 if document_family == "family_a" else 1.0

    feat_dict: Dict[str, float] = {
        "qr_readable": qr_readable,
        "text_qr_match_score": text_qr_match_score,
        "checksum_valid": checksum_valid,
        "format_valid": format_valid,
        "missing_field_count": missing_field_count,
        "ocr_field_presence_count": ocr_field_presence_count,
        "ocr_field_presence_rate": ocr_field_presence_rate,
        "qr_field_count": qr_field_count,
        "ocr_qr_key_overlap_count": ocr_qr_key_overlap_count,
        "checksum_computed": checksum_computed,
        "document_family": fam_numeric,
    }
    feat_dict.update(field_sims)

    return feat_dict


def get_feature_vector(
    feat_dict: Dict[str, float],
    include_family: bool = False,
) -> np.ndarray:
    """
    Returns an ordered 1D numpy array from the feature dictionary according to
    PART4_FEATURE_ORDER_NO_FAMILY (15 features) or
    PART4_FEATURE_ORDER_WITH_FAMILY (16 features).
    """
    feature_order = PART4_FEATURE_ORDER_WITH_FAMILY if include_family else PART4_FEATURE_ORDER_NO_FAMILY
    return np.array([feat_dict[k] for k in feature_order], dtype=np.float64)
