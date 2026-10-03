"""
features.py
-----------
Feature extraction for M3 Phase 1 — Part 1: Precomputed Consistency Baseline.

This module is the single place responsible for converting raw M3Record objects
and consistency vectors into model-ready numeric arrays.

Strict constraints:
- Input vector has exactly 5 dimensions in this exact order:
    1. qr_readable
    2. text_qr_match_score
    3. checksum_valid
    4. format_valid
    5. missing_field_count
- No NLP, no OCR text flattening, and no image embeddings in Part 1.
- No leakage fields (tamper_type, record_id, final_label, paths) enter X.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

from . import config


def extract_consistency_features(
    consistency_vector: Dict[str, Any],
    impute_missing_score: float = config.DEFAULT_IMPUTE_TEXT_QR_MATCH_SCORE
) -> np.ndarray:
    """
    Extract the 5 precomputed consistency metrics in strict canonical order:
      [qr_readable, text_qr_match_score, checksum_valid, format_valid, missing_field_count]

    Coercion & Imputation Policy:
      - qr_readable: bool/int -> 1.0 (readable) or 0.0 (unreadable)
      - text_qr_match_score: float in [0.0, 1.0]. If None or missing, impute with
        impute_missing_score (0.0). Retains exact 5-feature shape.
      - checksum_valid: bool/int -> 1.0 (valid) or 0.0 (invalid)
      - format_valid: bool/int -> 1.0 (valid) or 0.0 (invalid)
      - missing_field_count: int/float -> float count in [0.0, 5.0]

    Raises:
      ValueError if any required key is missing from consistency_vector.
    """
    values = []
    for key in config.CONSISTENCY_FEATURE_ORDER:
        if key not in consistency_vector:
            raise ValueError(
                f"consistency_vector is missing required key '{key}'. "
                f"Keys present: {list(consistency_vector.keys())}"
            )
        v = consistency_vector[key]

        if key == "text_qr_match_score":
            if v is None:
                # Documented imputation strategy: impute 0.0 for uncomputed/missing match
                v = impute_missing_score
            else:
                v = float(v)
        elif isinstance(v, bool):
            v = 1.0 if v else 0.0
        elif v is None:
            v = 0.0
        else:
            v = float(v)

        values.append(float(v))

    arr = np.array(values, dtype=np.float64)
    if len(arr) != config.EXPECTED_NUM_FEATURES:
        raise ValueError(
            f"Expected {config.EXPECTED_NUM_FEATURES} features, got {len(arr)}"
        )
    return arr


def build_baseline_matrix(records: List[Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build model-ready matrices from a list of M3Record instances.

    Parameters:
      records: List of M3Record instances

    Returns:
      X: np.ndarray of shape (n_samples, 5), dtype=float64
      y: np.ndarray of shape (n_samples,), dtype=int64 (0=genuine, 1=forged)
      groups: List[str] of record_id per sample (for grouping/leakage verification)

    Guarantees:
      - X has no NaN or infinite values.
      - No forbidden fields enter X.
    """
    if not records:
        return (
            np.empty((0, config.EXPECTED_NUM_FEATURES), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            []
        )

    X_list = []
    y_list = []
    groups = []

    for r in records:
        feat = extract_consistency_features(r.consistency_vector)
        X_list.append(feat)
        y_list.append(int(r.target))
        groups.append(str(r.record_id))

    X = np.vstack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=np.int64)

    # Sanity verification on matrix integrity
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix X contains NaN or infinite values.")

    if X.shape[1] != config.EXPECTED_NUM_FEATURES:
        raise ValueError(
            f"Feature matrix dimension mismatch: expected {config.EXPECTED_NUM_FEATURES} columns, "
            f"got {X.shape[1]}."
        )

    return X, y, groups


FEATURE_NAMES = list(config.CONSISTENCY_FEATURE_ORDER)
