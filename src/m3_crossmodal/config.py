"""
config.py
---------
Central configuration for Member 3 (M3) Phase 1 — Part 1:
Data Interface & Precomputed Consistency Baseline.

This file centralizes all paths, schema column names, feature orderings,
label encodings, and leakage boundaries.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — M1 Source Data
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("~/Downloads/dataset_final_v2").expanduser()

M1_DIR = Path("~/Desktop").expanduser()
MANIFEST_PATH = M1_DIR / "manifest.json"
RECORD_SPLIT_MAP_PATH = M1_DIR / "record_id_split_map.csv"
TRAIN_CSV = M1_DIR / "train.csv"
VAL_CSV = M1_DIR / "val.csv"
TEST_CSV = M1_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Verified M1 CSV & Manifest Column Names
# Note: The document family column in M1's schema is 'document_family', NOT 'family'.
# ---------------------------------------------------------------------------
COL_ID = "id"
COL_RECORD_ID = "record_id"
COL_DOCUMENT_FAMILY = "document_family"
COL_TEMPLATE_VARIANT = "template_variant"
COL_TAMPER_TYPE = "tamper_type"
COL_FINAL_LABEL = "final_label"
COL_CNN_LABEL = "cnn_label"
COL_IMAGE_PATH = "image_path"       # Bookkeeping / path resolution only — never a feature
COL_LABEL_PATH = "label_path"       # Bookkeeping / path resolution only — never a feature

# Backward compatibility alias
COL_FAMILY = COL_DOCUMENT_FAMILY

# ---------------------------------------------------------------------------
# Label JSON Blocks (M3 Owned)
# ---------------------------------------------------------------------------
KEY_OCR_FIELDS = "ocr_extracted_fields"
KEY_QR_FIELDS = "qr_decoded_fields"
KEY_CONSISTENCY = "consistency_vector"

# ---------------------------------------------------------------------------
# Feature Order — Strictly Preserved in this Exact Sequence (5 features)
# ---------------------------------------------------------------------------
CONSISTENCY_FEATURE_ORDER = [
    "qr_readable",
    "text_qr_match_score",
    "checksum_valid",
    "format_valid",
    "missing_field_count",
]

# Imputation value for text_qr_match_score when None or unparseable.
# When a QR is unreadable or text missing, match score cannot be computed.
# We impute 0.0 (no match) while preserving the exact 5-feature dimension.
DEFAULT_IMPUTE_TEXT_QR_MATCH_SCORE = 0.0

# ---------------------------------------------------------------------------
# Primary Binary Target Mapping:
# 0 = genuine
# 1 = forged
# ---------------------------------------------------------------------------
FINAL_LABEL_MAP = {
    "genuine": 0,
    "forged": 1,
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
}

# ---------------------------------------------------------------------------
# Strict Leakage Guard:
# Fields that must NEVER be used as model input features.
# ---------------------------------------------------------------------------
FORBIDDEN_FEATURE_FIELDS = {
    "id",
    "record_id",
    "filename",
    "path",
    "image_path",
    "label_path",
    "tamper_type",
    "final_label",
    "cnn_label",
    "ground_truth_fields",
    "expected_consistency_vector",
    "splice_region",
    "splice_variant",
    "splice_mask_seed",
    "splice_mask_parameters",
    "splice_mask_sha256",
    "fine_grained_edit_metadata",
}

# ---------------------------------------------------------------------------
# Verified Dataset Constants (Audited from M1 Ground Truth)
# ---------------------------------------------------------------------------
EXPECTED_TOTAL_RECORDS = 1000
EXPECTED_VARIANTS_PER_RECORD = 11
EXPECTED_TOTAL_IMAGES = 11000
EXPECTED_TRAIN_IMAGES = 7700
EXPECTED_VAL_IMAGES = 1650
EXPECTED_TEST_IMAGES = 1650
EXPECTED_TRAIN_IDENTITIES = 700
EXPECTED_VAL_IDENTITIES = 150
EXPECTED_TEST_IDENTITIES = 150
EXPECTED_NUM_FEATURES = 5
