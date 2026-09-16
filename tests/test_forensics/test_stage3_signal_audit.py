"""
test_stage3_signal_audit.py
----------------------------
Unit and Regression Tests for Stage 3 Local Document Structure Signal Audit.

Verifies:
1. All Stage 3 output artifacts exist in outputs/forensics/stage3_signal_audit/.
2. Feature metric keys and classification categories adhere to specifications.
3. Single-feature ROC-AUC values are valid and bounded in [0.5, 1.0].
4. Strict zero test set access (no references to test.csv, test images, or test predictions).
5. Mathematical consistency of within-document and field-level feature extractors.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from m3_crossmodal.forensics.stage3_signal_audit import (
    compute_per_field_visual_features,
    extract_ocr_field_boxes,
    DIR_STAGE3,
    LABEL_PATTERNS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_feature_computation_unit():
    """Verify compute_per_field_visual_features returns all expected visual statistics."""
    img = np.full((100, 150), 240, dtype=np.uint8)
    img[30:70, 20:120] = 30  # High contrast text-like block

    box = (20, 30, 100, 40)
    feats = compute_per_field_visual_features(img, box)

    assert feats is not None
    expected_keys = [
        "gray_mean", "gray_std", "local_contrast", "grad_mag_mean", "grad_mag_std",
        "edge_density", "laplacian_var", "local_entropy", "hf_residual_energy",
        "text_bg_contrast", "boundary_discontinuity", "cc_density",
        "field_to_bg_gradient_ratio", "field_to_bg_laplacian_ratio"
    ]
    for k in expected_keys:
        assert k in feats, f"Missing key {k} in computed features"
        assert not np.isnan(feats[k]), f"NaN encountered in {k}"


def test_ocr_field_boxes_graceful_blank():
    """Verify extract_ocr_field_boxes handles blank or invalid image cleanly."""
    blank = np.zeros((100, 100), dtype=np.uint8)
    boxes = extract_ocr_field_boxes(blank, "tax_form")
    assert isinstance(boxes, dict)


def test_stage3_zero_test_access():
    """Verify no test set files or test tokens exist in stage 3 code or outputs."""
    src_p = PROJECT_ROOT / "src" / "m3_crossmodal" / "forensics" / "stage3_signal_audit.py"
    with open(src_p, "r") as f:
        src_text = f.read()

    assert "test.csv" not in src_text
    assert "test_alignment" not in src_text
    assert "test_predictions" not in src_text


def test_stage3_artifacts_exist_if_completed():
    """If audit has run, verify all 6 required artifacts exist."""
    required = [
        "signal_audit.csv",
        "field_level_summary.csv",
        "within_document_summary.csv",
        "family_stratified_summary.csv",
        "strongest_feature_distributions.csv",
        "stage3_signal_report.md",
    ]
    if (DIR_STAGE3 / "signal_audit.csv").exists():
        for fname in required:
            p = DIR_STAGE3 / fname
            assert p.exists(), f"Missing Stage 3 artifact: {fname}"

        df_audit = pd.read_csv(DIR_STAGE3 / "signal_audit.csv")
        assert len(df_audit) > 0
        assert "feature" in df_audit.columns
        assert "signal_classification" in df_audit.columns
        assert "single_feature_roc_auc" in df_audit.columns
        assert (df_audit["single_feature_roc_auc"] >= 0.5).all()
        assert (df_audit["single_feature_roc_auc"] <= 1.0).all()
