"""
Unit tests for Fusion Step 1: M2 and M3 Prediction Alignment.
Verifies:
1. val_alignment.csv exists and contains exactly 1650 rows.
2. Required columns are present with zero missing/null values.
3. 150 unique record_ids with exactly 11 variants per record (sample identity isolation).
4. cnn_probability and m3_probability ranges are strictly within [0.0, 1.0].
5. Exactly 150 genuine and 1500 forged samples.
6. Alignment metrics JSON artifact exists and contains valid statistics, correlations, and disagreement counts.
7. Test split is NOT touched or referenced in the alignment artifacts.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIR_FUSION = PROJECT_ROOT / "outputs" / "fusion"


def test_val_alignment_file_exists():
    """Verify val_alignment.csv and metrics json exist."""
    csv_p = DIR_FUSION / "val_alignment.csv"
    json_p = DIR_FUSION / "val_alignment_metrics.json"
    assert csv_p.exists(), f"Missing {csv_p}"
    assert json_p.exists(), f"Missing {json_p}"


def test_val_alignment_shape_and_columns():
    """Verify val_alignment.csv has 1,650 rows and required columns."""
    csv_p = DIR_FUSION / "val_alignment.csv"
    df = pd.read_csv(csv_p)
    assert len(df) == 1650, f"Expected 1650 rows, got {len(df)}"

    required_cols = [
        "id",
        "record_id",
        "document_family",
        "tamper_type",
        "final_label",
        "target",
        "cnn_probability",
        "cnn_prediction",
        "m3_probability",
        "m3_calibrated_probability",
        "m3_prediction",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col}"

    # Zero nulls across all columns
    assert df.isnull().sum().sum() == 0, "Found null values in alignment table"


def test_val_alignment_record_and_label_integrity():
    """Verify 150 unique record_ids, exactly 11 variants per record, and 150 genuine / 1500 forged."""
    csv_p = DIR_FUSION / "val_alignment.csv"
    df = pd.read_csv(csv_p)

    # 150 unique record_ids
    assert df["record_id"].nunique() == 150
    counts_per_rec = df["record_id"].value_counts()
    assert (counts_per_rec == 11).all(), "Each record_id must have exactly 11 variants"

    # 1,650 unique sample IDs
    assert df["id"].nunique() == 1650

    # Class balance: 150 genuine, 1500 forged
    assert (df["final_label"] == "genuine").sum() == 150
    assert (df["final_label"] == "forged").sum() == 1500
    assert (df["target"] == 0).sum() == 150
    assert (df["target"] == 1).sum() == 1500


def test_val_alignment_probabilities_validity():
    """Verify probabilities are strictly bounded in [0.0, 1.0]."""
    csv_p = DIR_FUSION / "val_alignment.csv"
    df = pd.read_csv(csv_p)

    for col in ["cnn_probability", "m3_probability", "m3_calibrated_probability"]:
        assert (df[col] >= 0.0).all(), f"Values in {col} below 0.0"
        assert (df[col] <= 1.0).all(), f"Values in {col} above 1.0"

    for col in ["cnn_prediction", "m3_prediction"]:
        assert set(df[col].unique()).issubset({0, 1}), f"Non-binary values in {col}"


def test_val_alignment_metrics_artifact():
    """Verify val_alignment_metrics.json has all required fields."""
    json_p = DIR_FUSION / "val_alignment_metrics.json"
    with open(json_p) as f:
        metrics = json.load(f)

    assert metrics["total_samples"] == 1650
    assert metrics["genuine_count"] == 150
    assert metrics["forged_count"] == 1500
    assert metrics["unique_records"] == 150
    assert metrics["unique_ids"] == 1650

    # Disagreement matrix sums to 1650
    dmat = metrics["disagreement_matrix"]
    total_dmat = (
        dmat["both_correct"]
        + dmat["m2_correct_m3_wrong"]
        + dmat["m3_correct_m2_wrong"]
        + dmat["both_wrong"]
    )
    assert total_dmat == 1650

    # Check correlations exist
    assert "m2_vs_m3_raw" in metrics["correlations"]
    assert "pearson" in metrics["correlations"]["m2_vs_m3_raw"]
    assert "spearman" in metrics["correlations"]["m2_vs_m3_raw"]


def test_no_test_split_leakage():
    """Ensure no test split files or labels were referenced or created in outputs/fusion/."""
    fusion_files = list(DIR_FUSION.glob("*"))
    for p in fusion_files:
        if p.name in ("final_test_evaluation", "final_pretest_verification"):
            continue
        assert "test" not in p.name.lower(), f"Unexpected test file in fusion outputs: {p.name}"
