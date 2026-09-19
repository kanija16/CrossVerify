"""
verification.py
----------------
Read-only verification tooling to validate computed consistency vectors
against M1's precomputed consistency_vector stored in label JSONs.

This is diagnostic tooling only — it never writes to or modifies M1 files,
and never feeds ground-truth label data into any model.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .consistency import compute_consistency_vector
from .config import DATASET_ROOT


def load_label_json(label_path: str) -> Dict[str, Any]:
    path = Path(label_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Label JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def verify_against_m1(label_path: str, document_family: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads one M1 label JSON, recomputes the consistency vector from its
    ocr_extracted_fields and qr_decoded_fields, and compares it field by
    field against the stored consistency_vector.

    Returns comparison dict with per-feature match status and all_match boolean.
    """
    data = load_label_json(label_path)

    stored_vector = data.get("consistency_vector")
    if stored_vector is None:
        raise ValueError(f"{label_path} has no consistency_vector to compare against")

    family = document_family or data.get("document_family")
    if family is None:
        raise ValueError(
            "document_family not found in label JSON and none was passed in."
        )

    ocr_fields = data.get("ocr_extracted_fields", {})
    qr_fields = data.get("qr_decoded_fields", {})

    computed = compute_consistency_vector(family, ocr_fields, qr_fields)

    comparison: Dict[str, Any] = {}
    for key in ("qr_readable", "checksum_valid", "format_valid", "missing_field_count"):
        stored_val = stored_vector.get(key)
        computed_val = computed[key]
        comparison[key] = {
            "stored": stored_val,
            "computed": computed_val,
            "match": stored_val == computed_val,
        }

    stored_score = stored_vector.get("text_qr_match_score")
    computed_score = computed["text_qr_match_score"]
    if stored_score is None or computed_score is None:
        score_match = stored_score == computed_score
        score_diff = 0.0 if score_match else 1.0
    else:
        score_diff = abs(float(stored_score) - float(computed_score))
        score_match = score_diff < 1e-4

    comparison["text_qr_match_score"] = {
        "stored": stored_score,
        "computed": computed_score,
        "match": score_match,
        "diff": round(score_diff, 4),
    }

    comparison["all_match"] = all(
        entry["match"] for k, entry in comparison.items() if k != "all_match"
    )
    return comparison


def verify_dataset(
    labels_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    score_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Runs verification across all (or max_samples) label JSON files in labels_dir.
    Computes per-feature match accuracy, overall match accuracy, and score differences.
    """
    if labels_dir is None:
        labels_dir = str(DATASET_ROOT / "labels")

    dir_path = Path(labels_dir).expanduser()
    if not dir_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {dir_path}")

    label_files = sorted(dir_path.glob("*.json"))
    if max_samples is not None:
        label_files = label_files[:max_samples]

    total = len(label_files)
    if total == 0:
        return {"total": 0, "status": "no files found"}

    feature_matches = {
        "qr_readable": 0,
        "text_qr_match_score": 0,
        "checksum_valid": 0,
        "format_valid": 0,
        "missing_field_count": 0,
    }
    mismatch_samples: Dict[str, List[Dict[str, Any]]] = {
        "qr_readable": [],
        "text_qr_match_score": [],
        "checksum_valid": [],
        "format_valid": [],
        "missing_field_count": [],
    }

    all_exact_match = 0
    all_within_tolerance = 0
    score_diffs: List[float] = []

    for path in label_files:
        result = verify_against_m1(str(path))
        if result["all_match"]:
            all_exact_match += 1

        score_diff = result["text_qr_match_score"].get("diff", 0.0)
        score_diffs.append(score_diff)

        # Check tolerance match: all discrete features match and score is within tolerance
        discrete_match = (
            result["qr_readable"]["match"]
            and result["checksum_valid"]["match"]
            and result["format_valid"]["match"]
            and result["missing_field_count"]["match"]
        )
        if discrete_match and (score_diff <= score_tolerance):
            all_within_tolerance += 1

        for feat in feature_matches:
            if result[feat]["match"]:
                feature_matches[feat] += 1
            else:
                if len(mismatch_samples[feat]) < 10:
                    mismatch_samples[feat].append({
                        "file": path.name,
                        "stored": result[feat]["stored"],
                        "computed": result[feat]["computed"],
                    })

    summary = {
        "total_files": total,
        "exact_5_vector_matches": all_exact_match,
        "exact_5_vector_accuracy": round(all_exact_match / total * 100, 2),
        "total_mismatching_records": total - all_exact_match,
        "within_tolerance_matches": all_within_tolerance,
        "within_tolerance_accuracy": round(all_within_tolerance / total * 100, 2),
        "score_tolerance_threshold": score_tolerance,
        "per_feature_accuracy": {
            feat: {
                "matches": count,
                "mismatches": total - count,
                "percentage": round(count / total * 100, 2),
            }
            for feat, count in feature_matches.items()
        },
        "text_qr_score_stats": {
            "mean_diff": round(sum(score_diffs) / len(score_diffs), 5),
            "max_diff": round(max(score_diffs), 4),
            "within_0_001": sum(1 for d in score_diffs if d <= 0.001),
            "within_0_01": sum(1 for d in score_diffs if d <= 0.01),
            "within_0_05": sum(1 for d in score_diffs if d <= 0.05),
            "within_0_05_pct": round(
                sum(1 for d in score_diffs if d <= 0.05) / total * 100, 2
            ),
        },
        "mismatch_samples": mismatch_samples,
    }
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Running verification on full dataset...")
        summary = verify_dataset()
        print(json.dumps(summary, indent=2))
    elif sys.argv[1] == "--batch":
        max_s = int(sys.argv[2]) if len(sys.argv) > 2 else None
        summary = verify_dataset(max_samples=max_s)
        print(json.dumps(summary, indent=2))
    else:
        label_json_path = sys.argv[1]
        fam = sys.argv[2] if len(sys.argv) > 2 else None
        result = verify_against_m1(label_json_path, fam)
        print(json.dumps(result, indent=2))
