"""
diagnostics.py
--------------
Diagnostic runner for checking the live OCR/QR pipeline against dataset images.
This is diagnostic reporting tooling only.

CRITICAL COMPLIANCE:
Uses COL_DOCUMENT_FAMILY = "document_family" as primary column name.
Operates on image pixels + explicit document_family.
Never leaks ground truth or tamper metadata into extraction.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .live_pipeline import run_live_crossmodal

COL_IMAGE_PATH = "image_path"
COL_LABEL_PATH = "label_path"
COL_DOCUMENT_FAMILY = "document_family"
COL_FAMILY_ALIAS = "family"


def _load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _resolve_path(raw_path: str, dataset_root: Optional[str]) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if dataset_root:
        return Path(dataset_root).expanduser() / raw_path
    return p


def _get_family(row: Dict[str, str]) -> Optional[str]:
    return row.get(COL_DOCUMENT_FAMILY) or row.get(COL_FAMILY_ALIAS)


def _sample_rows(rows: List[Dict[str, str]], sample_size: int, families) -> List[Dict[str, str]]:
    by_family = defaultdict(list)
    for row in rows:
        fam = _get_family(row)
        if fam in families:
            by_family[fam].append(row)

    sampled = []
    per_family_n = max(1, sample_size // max(1, len(families)))
    for fam in families:
        pool = by_family.get(fam, [])
        sampled.extend(random.sample(pool, min(per_family_n, len(pool))))
    return sampled


def _load_reference_label(label_path: Path) -> Optional[Dict[str, Any]]:
    """Read-only load of label JSON, used ONLY for post-hoc comparison."""
    if not label_path.exists():
        return None
    with open(label_path, "r") as f:
        return json.load(f)


def run_diagnostics(
    csv_path: str,
    dataset_root: Optional[str] = None,
    sample_size: int = 25,
    families=("family_a", "family_b"),
    seed: int = 42,
) -> Dict[str, Any]:
    random.seed(seed)
    rows = _load_csv_rows(csv_path)
    sample = _sample_rows(rows, sample_size, families)

    per_family_stats = {
        fam: {
            "n": 0,
            "ocr_text_obtained": 0,
            "ocr_structured_success": 0,
            "qr_success": 0,
            "both_success": 0,
            "field_present_counts": defaultdict(int),
        }
        for fam in families
    }
    ocr_failure_examples = []
    qr_failure_examples = []

    for row in sample:
        family = _get_family(row)
        image_path = _resolve_path(row[COL_IMAGE_PATH], dataset_root)

        live_result = run_live_crossmodal(str(image_path), family)
        ocr_fields = live_result["ocr_extracted_fields"]
        qr_readable = live_result["consistency_vector"]["qr_readable"]

        ocr_structured_ok = any(v not in (None, "") for v in ocr_fields.values())
        stats = per_family_stats[family]
        stats["n"] += 1
        stats["ocr_text_obtained"] += int(bool(ocr_fields))
        stats["ocr_structured_success"] += int(ocr_structured_ok)
        stats["qr_success"] += int(qr_readable)
        stats["both_success"] += int(ocr_structured_ok and qr_readable)

        for field_name, value in ocr_fields.items():
            if value not in (None, ""):
                stats["field_present_counts"][field_name] += 1

        if not ocr_structured_ok and len(ocr_failure_examples) < 5:
            ocr_failure_examples.append(str(image_path))
        if not qr_readable and len(qr_failure_examples) < 5:
            qr_failure_examples.append(str(image_path))

    report = {
        "sample_size": len(sample),
        "per_family": {
            fam: {
                "n": s["n"],
                "ocr_text_rate": (s["ocr_text_obtained"] / s["n"]) if s["n"] else None,
                "ocr_structured_rate": (s["ocr_structured_success"] / s["n"]) if s["n"] else None,
                "qr_success_rate": (s["qr_success"] / s["n"]) if s["n"] else None,
                "both_success_rate": (s["both_success"] / s["n"]) if s["n"] else None,
                "field_presence_rate": {
                    field: count / s["n"] for field, count in s["field_present_counts"].items()
                } if s["n"] else {},
            }
            for fam, s in per_family_stats.items()
        },
        "ocr_failure_examples": ocr_failure_examples,
        "qr_failure_examples": qr_failure_examples,
    }
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print("=" * 60)
    print(f"LIVE OCR/QR diagnostics — sample size: {report['sample_size']}")
    print("=" * 60)
    for family, stats in report["per_family"].items():
        print(f"\n[{family}] n={stats['n']}")
        print(f"  OCR text obtained rate     : {stats['ocr_text_rate']}")
        print(f"  OCR structured fields rate : {stats['ocr_structured_rate']}")
        print(f"  QR success rate            : {stats['qr_success_rate']}")
        print(f"  Both success rate          : {stats['both_success_rate']}")
        print(f"  Field presence rates       : {stats['field_presence_rate']}")
    print("\nOCR failure examples:")
    for path in report["ocr_failure_examples"]:
        print(f"  - {path}")
    print("\nQR failure examples:")
    for path in report["qr_failure_examples"]:
        print(f"  - {path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live OCR/QR diagnostics against real images.")
    parser.add_argument("--csv", required=True, help="Path to train.csv / val.csv / test.csv")
    parser.add_argument("--dataset-root", default=None, help="Root dir for relative image/label paths")
    parser.add_argument("--sample-size", type=int, default=25)
    args = parser.parse_args()

    result = run_diagnostics(args.csv, dataset_root=args.dataset_root, sample_size=args.sample_size)
    _print_report(result)
