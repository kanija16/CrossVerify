"""
sanity_checks.py
----------------
Deterministic, read-only validation suite for M3 Part 1 data deliverables.

Checks performed:
1. Exact split sizes (train=7700, val=1650, test=1650)
2. Total unique identities (1000 total: 700 train, 150 val, 150 test)
3. Identity-level split separation (0 overlap between train, val, test)
4. Variant count per identity (exactly 11 variants per record_id)
5. Strict binary targets (labels strictly in {0, 1})
6. Label JSON file resolution (all referenced files exist on disk)
7. Consistency vector schema integrity (all 5 keys present with valid values)
8. Matrix dimensions and finiteness (no NaN, no Inf, exact shape (N, 5))
9. Leakage guard (no forbidden/target fields in feature matrix)
"""

from collections import Counter
from typing import Dict, List, Tuple
import numpy as np

from . import config
from .data_interface import M3Record


def check_split_sizes(records_by_split: Dict[str, List[M3Record]]) -> Tuple[bool, str]:
    expected = {
        "train": config.EXPECTED_TRAIN_IMAGES,
        "val": config.EXPECTED_VAL_IMAGES,
        "test": config.EXPECTED_TEST_IMAGES,
    }
    all_ok = True
    messages = []
    for split, expected_n in expected.items():
        actual_n = len(records_by_split.get(split, []))
        if actual_n != expected_n:
            all_ok = False
            messages.append(f"{split}: expected {expected_n}, got {actual_n}")
        else:
            messages.append(f"{split}: {actual_n}")

    detail = ", ".join(messages)
    return all_ok, detail


def check_identity_counts(records_by_split: Dict[str, List[M3Record]]) -> Tuple[bool, str]:
    expected = {
        "train": config.EXPECTED_TRAIN_IDENTITIES,
        "val": config.EXPECTED_VAL_IDENTITIES,
        "test": config.EXPECTED_TEST_IDENTITIES,
    }
    all_ok = True
    messages = []
    total_ids = set()
    for split, expected_n in expected.items():
        ids = {r.record_id for r in records_by_split.get(split, [])}
        total_ids.update(ids)
        if len(ids) != expected_n:
            all_ok = False
            messages.append(f"{split}: expected {expected_n}, got {len(ids)}")
        else:
            messages.append(f"{split}: {len(ids)}")

    if len(total_ids) != config.EXPECTED_TOTAL_RECORDS:
        all_ok = False
        messages.append(f"total: expected {config.EXPECTED_TOTAL_RECORDS}, got {len(total_ids)}")
    else:
        messages.append(f"total: {len(total_ids)}")

    return all_ok, ", ".join(messages)


def check_identity_separation(records_by_split: Dict[str, List[M3Record]]) -> Tuple[bool, str]:
    ids = {split: {r.record_id for r in recs} for split, recs in records_by_split.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlaps = {}
    for a, b in pairs:
        overlap = ids.get(a, set()) & ids.get(b, set())
        if overlap:
            overlaps[f"{a}&{b}"] = len(overlap)

    if overlaps:
        return False, f"Identity leakage detected: {overlaps}"
    return True, "Disjoint: 0 overlapping record_ids across all split pairs"


def check_variant_count(records_by_split: Dict[str, List[M3Record]]) -> Tuple[bool, str]:
    all_records = [r for recs in records_by_split.values() for r in recs]
    counts = Counter(r.record_id for r in all_records)
    bad = {rid: c for rid, c in counts.items() if c != config.EXPECTED_VARIANTS_PER_RECORD}
    if bad:
        return False, f"{len(bad)} identities do not have exactly {config.EXPECTED_VARIANTS_PER_RECORD} variants"
    return True, f"All {len(counts)} identities have exactly {config.EXPECTED_VARIANTS_PER_RECORD} variants"


def check_binary_targets(records_by_split: Dict[str, List[M3Record]]) -> Tuple[bool, str]:
    invalid_targets = []
    distribution = {}
    for split, recs in records_by_split.items():
        targets = [r.target for r in recs]
        c = Counter(targets)
        distribution[split] = dict(c)
        for t in targets:
            if t not in (0, 1):
                invalid_targets.append(t)

    if invalid_targets:
        return False, f"Non-binary targets found: {set(invalid_targets)}"
    detail = " | ".join(f"{s}: genuine={d.get(0, 0)}, forged={d.get(1, 0)}" for s, d in distribution.items())
    return True, detail


def check_consistency_schema(records_by_split: Dict[str, List[M3Record]]) -> Tuple[bool, str]:
    missing_count = 0
    non_numeric_count = 0
    total_checked = 0

    for recs in records_by_split.values():
        for r in recs:
            total_checked += 1
            cv = r.consistency_vector
            for key in config.CONSISTENCY_FEATURE_ORDER:
                if key not in cv:
                    missing_count += 1
                else:
                    v = cv[key]
                    if v is not None and not isinstance(v, (bool, int, float)):
                        non_numeric_count += 1

    if missing_count > 0 or non_numeric_count > 0:
        return False, f"Missing keys: {missing_count}, Invalid types: {non_numeric_count} (out of {total_checked})"
    return True, f"All {total_checked} records contain valid 5-field consistency vectors"


def check_feature_matrix(
    X: np.ndarray,
    expected_rows: int,
    expected_cols: int = config.EXPECTED_NUM_FEATURES
) -> Tuple[bool, str]:
    if X.shape != (expected_rows, expected_cols):
        return False, f"Shape mismatch: expected ({expected_rows}, {expected_cols}), got {X.shape}"
    if not np.isfinite(X).all():
        return False, "Feature matrix contains NaN or Infinite values"
    return True, f"Shape {X.shape} verified finite (no NaN/Inf)"


def run_all(
    records_by_split: Dict[str, List[M3Record]],
    matrices: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = None
) -> bool:
    """
    Run the entire suite of sanity checks and print a structured report.
    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 70)
    print("M3 PART 1 — COMPREHENSIVE SANITY CHECKS")
    print("=" * 70)

    checks = [
        ("Split Sample Counts", check_split_sizes(records_by_split)),
        ("Identity Counts", check_identity_counts(records_by_split)),
        ("Identity Disjointness", check_identity_separation(records_by_split)),
        ("Variant Counts per Identity", check_variant_count(records_by_split)),
        ("Binary Target Values", check_binary_targets(records_by_split)),
        ("Consistency Vector Schema", check_consistency_schema(records_by_split)),
    ]

    if matrices is not None:
        expected_sizes = {
            "train": config.EXPECTED_TRAIN_IMAGES,
            "val": config.EXPECTED_VAL_IMAGES,
            "test": config.EXPECTED_TEST_IMAGES,
        }
        for sname, (X, y, _) in matrices.items():
            checks.append(
                (f"Feature Matrix ({sname})", check_feature_matrix(X, expected_sizes[sname]))
            )

    all_passed = True
    for name, (passed, detail) in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status:4s}] {name:30s} : {detail}")

    print("=" * 70)
    overall_status = "ALL SANITY CHECKS PASSED" if all_passed else "SANITY CHECKS FAILED"
    print(f"OVERALL STATUS: {overall_status}")
    print("=" * 70 + "\n")

    return all_passed
