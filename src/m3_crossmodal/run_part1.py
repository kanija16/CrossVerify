"""
run_part1.py
------------
Main executable entry point for M3 Phase 1 — Part 1.

Execution Workflow:
1. Load train, val, and test splits from M1 Desktop CSVs.
2. Resolve label JSON paths and parse M3 consistency blocks.
3. Execute comprehensive sanity checks across all splits.
4. Extract 5D consistency feature matrices: X_train, X_val, X_test.
5. Train Logistic Regression baseline on train split.
6. Evaluate Learned Baseline on validation and test sets.
7. Evaluate Rule-Based Baseline on validation and test sets.
8. Persist results to outputs/baseline_metrics.json.

Usage:
    python -m src.m3_crossmodal.run_part1
    or from inside src:
    python -m m3_crossmodal.run_part1
"""

import sys
import json
from pathlib import Path

# Add src to path if needed when run as script
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from m3_crossmodal import config, data_interface, features, sanity_checks, baseline
except ImportError:
    from src.m3_crossmodal import config, data_interface, features, sanity_checks, baseline


def format_metrics(metrics: dict) -> str:
    parts = []
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if k in metrics:
            val = metrics[k]
            parts.append(f"{k}: {val:.4f}" if isinstance(val, float) and not (val != val) else f"{k}: N/A")
    return " | ".join(parts)


def main():
    print("=" * 70)
    print("CROSSVERIFY — MEMBER 3 (M3) PHASE 1 — PART 1 EXECUTION")
    print("=" * 70)
    print(f"Dataset Root   : {config.DATASET_ROOT}")
    print(f"Train CSV      : {config.TRAIN_CSV}")
    print(f"Val CSV        : {config.VAL_CSV}")
    print(f"Test CSV       : {config.TEST_CSV}")
    print(f"Feature Vector : {config.CONSISTENCY_FEATURE_ORDER} (dim={config.EXPECTED_NUM_FEATURES})")
    print()

    # 1. Load Data
    print("[1/5] Loading M1 split CSVs and label JSONs...")
    records_by_split = data_interface.load_all_splits()
    print(f"      Loaded {len(records_by_split['train'])} train, "
          f"{len(records_by_split['val'])} val, {len(records_by_split['test'])} test records.")

    # 2. Build Feature Matrices
    print("\n[2/5] Building precomputed consistency feature matrices...")
    X_train, y_train, groups_train = features.build_baseline_matrix(records_by_split["train"])
    X_val, y_val, groups_val = features.build_baseline_matrix(records_by_split["val"])
    X_test, y_test, groups_test = features.build_baseline_matrix(records_by_split["test"])

    matrices = {
        "train": (X_train, y_train, groups_train),
        "val": (X_val, y_val, groups_val),
        "test": (X_test, y_test, groups_test),
    }

    print(f"      X_train shape : {X_train.shape} | y_train pos/neg: {sum(y_train)}/{len(y_train) - sum(y_train)}")
    print(f"      X_val shape   : {X_val.shape} | y_val pos/neg: {sum(y_val)}/{len(y_val) - sum(y_val)}")
    print(f"      X_test shape  : {X_test.shape} | y_test pos/neg: {sum(y_test)}/{len(y_test) - sum(y_test)}")

    # 3. Sanity Checks
    print("\n[3/5] Executing sanity checks suite...")
    all_passed = sanity_checks.run_all(records_by_split, matrices=matrices)
    if not all_passed:
        print("WARNING: Some sanity checks reported failures. Review diagnostic output above.")

    # 4. Train & Evaluate Learned Baseline
    print("\n[4/5] Training & Evaluating Learned Baseline (Logistic Regression)...")
    lr_model = baseline.train_baseline(X_train, y_train, random_state=42)
    learned_val_metrics = baseline.evaluate(lr_model, X_val, y_val)
    learned_test_metrics = baseline.evaluate(lr_model, X_test, y_test)

    print(f"      Validation -> {format_metrics(learned_val_metrics)}")
    print(f"      Test       -> {format_metrics(learned_test_metrics)}")

    # 5. Evaluate Rule-Based Baseline
    print("\n[5/5] Evaluating Rule-Based Baseline (Cryptographic / Format Floor)...")
    y_val_rule = baseline.rule_based_baseline(X_val, features.FEATURE_NAMES)
    y_test_rule = baseline.rule_based_baseline(X_test, features.FEATURE_NAMES)
    rule_val_metrics = baseline.evaluate_predictions(y_val, y_val_rule)
    rule_test_metrics = baseline.evaluate_predictions(y_test, y_test_rule)

    print(f"      Validation -> {format_metrics(rule_val_metrics)}")
    print(f"      Test       -> {format_metrics(rule_test_metrics)}")

    # Persist Results
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "baseline_metrics.json"

    results_payload = {
        "dataset_root": str(config.DATASET_ROOT),
        "feature_order": config.CONSISTENCY_FEATURE_ORDER,
        "sample_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "learned_baseline_logistic_regression": {
            "validation": learned_val_metrics,
            "test": learned_test_metrics,
            "coefficients": {
                name: float(coef)
                for name, coef in zip(features.FEATURE_NAMES, lr_model.coef_[0])
            },
            "intercept": float(lr_model.intercept_[0]),
        },
        "rule_based_baseline": {
            "validation": rule_val_metrics,
            "test": rule_test_metrics,
            "rule": "forged if checksum_valid == 0 or format_valid == 0 or qr_readable == 0",
        },
        "all_sanity_checks_passed": bool(all_passed),
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nResults successfully saved to: {results_file}")
    print("=" * 70)
    print("PART 1 COMPLETE: Data Interface & Baselines Operational.")
    print("=" * 70)


if __name__ == "__main__":
    main()
