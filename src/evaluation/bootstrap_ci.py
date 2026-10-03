"""
Bootstrap 95% confidence intervals for the test-set results reported in the paper.

Protocol (paper, Section VI-D): 1,000 resamples of the 1,650 test documents with
replacement, numpy RandomState(42), percentile intervals [2.5, 97.5]. Each branch is
scored against its own target at its frozen threshold:

    CNN    target cnn_label   (visual splice)   threshold 0.50
    M3     target final_label                   threshold 0.25
    Fusion target final_label                   threshold 0.82

Usage:
    python -m src.evaluation.bootstrap_ci
Writes outputs/fusion/test_bootstrap_ci.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_PREDS = PROJECT_ROOT / "outputs" / "fusion" / "corrected_fusion_test_predictions.csv"
OUT = PROJECT_ROOT / "outputs" / "fusion" / "test_bootstrap_ci.json"
N_BOOT, SEED = 1000, 42

BRANCHES = {
    "cnn": ("cnn_label", "cnn_probability", 0.50),
    "m3": ("final_label", "m3_probability", 0.25),
    "fusion": ("final_label", "fusion_probability", 0.82),
}
METRICS = {
    "roc_auc": lambda y, s, p: roc_auc_score(y, s),
    "pr_auc": lambda y, s, p: average_precision_score(y, s),
    "accuracy": lambda y, s, p: accuracy_score(y, p),
    "precision": lambda y, s, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, s, p: recall_score(y, p, zero_division=0),
    "f1": lambda y, s, p: f1_score(y, p, zero_division=0),
}


def bootstrap(y, s, threshold):
    p = (s >= threshold).astype(int)
    rng = np.random.RandomState(SEED)
    draws = {m: [] for m in METRICS}
    for _ in range(N_BOOT):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        for name, fn in METRICS.items():
            draws[name].append(fn(y[i], s[i], p[i]))
    return {
        name: {
            "point_estimate": round(float(fn(y, s, p)), 5),
            "ci_95_lower": round(float(np.percentile(draws[name], 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(draws[name], 97.5)), 4),
        }
        for name, fn in METRICS.items()
    }


def main() -> None:
    df = pd.read_csv(TEST_PREDS)
    assert len(df) == 1650
    results = {"protocol": f"{N_BOOT} document-level resamples, RandomState({SEED}), percentile 95% CI"}
    for branch, (target, score, threshold) in BRANCHES.items():
        y = (df[target] == "forged").astype(int).values
        results[branch] = {"target": target, "threshold": threshold,
                           **bootstrap(y, df[score].values, threshold)}
        r = results[branch]
        print(f"{branch:6s} ROC-AUC {r['roc_auc']['point_estimate']:.4f} "
              f"[{r['roc_auc']['ci_95_lower']}, {r['roc_auc']['ci_95_upper']}]  "
              f"F1 {r['f1']['point_estimate']:.4f} [{r['f1']['ci_95_lower']}, {r['f1']['ci_95_upper']}]")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"saved {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
