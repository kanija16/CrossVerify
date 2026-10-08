"""
Identity-level (cluster) bootstrap 95% confidence intervals for the test-set results.

The 1,650 test documents are 150 identities x 11 variants, so documents are not
independent. This script resamples the 150 identities with replacement and keeps all
11 documents of each drawn identity, which gives intervals that account for the
within-identity correlation. Everything else matches src/evaluation/bootstrap_ci.py
(1,000 resamples, numpy RandomState(42), percentile [2.5, 97.5], same targets,
thresholds and metrics), so the two outputs can be compared directly.

Usage:
    python -m src.evaluation.bootstrap_ci_identity
Writes outputs/fusion/test_bootstrap_ci_identity.json
"""
import json

import numpy as np
import pandas as pd

from src.evaluation.bootstrap_ci import BRANCHES, METRICS, N_BOOT, SEED, TEST_PREDS, PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "fusion" / "test_bootstrap_ci_identity.json"


def bootstrap_by_identity(df, target, score, threshold):
    y_all = (df[target] == "forged").astype(int).values
    s_all = df[score].values
    p_all = (s_all >= threshold).astype(int)
    ids = df["record_id"].unique()
    rows_of = {rid: np.flatnonzero(df["record_id"].values == rid) for rid in ids}
    rng = np.random.RandomState(SEED)
    draws = {m: [] for m in METRICS}
    for _ in range(N_BOOT):
        pick = rng.randint(0, len(ids), len(ids))
        i = np.concatenate([rows_of[ids[k]] for k in pick])
        if len(np.unique(y_all[i])) < 2:
            continue
        for name, fn in METRICS.items():
            draws[name].append(fn(y_all[i], s_all[i], p_all[i]))
    return {
        name: {
            "point_estimate": round(float(fn(y_all, s_all, p_all)), 5),
            "ci_95_lower": round(float(np.percentile(draws[name], 2.5)), 4),
            "ci_95_upper": round(float(np.percentile(draws[name], 97.5)), 4),
        }
        for name, fn in METRICS.items()
    }


def main() -> None:
    df = pd.read_csv(TEST_PREDS)
    assert len(df) == 1650 and df["record_id"].nunique() == 150
    assert (df.groupby("record_id").size() == 11).all()
    results = {"protocol": f"{N_BOOT} identity-level resamples (150 identities x 11 documents), "
                           f"RandomState({SEED}), percentile 95% CI"}
    for branch, (target, score, threshold) in BRANCHES.items():
        results[branch] = {"target": target, "threshold": threshold,
                           **bootstrap_by_identity(df, target, score, threshold)}
        r = results[branch]
        print(f"{branch:6s} " + "  ".join(
            f"{m} {r[m]['point_estimate']:.4f} [{r[m]['ci_95_lower']}, {r[m]['ci_95_upper']}]" for m in METRICS))
    # CNN branch on the VALIDATION split (paper, Section 7.1: validation vs. test ROC-AUC)
    val = pd.read_csv(PROJECT_ROOT / "models" / "cnn" / "val_predictions.csv")
    assert len(val) == 1650 and val["record_id"].nunique() == 150
    r = bootstrap_by_identity(val, "cnn_label", "cnn_probability", 0.50)
    results["cnn_validation"] = {"target": "cnn_label", "threshold": 0.50, "roc_auc": r["roc_auc"]}
    print(f"cnn validation ROC-AUC {r['roc_auc']['point_estimate']:.4f} "
          f"[{r['roc_auc']['ci_95_lower']}, {r['roc_auc']['ci_95_upper']}]")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"saved {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
