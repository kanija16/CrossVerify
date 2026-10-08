"""
Control experiment: fuse the END-TO-END CNN (instead of the visual-splice CNN branch)
with the cross-modal branch M3, to test whether the two-label decomposition matters
once the cross-modal branch is present.

Out-of-fold end-to-end CNN predictions on the training split do not exist (the
end-to-end weights were not kept), so the paper's OOF protocol cannot be repeated for
this pair. Instead both pairs are fused under one identical protocol that uses only
saved predictions:

  * meta-learner: LogisticRegression(C=1.0, lbfgs, random_state=42), the same
    specification as the paper's fusion, fitted on the VALIDATION predictions;
  * threshold: selected on validation with the paper's rule (max F1 subject to
    specificity >= 0.90, grid 0.01..0.99);
  * the test split is scored once.

Pairs:  [visual-splice CNN branch (M2) + M3]  vs  [end-to-end CNN (30-epoch run) + M3]

Run from the repository root:  python -m src.analysis.e2e_m3_fusion
Writes outputs/baselines/cnn_e2e_ep30/e2e_m3_fusion_test.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "outputs/fusion/corrected_fusion_validation_predictions.csv"
TEST = ROOT / "outputs/fusion/corrected_fusion_test_predictions.csv"
E2E = ROOT / "outputs/baselines/cnn_e2e_ep30"
OUT = E2E / "e2e_m3_fusion_test.json"


def select_threshold(y, p, min_spec=0.90):
    """Max F1 subject to specificity >= min_spec (same rule as the fusion model)."""
    best = (-1.0, None)
    for th in np.round(np.linspace(0.01, 0.99, 99), 2):
        tn, fp, fn, tp = confusion_matrix(y, p >= th, labels=[0, 1]).ravel()
        if tn + fp and tn / (tn + fp) >= min_spec:
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best[0]:
                best = (f1, float(th))
    return best[1]


def load():
    v = pd.read_csv(VAL).rename(columns={"cnn_probability_val": "cnn", "m3_probability_val": "m3"})
    t = pd.read_csv(TEST).rename(columns={"cnn_probability": "cnn", "m3_probability": "m3"})
    v = v.merge(pd.read_csv(E2E / "val_predictions.csv")[["id", "e2e_probability"]], on="id", validate="one_to_one")
    t = t.merge(pd.read_csv(E2E / "test_predictions.csv")[["id", "e2e_probability"]], on="id", validate="one_to_one")
    for d in (v, t):
        assert len(d) == 1650
        d["y"] = (d.final_label == "forged").astype(int)
    return v.rename(columns={"e2e_probability": "e2e"}), t.rename(columns={"e2e_probability": "e2e"})


def run(v, t, cols):
    lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=42).fit(v[cols], v.y)
    pv, pt = lr.predict_proba(v[cols])[:, 1], lr.predict_proba(t[cols])[:, 1]
    th = select_threshold(v.y.values, pv)
    tn, fp, fn, tp = confusion_matrix(t.y, pt >= th, labels=[0, 1]).ravel()
    mech = {k: round(float((pt[t.tamper_type.values == k] >= th).mean()), 4) for k in sorted(t.tamper_type.unique())}
    return {
        "inputs": cols,
        "weights": [round(float(w), 4) for w in lr.coef_[0]], "intercept": round(float(lr.intercept_[0]), 4),
        "val_roc_auc": round(float(roc_auc_score(v.y, pv)), 5),
        "threshold": th,
        "test_roc_auc": round(float(roc_auc_score(t.y, pt)), 5),
        "test_pr_auc": round(float(average_precision_score(t.y, pt)), 5),
        "test_recall": round(tp / (tp + fn), 4), "test_specificity": round(tn / (tn + fp), 4),
        "test_f1": round(2 * tp / (2 * tp + fp + fn), 4),
        "test_flag_rate_by_mechanism": mech,
    }


def main():
    v, t = load()
    res = {"protocol": "LogisticRegression fitted on validation predictions; threshold selected on "
                       "validation (max F1, specificity >= 0.90); test scored once",
           "cnn_branch_plus_m3": run(v, t, ["cnn", "m3"]),
           "e2e_cnn_plus_m3": run(v, t, ["e2e", "m3"]),
           "m3_alone_test_roc_auc": round(float(roc_auc_score(t.y, t.m3)), 5)}
    OUT.write_text(json.dumps(res, indent=2))
    for k in ("cnn_branch_plus_m3", "e2e_cnn_plus_m3"):
        r = res[k]
        print(f"{k:20s} test ROC-AUC {r['test_roc_auc']:.4f}  PR-AUC {r['test_pr_auc']:.4f}  "
              f"th {r['threshold']}  recall {r['test_recall']}  spec {r['test_specificity']}  F1 {r['test_f1']}")
        print("    " + "  ".join(f"{m}={x:.3f}" for m, x in r["test_flag_rate_by_mechanism"].items()))
    print("M3 alone test ROC-AUC", res["m3_alone_test_roc_auc"])
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
