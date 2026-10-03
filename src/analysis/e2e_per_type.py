"""
Per-mechanism analysis of the end-to-end image-only CNN baseline (paper Section VIII).

For each run (outputs/baselines/cnn_e2e = 15-epoch budget, cnn_e2e_ep30 = 30-epoch budget,
headline) and each forgery mechanism, reports on the test split:
  - flag rate of the baseline and of the fusion model,
  - baseline ROC-AUC of that mechanism against genuine documents (overall and per family),
  - Fisher's exact test on flag rates and Mann-Whitney U on scores, mechanism vs. genuine.

Run from the repository root:  python -m src.analysis.e2e_per_type
Writes outputs/baselines/<run>/e2e_per_type_test.csv
"""
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
FUSION = ROOT / "outputs/fusion/corrected_fusion_test_predictions.csv"
RUNS = ["cnn_e2e_ep30", "cnn_e2e"]


def auc_vs_genuine(s, g):
    x = pd.concat([s, g])
    return roc_auc_score((x.tamper_type != "genuine").astype(int), x.e2e_probability)


def main():
    fus = pd.read_csv(FUSION)[["id", "final_prediction"]]
    for run in RUNS:
        path = ROOT / "outputs/baselines" / run / "test_predictions.csv"
        d = pd.read_csv(path).merge(fus, on="id")
        overall = roc_auc_score((d.final_label == "forged").astype(int), d.e2e_probability)
        g = d[d.tamper_type == "genuine"]
        rows = []
        for t, s in d.groupby("tamper_type"):
            r = dict(tamper_type=t, n=len(s),
                     e2e_flag_rate=round((s.e2e_prediction == 1).mean(), 4),
                     fusion_flag_rate=round((s.final_prediction == 1).mean(), 4))
            if t != "genuine":
                r["e2e_auc_vs_genuine"] = round(auc_vs_genuine(s, g), 3)
                table = [[(s.e2e_prediction == 1).sum(), (s.e2e_prediction == 0).sum()],
                         [(g.e2e_prediction == 1).sum(), (g.e2e_prediction == 0).sum()]]
                r["fisher_p_flag_vs_genuine"] = round(fisher_exact(table)[1], 3)
                r["mannwhitney_p_score_vs_genuine"] = round(
                    mannwhitneyu(s.e2e_probability, g.e2e_probability, alternative="two-sided").pvalue, 3)
                for fam in ["family_a", "family_b"]:
                    r["auc_" + fam] = round(auc_vs_genuine(s[s.document_family == fam],
                                                           g[g.document_family == fam]), 3)
            rows.append(r)
        out = pd.DataFrame(rows)
        out.to_csv(path.parent / "e2e_per_type_test.csv", index=False)
        print(f"\n== {run}: test ROC-AUC {overall:.4f}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
