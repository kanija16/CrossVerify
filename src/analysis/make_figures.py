"""
Paper figures, computed directly from the released test-set prediction files.

Nothing is typed in by hand: every curve, bar and histogram below is computed from
  outputs/fusion/corrected_fusion_test_predictions.csv   (CNN branch, M3, fusion)
  outputs/baselines/cnn_e2e_ep30/test_predictions.csv     (end-to-end CNN, headline run)
The script also prints the numbers behind each figure so they can be compared with the
paper's tables.

Run from the repository root:   python -m src.analysis.make_figures
Writes:   outputs/figures/fig_roc.pdf, fig_mechanism_recall.pdf, fig_score_distribution.pdf
          (+ .png previews)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
FUSION_CSV = ROOT / "outputs/fusion/corrected_fusion_test_predictions.csv"
E2E_CSV = ROOT / "outputs/baselines/cnn_e2e_ep30/test_predictions.csv"
OUT = ROOT / "outputs/figures"

# Operating thresholds used in the paper (all selected on the validation split)
TAU_CNN, TAU_M3, TAU_F = 0.5, 0.25, 0.82

# Fixed colour per model (validated categorical palette) + a second encoding
# (line style / hatch) so the figures also work in greyscale.
STYLE = {
    "CNN branch (M2)":        dict(color="#2a78d6", ls="-",  hatch=""),
    "Cross-modal branch (M3)": dict(color="#eb6834", ls="--", hatch="//"),
    "Fusion (CrossVerify)":   dict(color="#1baf7a", ls="-",  hatch=""),
    "End-to-end CNN":         dict(color="#eda100", ls=":",  hatch=".."),
}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

MECH_ORDER = [
    ("checksum_invalid", "Checksum invalid"),
    ("format_invalid", "Format invalid"),
    ("field_missing", "Field missing"),
    ("fine_grained_edit", "Fine-grained edit"),
    ("qr_only_mismatch", "QR-only mismatch"),
    ("text_qr_mismatch", "Text–QR mismatch"),
    ("visual_splice", "Visual splice"),
    ("coordinated_full_forgery", "Coordinated full forgery"),
]


def setup_matplotlib():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight",
    })


def load():
    fus = pd.read_csv(FUSION_CSV)
    e2e = pd.read_csv(E2E_CSV)[["id", "e2e_probability", "e2e_prediction"]]
    d = fus.merge(e2e, on="id", validate="one_to_one")
    assert len(d) == 1650, len(d)
    d["y"] = (d.final_label == "forged").astype(int)
    d["flag_cnn"] = (d.cnn_probability >= TAU_CNN).astype(int)
    d["flag_m3"] = (d.m3_probability >= TAU_M3).astype(int)
    d["flag_fusion"] = d.final_prediction.astype(int)
    d["flag_e2e"] = d.e2e_prediction.astype(int)
    return d


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def fig_roc(d):
    scores = {"CNN branch (M2)": "cnn_probability", "Cross-modal branch (M3)": "m3_probability",
              "End-to-end CNN": "e2e_probability", "Fusion (CrossVerify)": "fusion_probability"}
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=(0, (2, 2)), label="Chance (0.500)")
    print("\n[ROC] test ROC-AUC against final_label")
    for name, col in scores.items():
        fpr, tpr, _ = roc_curve(d.y, d[col])
        auc = roc_auc_score(d.y, d[col])
        print(f"  {name:26s} {auc:.5f}")
        st = STYLE[name]
        ax.plot(fpr, tpr, color=st["color"], ls=st["ls"], lw=1.8, label=f"{name} ({auc:.3f})")
    # fusion operating point at tau_F
    g = d[d.y == 0]
    f = d[d.y == 1]
    op_fpr, op_tpr = g.flag_fusion.mean(), f.flag_fusion.mean()
    print(f"  fusion operating point (tau_F={TAU_F}): FPR {op_fpr:.4f}  TPR {op_tpr:.4f}")
    ax.plot(op_fpr, op_tpr, "o", ms=6, mfc="white", mec=STYLE["Fusion (CrossVerify)"]["color"], mew=1.6, zorder=5)
    ax.annotate(f"fusion at $\\tau_F$={TAU_F}", (op_fpr, op_tpr), xytext=(op_fpr + 0.08, op_tpr - 0.12),
                fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate (genuine flagged)")
    ax.set_ylabel("True-positive rate (forgeries flagged)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor=GRID, title="Test ROC-AUC",
              title_fontsize=7)
    save(fig, "fig_roc")


def fig_mechanism_recall(d):
    models = [("CNN branch (M2)", "flag_cnn"), ("Cross-modal branch (M3)", "flag_m3"),
              ("End-to-end CNN", "flag_e2e"), ("Fusion (CrossVerify)", "flag_fusion")]
    rows = MECH_ORDER + [("genuine", "Genuine (false-positive rate)")]
    vals = {m: [d[d.tamper_type == t][c].mean() * 100 for t, _ in rows] for m, c in models}
    print("\n[Mechanism recall, %] (genuine row = false-positive rate)")
    print("  " + f"{'':30s}" + "".join(f"{m.split(' (')[0][:14]:>16s}" for m, _ in models))
    for i, (_, lab) in enumerate(rows):
        print(f"  {lab:30s}" + "".join(f"{vals[m][i]:16.2f}" for m, _ in models))

    n, k = len(rows), len(models)
    h = 0.19
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = np.arange(n)[::-1]
    for j, (m, _) in enumerate(models):
        st = STYLE[m]
        ax.barh(y + (k / 2 - j - 0.5) * h, vals[m], height=h * 0.9, color=st["color"],
                hatch=st["hatch"], edgecolor="white", linewidth=0.4, label=m)
    ax.axhline(y[-1] + 0.5, color=MUTED, lw=0.6)          # separates genuine row
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Documents flagged as forged (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.09), ncol=4, frameon=False)
    save(fig, "fig_mechanism_recall")


def fig_score_distribution(d):
    groups = [("genuine", "Genuine", "#666666", "-"),
              ("coordinated_full_forgery", "Coordinated full forgery", "#eb6834", "--"),
              ("other", "All other forgeries", "#2a78d6", ":")]
    lo = 0.45   # p_fusion is never below sigmoid(-1.469) ~ 0.19; all test scores are > 0.5
    assert d.fusion_probability.min() > lo, d.fusion_probability.min()
    bins = np.linspace(lo, 1, 23)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    print(f"\n[Fusion score distribution] share of documents with p_fusion >= {TAU_F}")
    for key, lab, colr, ls in groups:
        if key == "other":
            s = d[(d.y == 1) & (d.tamper_type != "coordinated_full_forgery")].fusion_probability
        else:
            s = d[d.tamper_type == key].fusion_probability
        w = np.ones(len(s)) * 100 / len(s)
        ax.hist(s, bins=bins, weights=w, histtype="step", color=colr, ls=ls, lw=1.6,
                label=f"{lab} (n={len(s)})")
        print(f"  {lab:28s} n={len(s):5d}  flagged {np.mean(s >= TAU_F) * 100:6.2f}%")
    ax.axvline(TAU_F, color=INK, lw=0.8)
    ax.text(TAU_F + 0.008, ax.get_ylim()[1] * 0.55, f"$\\tau_F$ = {TAU_F}", ha="left", va="center", fontsize=7)
    ax.set_xlabel("Fusion forgery probability $p_{\\mathrm{fusion}}$")
    ax.set_ylabel("Documents in group (%)")
    ax.set_xlim(lo, 1)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_score_distribution")


def main():
    setup_matplotlib()
    d = load()
    fig_roc(d)
    fig_mechanism_recall(d)
    fig_score_distribution(d)
    print(f"\nFigures written to {OUT}")


if __name__ == "__main__":
    main()
