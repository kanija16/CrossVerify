"""
CrossVerify revision analyses (paper revision, 2026).
Run from anywhere:  python -m src.analysis.crossverify_audit
Needs: pandas, numpy, scikit-learn, scipy, statsmodels.
Uses only prediction/feature files already in the repo (no images, no retraining).
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import (roc_auc_score as AUC, average_precision_score as AP,
                             confusion_matrix, brier_score_loss)
from statsmodels.stats.contingency_tables import mcnemar
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent.parent)  # repo root

OOF = pd.read_csv("outputs/fusion/oof/corrected_fusion_oof.csv")
VAL = pd.read_csv("outputs/fusion/corrected_fusion_validation_predictions.csv")
VAL_ALIGN = pd.read_csv("outputs/fusion/val_alignment.csv")
TEST = pd.read_csv("outputs/fusion/corrected_fusion_test_predictions.csv")
NAIVE = pd.read_csv("outputs/fusion/final_test_evaluation/final_test_predictions.csv")
F_TR = pd.read_csv("outputs/features/train_features.csv")
F_VA = pd.read_csv("outputs/features/val_features.csv")
F_TE = pd.read_csv("outputs/features/test_features.csv")
assert (TEST.id == NAIVE.id).all() and (F_TE.id == TEST.id).all() and (F_TR.id == OOF.id).all()

y_oof = (OOF.final_label == "forged").astype(int).values
y_val = VAL.target.values
y_te = (TEST.final_label == "forged").astype(int).values
TAU_F, TAU_M3, TAU_CNN = 0.82, 0.25, 0.50


def cm(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return (f"TN {tn} FP {fp} FN {fn} TP {tp} | rec {tp/(tp+fn):.4f} spec {tn/(tn+fp):.4f} "
            f"prec {tp/(tp+fp):.4f} F1 {2*tp/(2*tp+fp+fn):.4f}")


def section(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------------------
section("1. M3 out-of-fold PR-AUC: why pooled (0.9808) > every fold")
p = OOF.m3_probability_oof.values
f = OOF.fold.values
for k in range(5):
    m = f == k
    print(f"fold {k}: ROC {AUC(y_oof[m], p[m]):.5f}  AP {AP(y_oof[m], p[m]):.5f}")
print(f"pooled raw:          AP {AP(y_oof, p):.5f}")
q = p.copy()
for k in range(5):
    m = f == k
    vals, c = np.unique(p[m], return_counts=True)
    q[m & (p == vals[c.argmax()])] = 0.0  # merge each fold's big tie block into one common value
print(f"pooled, ties merged: AP {AP(y_oof, q):.5f}  <- use this (or report mean of folds)")

# ---------------------------------------------------------------------------
section("2. OOF (fusion training inputs) vs deployment (val/test inputs) CNN scores")
pct = [10, 25, 50, 75, 90]
for t in ["genuine", "visual_splice", "coordinated_full_forgery"]:
    a = OOF.cnn_probability_oof[OOF.tamper_type == t]
    b = VAL_ALIGN.cnn_probability[VAL_ALIGN.tamper_type == t]
    c = TEST.cnn_probability[TEST.tamper_type == t]
    print(f"{t:25s} OOF {np.round(np.percentile(a, pct), 3)}  VAL {np.round(np.percentile(b, pct), 3)}"
          f"  TEST {np.round(np.percentile(c, pct), 3)}")

# ---------------------------------------------------------------------------
section("3. Headline test results")
print("OOF fusion  :", cm(y_te, TEST.final_prediction), f"| AUC {AUC(y_te, TEST.fusion_probability):.5f}")
print("Naive fusion:", cm(y_te, NAIVE.fusion_prediction), f"| AUC {AUC(y_te, NAIVE.fusion_probability):.5f}")
print("M3          :", cm(y_te, (TEST.m3_probability >= TAU_M3).astype(int)))

# ---------------------------------------------------------------------------
section("4. Ablation: McNemar OOF fusion vs naive fusion (paired, test)")
a = TEST.final_prediction.values == y_te
b = NAIVE.fusion_prediction.values == y_te
table = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
res = mcnemar(table, exact=True)
print("table [[both right, only OOF right],[only naive right, both wrong]] =", table)
print(f"exact McNemar p = {res.pvalue:.4f}")
d = TEST.final_prediction.values != NAIVE.fusion_prediction.values
print("documents with different decisions:", d.sum(), TEST.tamper_type[d].value_counts().to_dict())
print(f"Spearman correlation of the two fusion scores: {spearmanr(TEST.fusion_probability, NAIVE.fusion_probability).correlation:.4f}")

# ---------------------------------------------------------------------------
section("5. Coordinated full forgery vs genuine (is it above chance?)")
sub = TEST[TEST.tamper_type.isin(["genuine", "coordinated_full_forgery"])]
ys = (sub.tamper_type == "coordinated_full_forgery").astype(int)
for col, th in [("fusion_probability", TAU_F), ("m3_probability", TAU_M3), ("cnn_probability", TAU_CNN)]:
    k1 = int((sub[col][ys == 1] >= th).sum()); k0 = int((sub[col][ys == 0] >= th).sum())
    pval = fisher_exact([[k1, 150 - k1], [k0, 150 - k0]])[1]
    print(f"{col:20s} flagged coord {k1}/150 vs genuine {k0}/150 | Fisher p={pval:.3f} | AUC coord-vs-genuine {AUC(ys, sub[col]):.3f}")

# ---------------------------------------------------------------------------
section("6. Read failures (QR unreadable / OCR failed) drive M3 false positives")
def read_status(F):
    return np.where(F.qr_readable == 0, "QR unreadable",
           np.where(F.ocr_field_presence_count == 0, "OCR failed",
           np.where(F.ocr_field_presence_count < 5, "OCR partial", "read OK")))
st = pd.Series(read_status(F_TE))
flag_m3 = TEST.m3_probability >= TAU_M3
g = TEST.tamper_type == "genuine"
print(f"QR unreadable rate: train {(F_TR.qr_readable==0).mean():.3f}  val {(F_VA.qr_readable==0).mean():.3f}  test {(F_TE.qr_readable==0).mean():.3f}")
print(f"QR-unreadable docs flagged by M3 (test): {(flag_m3 & (st=='QR unreadable')).sum()}/{(st=='QR unreadable').sum()}")
print(f"M3 false positives on genuine: {(flag_m3 & g).sum()}, of which read failures: {(flag_m3 & g & (st!='read OK')).sum()}")
print("M3 score for QR-unreadable docs:", TEST.m3_probability[st == "QR unreadable"].round(4).value_counts().to_dict())

# ---------------------------------------------------------------------------
section("7. M3 Random Forest @0.25 == a 5-line rule")
def rule(F):
    return ((F.checksum_valid != 1) | (F.format_valid != 1) | (F.missing_field_count > 0) |
            (F.qr_readable != 1) | (F.text_qr_match_score.fillna(0) < 1)).astype(int).values
print(f"agreement test {np.mean(rule(F_TE) == flag_m3.astype(int).values):.4f}  "
      f"val {np.mean(rule(F_VA) == (VAL.m3_probability_val >= TAU_M3).astype(int).values):.4f}  "
      f"train-OOF {np.mean(rule(F_TR) == (OOF.m3_probability_oof >= TAU_M3).astype(int).values):.4f}")

# ---------------------------------------------------------------------------
section("8. Why fusion loses structural recall (needs pM3 >= ~0.45 when pCNN ~ 0)")
w_c, w_m, b0 = 3.536043, 6.615993, -1.469315
need = (np.log(TAU_F / (1 - TAU_F)) - b0) / w_m
print(f"min pM3 to flag at pCNN=0: {need:.3f}")
miss = TEST[(TEST.final_prediction == 0) & TEST.tamper_type.isin(
    ["text_qr_mismatch", "qr_only_mismatch", "field_missing", "fine_grained_edit"])]
print("pM3 values of missed structural forgeries:", miss.m3_probability.round(4).value_counts().to_dict())

# ---------------------------------------------------------------------------
section("9. Simple fusion baselines (threshold: max F1 s.t. val spec >= 0.90)")
def pick(y, s):
    best = (-1, None)
    for th in np.unique(np.round(np.r_[np.linspace(0.01, 0.99, 99), np.unique(s)], 6)):
        tn, fp, fn, tp = confusion_matrix(y, s >= th).ravel()
        if tn / (tn + fp) >= 0.90 and 2*tp/(2*tp+fp+fn) > best[0]:
            best = (2*tp/(2*tp+fp+fn), th)
    return best[1]
cv, mv = VAL.cnn_probability_val.values, VAL.m3_probability_val.values
ct, mt = TEST.cnn_probability.values, TEST.m3_probability.values
print(f"{'logistic regression (paper)':28s} AUC {AUC(y_te, TEST.fusion_probability):.4f} | {cm(y_te, TEST.final_prediction)}")
for name, sv, st_ in [("max(pCNN, pM3)", np.maximum(cv, mv), np.maximum(ct, mt)),
                      ("noisy-OR", 1-(1-cv)*(1-mv), 1-(1-ct)*(1-mt)),
                      ("mean", (cv+mv)/2, (ct+mt)/2)]:
    th = pick(y_val, sv)
    print(f"{name:28s} AUC {AUC(y_te, st_):.4f} | th {th:.3f} | {cm(y_te, st_ >= th)}")

# ---------------------------------------------------------------------------
section("10. Realistic prevalence and genuine-as-positive PR-AUC")
def prec_at(rec, spec, prev):
    return rec * prev / (rec * prev + (1 - spec) * (1 - prev))
for prev in [0.01, 0.05, 0.10]:
    print(f"forgery prevalence {prev:.0%}: fusion precision {prec_at(0.886, 0.880, prev):.3f}")
for name, s in [("fusion", TEST.fusion_probability), ("M3", TEST.m3_probability), ("CNN", TEST.cnn_probability)]:
    print(f"{name:6s} AP forged+ {AP(y_te, s):.4f} (chance {y_te.mean():.3f}) | AP genuine+ {AP(1-y_te, -s):.4f} (chance {1-y_te.mean():.3f})")

# ---------------------------------------------------------------------------
section("11. Brier: raw M3 is worse than a constant predictor")
print(f"M3 raw {brier_score_loss(y_te, TEST.m3_probability):.5f} | M3 calibrated {brier_score_loss(y_te, NAIVE.m3_calibrated_probability):.5f} | "
      f"fusion {brier_score_loss(y_te, TEST.fusion_probability):.5f} | constant base rate {brier_score_loss(y_te, np.full(len(y_te), y_te.mean())):.5f}")
