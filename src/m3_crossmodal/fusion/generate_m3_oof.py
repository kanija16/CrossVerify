"""
Generate M3 (cross-modal Random Forest) out-of-fold predictions on the training split.

Reproduces outputs/fusion/oof/train_oof_predictions.csv, the M3 input used to train the
leakage-controlled fusion meta-learner (train_corrected_fusion.py).

Protocol (matches the paper, Section V-D/V-E):
  * 5-fold GroupKFold assignments by record_id, read from outputs/fusion/train_oof_folds.csv
  * For each fold, a FRESH pipeline is fit on the other four folds only:
      SimpleImputer(strategy="median", add_indicator=True)
      RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4,
                             min_samples_leaf=1, max_features="sqrt",
                             class_weight="balanced", random_state=42)
    so imputation statistics are computed on the training folds only.
  * The held-out fold is predicted; the five held-out blocks cover all 7,700 rows.

Verification: the output matches the m3_probability_oof column of
outputs/fusion/oof/corrected_fusion_oof.csv exactly (max |diff| = 0 at 6 decimals).

Usage:
    python -m src.m3_crossmodal.fusion.generate_m3_oof
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAIN_FEATURES = PROJECT_ROOT / "outputs" / "features" / "train_features.csv"
TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "train.csv"
FOLDS = PROJECT_ROOT / "outputs" / "fusion" / "train_oof_folds.csv"
SCHEMA = PROJECT_ROOT / "outputs" / "final_m3_model_freeze" / "final_feature_schema.json"
OUT = PROJECT_ROOT / "outputs" / "fusion" / "oof" / "train_oof_predictions.csv"
REFERENCE = PROJECT_ROOT / "outputs" / "fusion" / "oof" / "corrected_fusion_oof.csv"


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=4, min_samples_leaf=1,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])


def main() -> None:
    feats = pd.read_csv(TRAIN_FEATURES)
    split = pd.read_csv(TRAIN_SPLIT)
    folds = pd.read_csv(FOLDS)
    feature_order = json.loads(SCHEMA.read_text())["feature_order"]
    assert len(feature_order) == 15

    # Strict row alignment across all inputs
    assert len(feats) == len(split) == len(folds) == 7700
    assert (feats["id"].values == split["id"].values).all()
    assert (folds["id"].values == split["id"].values).all()
    assert (folds["record_id"].values == split["record_id"].values).all()

    X = feats[feature_order]
    y = (split["final_label"] == "forged").astype(int).values
    fold = folds["oof_fold"].values
    oof = np.full(len(split), np.nan)

    for k in range(5):
        held = fold == k
        train_ids = set(split.loc[~held, "record_id"])
        held_ids = set(split.loc[held, "record_id"])
        assert train_ids.isdisjoint(held_ids), f"identity leak in fold {k}"
        model = make_pipeline().fit(X[~held], y[~held])
        oof[held] = model.predict_proba(X[held])[:, 1]
        print(f"fold {k}: trained on {(~held).sum()} rows, predicted {held.sum()} rows")

    assert not np.isnan(oof).any()
    out = pd.DataFrame({
        "id": split["id"],
        "document_family": split["document_family"],
        "record_id": split["record_id"],
        "tamper_type": split["tamper_type"],
        "final_label": split["final_label"],
        "fold": fold,
        "m3_probability_oof": np.round(oof, 6),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"saved {OUT.relative_to(PROJECT_ROOT)}")

    if REFERENCE.exists():
        ref = pd.read_csv(REFERENCE)
        diff = np.abs(out["m3_probability_oof"].values - ref["m3_probability_oof"].values)
        print(f"check vs {REFERENCE.name}: max |diff| = {diff.max():.2e}")


if __name__ == "__main__":
    main()
