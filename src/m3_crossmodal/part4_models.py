"""
part4_models.py
---------------
Model definitions, pipelines, and evaluation metrics for M3 Part 4.

Models implemented:
1. Logistic Regression (with explicit missing indicators + median imputation + standard scaler)
2. Random Forest (with explicit missing indicators + median imputation + balanced class weighting)
3. HistGradientBoostingClassifier (native NaN support, class_weight='balanced')

All stochastic components use fixed random_state for strict reproducibility.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .part4_features import assert_no_forbidden_features

RANDOM_SEED = 42
MODELS_DIR = Path("outputs/models")


# ---------------------------------------------------------------------------
# Pipeline Factory Functions
# ---------------------------------------------------------------------------

def create_logistic_regression_pipeline(
    C: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """
    Creates a Logistic Regression pipeline with:
      - SimpleImputer(strategy='median', add_indicator=True)
      - StandardScaler()
      - LogisticRegression(class_weight='balanced')
    Missing indicators ensure missingness (unreadable QR/OCR) is never confused
    with zero similarity.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
            solver="lbfgs",
        )),
    ])


def create_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = 8,
    min_samples_split: int = 4,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """
    Creates a Random Forest pipeline with:
      - SimpleImputer(strategy='median', add_indicator=True)
      - RandomForestClassifier(class_weight='balanced')
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def create_hist_gradient_boosting(
    learning_rate: float = 0.1,
    max_iter: int = 100,
    max_depth: Optional[int] = 5,
    min_samples_leaf: int = 20,
    random_state: int = RANDOM_SEED,
) -> HistGradientBoostingClassifier:
    """
    Creates a HistGradientBoostingClassifier:
      - Natively handles NaN values without imputation (missing values branch left/right)
      - Directly supports class_weight='balanced' in scikit-learn 1.6.1
    """
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Evaluation & Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    split: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "threshold": round(float(self.threshold), 4),
            "accuracy": round(float(self.accuracy), 4),
            "precision": round(float(self.precision), 4),
            "recall": round(float(self.recall), 4),
            "f1": round(float(self.f1), 4),
            "roc_auc": round(float(self.roc_auc), 4),
            "pr_auc": round(float(self.pr_auc), 4),
            "brier_score": round(float(self.brier_score), 4),
            "confusion_matrix": self.confusion_matrix,
        }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "macro_f1",
) -> Tuple[float, float]:
    """
    Sweeps decision threshold on validation probabilities to find the optimal operating point.
    Under 10:1 class imbalance, naive positive-class F1 degenerates to predicting all-forged
    (TN=0, F1=0.9524). Using 'macro_f1' balances discrimination across both genuine and forged classes.
    Returns (best_threshold, best_score).
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_th = 0.5
    best_score = -1.0

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        if criterion == "macro_f1":
            score = float(f1_score(y_true, preds, average="macro", zero_division=0))
        elif criterion == "balanced_accuracy":
            from sklearn.metrics import balanced_accuracy_score
            score = float(balanced_accuracy_score(y_true, preds))
        else:
            score = float(f1_score(y_true, preds, zero_division=0))

        if score > best_score:
            best_score = score
            best_th = float(th)

    return best_th, best_score


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "test",
) -> EvaluationMetrics:
    """
    Computes all required threshold-dependent and threshold-free metrics.
    """
    assert_no_forbidden_features(X.columns)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-df))
    else:
        raise AttributeError("Model must support predict_proba or decision_function.")

    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y, y_prob))
    except Exception:
        pr_auc = 0.0

    brier = float(brier_score_loss(y, y_prob))
    cm = confusion_matrix(y, y_pred).tolist()

    return EvaluationMetrics(
        split=split_name,
        threshold=threshold,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        confusion_matrix=cm,
    )


def save_model_artifact(
    model: Any,
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Serializes a trained model to outputs/models/{model_name}.joblib."""
    out_dir = output_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{model_name}.joblib"
    joblib.dump(model, artifact_path)
    return artifact_path


def load_model_artifact(
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Any:
    """Loads a serialized model artifact."""
    out_dir = output_dir or MODELS_DIR
    artifact_path = out_dir / f"{model_name}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    return joblib.load(artifact_path)
