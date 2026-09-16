"""
baseline.py
-----------
Precomputed Consistency Baselines for M3 Phase 1 — Part 1.

Models implemented:
1. Learned Baseline:
   - Scikit-learn Logistic Regression trained strictly on the 5 consistency features:
     [qr_readable, text_qr_match_score, checksum_valid, format_valid, missing_field_count].
   - Configured with random_state=42 and class_weight='balanced' to handle the 10:1
     imbalance (1,000 genuine vs 10,000 forged).

2. Rule-Based Sanity Baseline (Heuristic Floor):
   - Deterministic rule flagging forgery if any hard cryptographic/format rule fails:
     forged = (checksum_valid == False) OR (format_valid == False) OR (qr_readable == False).
   - Establishes a floor for what simple logical verification catches without learning.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from . import config


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    max_iter: int = 1000
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline model on the precomputed consistency features.
    """
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a learned classifier on feature matrix X and ground truth y.
    Returns: accuracy, precision, recall, f1, and roc_auc.
    """
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(set(y)) > 1 and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def rule_based_baseline(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Heuristic rule-based verification:
    Flag forged (1) if:
      - checksum_valid == 0 (invalid checksum)
      - format_valid == 0 (invalid format regex)
      - qr_readable == 0 (unreadable QR)
    Otherwise genuine (0).

    Expected feature order:
      [qr_readable, text_qr_match_score, checksum_valid, format_valid, missing_field_count]
    """
    if feature_names is None:
        feature_names = config.CONSISTENCY_FEATURE_ORDER

    idx = {name: i for i, name in enumerate(feature_names)}
    qr_readable = X[:, idx["qr_readable"]]
    checksum_valid = X[:, idx["checksum_valid"]]
    format_valid = X[:, idx["format_valid"]]

    is_forged = (checksum_valid < 0.5) | (format_valid < 0.5) | (qr_readable < 0.5)
    return is_forged.astype(np.int64)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate arbitrary binary predictions against ground truth targets.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics
