"""
part4_plots.py
--------------
Generates publication-quality diagnostic plots for M3 Part 4:
- ROC curves (comparing baselines and final model)
- Precision-Recall curves
- Calibration curve (reliability diagram)
- Confusion matrix
- Permutation feature importance horizontal bar chart
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

OUTPUTS_DIR = Path("outputs")


def plot_roc_curve(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],  # label -> (fpr, tpr, auc)
    output_path: Optional[Path] = None,
) -> Path:
    """Plots one or more ROC curves."""
    out_p = output_path or OUTPUTS_DIR / "roc_curve.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    for label, (fpr, tpr, auc_val) in curves.items():
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) — Test Set", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close()
    return out_p


def plot_pr_curve(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],  # label -> (recall, precision, pr_auc)
    output_path: Optional[Path] = None,
) -> Path:
    """Plots Precision-Recall curves."""
    out_p = output_path or OUTPUTS_DIR / "pr_curve.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    for label, (rec, prec, pr_auc_val) in curves.items():
        plt.plot(rec, prec, label=f"{label} (PR-AUC = {pr_auc_val:.3f})", lw=2)

    # 10:1 class imbalance => positive fraction ~ 0.909
    plt.axhline(y=10.0 / 11.0, color="k", linestyle="--", lw=1.5, label="Base Rate (~0.909)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall (PR) Curve — Test Set", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close()
    return out_p


def plot_calibration_curve(
    prob_true: List[float],
    prob_pred: List[float],
    brier_score: float,
    model_name: str,
    output_path: Optional[Path] = None,
) -> Path:
    """Plots calibration reliability diagram."""
    out_p = output_path or OUTPUTS_DIR / "calibration_curve.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, "s-", color="#1f77b4", lw=2, label=f"{model_name} (Brier = {brier_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Forged)", fontsize=12)
    plt.title("Calibration Curve (Reliability Diagram) — Test Set", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close()
    return out_p


def plot_confusion_matrix(
    cm: List[List[int]],
    model_name: str,
    output_path: Optional[Path] = None,
) -> Path:
    """Plots confusion matrix heatmap."""
    out_p = output_path or OUTPUTS_DIR / "confusion_matrix.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix — {model_name} (Test)", fontsize=13, fontweight="bold")
    plt.colorbar()

    classes = ["Genuine (0)", "Forged (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    # Label counts inside cells
    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(
                j, i, format(cm_arr[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold"
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close()
    return out_p


def plot_feature_importance(
    importance_list: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plots horizontal bar chart of permutation feature importance."""
    out_p = output_path or OUTPUTS_DIR / "feature_importance.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Reverse so top feature is at the top
    items = list(reversed(importance_list))
    names = [x["feature"] for x in items]
    means = [x["importance_mean"] for x in items]
    stds = [x["importance_std"] for x in items]

    plt.figure(figsize=(9, 7))
    plt.barh(names, means, xerr=stds, align="center", color="#2ca02c", ecolor="black", alpha=0.85, capsize=3)
    plt.xlabel("Mean Permutation Importance (ROC-AUC drop)", fontsize=12)
    plt.title("Permutation Feature Importance — Test Set", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close()
    return out_p
