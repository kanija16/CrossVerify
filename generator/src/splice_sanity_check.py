"""Read-only forensic sanity check for genuine versus visual-splice samples.

ELA on these PNG-generated synthetic documents is a preliminary visualization
only: recompressing a PNG as JPEG does not reproduce real-world camera/JPEG
forensic conditions.  The script never writes inside the supplied dataset.

Example:
    python3 splice_sanity_check.py --dataset ../dataset_final
"""
import argparse
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
ELA_JPEG_QUALITY = 90
ELA_THRESHOLD = 20
MAX_GROUPS = 500
REPRESENTATIVE_PAIRS = 10


def load_manifest(dataset_dir):
    with (dataset_dir / "manifests" / "manifest.json").open() as handle:
        return json.load(handle)


def select_matched_samples(manifest, seed):
    """Select one genuine and one deterministic splice per record group."""
    by_record = defaultdict(list)
    for entry in manifest:
        if entry["tamper_type"] in {"genuine", "visual_splice"}:
            by_record[entry["record_id"]].append(entry)

    groups = []
    for record_id, entries in by_record.items():
        genuine = [entry for entry in entries if entry["tamper_type"] == "genuine"]
        splices = [entry for entry in entries if entry["tamper_type"] == "visual_splice"]
        if genuine and splices:
            groups.append((record_id, genuine[0], sorted(splices, key=lambda item: item["id"])))

    rng = random.Random(seed)
    groups.sort(key=lambda item: item[0])
    selected = rng.sample(groups, min(MAX_GROUPS, len(groups)))
    pairs = []
    for record_id, genuine, splices in selected:
        pairs.append((record_id, genuine, rng.choice(splices)))
    return pairs


def image_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def ela_image_and_stats(rgb):
    """Create an in-memory JPEG recompression ELA visualization and stats."""
    original = Image.fromarray(rgb, "RGB")
    encoded = io.BytesIO()
    original.save(encoded, format="JPEG", quality=ELA_JPEG_QUALITY)
    encoded.seek(0)
    with Image.open(encoded) as recompressed:
        recompressed_array = np.asarray(recompressed.convert("RGB"), dtype=np.int16)
    difference = np.abs(rgb.astype(np.int16) - recompressed_array).astype(np.uint8)
    error = difference.mean(axis=2)
    stats = {
        "mean_error": float(error.mean()),
        "std_error": float(error.std()),
        "max_error": int(error.max()),
        "pct_pixels_above_threshold": float((error > ELA_THRESHOLD).mean() * 100),
    }
    return np.clip(difference.astype(np.int16) * 8, 0, 255).astype(np.uint8), stats


def aggregate_ela(stats):
    return {
        name: float(np.mean([item[name] for item in stats]))
        for name in ("mean_error", "std_error", "max_error", "pct_pixels_above_threshold")
    }


def image_features(rgb, include_histogram=True, include_gradients=True):
    """Extract deliberately conservative image-derived, metadata-free features."""
    data = rgb.astype(np.float32)
    gray = data[..., 0] * 0.299 + data[..., 1] * 0.587 + data[..., 2] * 0.114
    names = ["rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "rgb_std_r", "rgb_std_g", "rgb_std_b",
             "gray_mean", "gray_std"]
    values = list(data.mean(axis=(0, 1))) + list(data.std(axis=(0, 1))) + [gray.mean(), gray.std()]

    grad_y, grad_x = np.gradient(gray)
    gradient = np.hypot(grad_x, grad_y)
    laplacian = (-4 * gray + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
                 + np.roll(gray, 1, 1) + np.roll(gray, -1, 1))[1:-1, 1:-1]
    names.extend(["edge_density", "laplacian_variance"])
    values.extend([float((gradient > 25).mean()), float(laplacian.var())])

    if include_gradients:
        names.extend(["gradient_mean", "gradient_std"])
        values.extend([float(gradient.mean()), float(gradient.std())])
    if include_histogram:
        histogram, _ = np.histogram(gray, bins=16, range=(0, 256), density=True)
        names.extend([f"gray_hist_{index:02d}" for index in range(16)])
        values.extend(histogram.tolist())
    return np.asarray(values, dtype=np.float32), names


def metric_report(y_true, predictions, scores):
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix": matrix,
    }


def fit_and_evaluate(features, labels, group_ids, seed):
    unique_groups = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    split_at = round(len(unique_groups) * 0.70)
    train_groups, test_groups = set(unique_groups[:split_at]), set(unique_groups[split_at:])
    train_indices = [index for index, group in enumerate(group_ids) if group in train_groups]
    test_indices = [index for index, group in enumerate(group_ids) if group in test_groups]

    # --- diagnostic: find near-zero-variance columns before fitting ---
    variances = features[train_indices].var(axis=0)
    for col_index, v in enumerate(variances):
        print(f"feature[{col_index}]: variance={v:.3e}")
    # --- end diagnostic ---

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    model.fit(features[train_indices], labels[train_indices])
    scores = model.predict_proba(features[test_indices])[:, 1]
    predictions = (scores >= 0.5).astype(int)
    return metric_report(labels[test_indices], predictions, scores), {
        "train_record_ids": len(train_groups),
        "test_record_ids": len(test_groups),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "test_labels": labels[test_indices].tolist(),
        "test_scores": scores.tolist(),
    }


def qualitative_interpretation(metrics, test_count):
    """Use the pilot's binomial chance uncertainty rather than a pass/fail cut-off."""
    chance_standard_error = math.sqrt(0.25 / test_count)
    excess = metrics["balanced_accuracy"] - 0.5
    if excess <= 2 * chance_standard_error:
        return "No strong global-statistics shortcut detected."
    if excess <= 5 * chance_standard_error:
        return "Some global image signal is present; inspect ELA examples and investigate before proceeding."
    return "Potentially strong shortcut/artifact detected; investigate before changing the generator."


def save_plots(output_dir, genuine_ela, splice_ela, full_metrics, evaluation):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist([item["mean_error"] for item in genuine_ela], bins=16, alpha=0.65, label="genuine")
    plt.hist([item["mean_error"] for item in splice_ela], bins=16, alpha=0.65, label="visual_splice")
    plt.xlabel("Mean ELA error (JPEG quality 90)")
    plt.ylabel("Images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "ela_distribution.png", dpi=150)
    plt.close()

    matrix = np.asarray(full_metrics["confusion_matrix"])
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["genuine", "splice"])
    plt.yticks([0, 1], ["genuine", "splice"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(plots_dir / "classifier_confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(evaluation["test_labels"], evaluation["test_scores"])
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {full_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curve.png", dpi=150)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=project_root / "dataset_final")
    parser.add_argument("--output", type=Path, default=project_root / "outputs" / "splice_sanity_check")
    args = parser.parse_args()
    dataset_dir, output_dir = args.dataset.resolve(), args.output.resolve()
    manifest_path = dataset_dir / "manifests" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ela_dir = output_dir / "ela"
    ela_dir.mkdir(exist_ok=True)
    pairs = select_matched_samples(load_manifest(dataset_dir), SEED)
    if not pairs:
        raise RuntimeError("No matched genuine/visual_splice record groups found")

    features, labels, group_ids = [], [], []
    genuine_ela, splice_ela = [], []
    for pair_index, (record_id, genuine, splice) in enumerate(pairs, start=1):
        for entry, label, ela_bucket in ((genuine, 0, genuine_ela), (splice, 1, splice_ela)):
            rgb = image_array(dataset_dir / entry["image_path"])
            feature_vector, feature_names = image_features(rgb)
            features.append(feature_vector)
            labels.append(label)
            group_ids.append(record_id)
            ela, stats = ela_image_and_stats(rgb)
            ela_bucket.append(stats)
            if pair_index <= REPRESENTATIVE_PAIRS:
                suffix = "genuine" if label == 0 else "splice"
                Image.fromarray(ela, "RGB").save(ela_dir / f"pair_{pair_index:02d}_{suffix}.png")

    features, labels = np.vstack(features), np.asarray(labels, dtype=int)
    global_indices = [feature_names.index(name) for name in (
        "rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "rgb_std_r", "rgb_std_g", "rgb_std_b",
        "gray_mean", "gray_std", "edge_density", "laplacian_variance")]
    full_metrics, evaluation = fit_and_evaluate(features, labels, group_ids, SEED)
    shortcut_metrics, shortcut_evaluation = fit_and_evaluate(features[:, global_indices], labels, group_ids, SEED)
    interpretation = qualitative_interpretation(shortcut_metrics, shortcut_evaluation["test_samples"])

    results = {
        "dataset_path": str(dataset_dir),
        "seed": SEED,
        "command": f"python3 splice_sanity_check.py --dataset {args.dataset}",
        "selection": {
            "genuine_samples": len(pairs), "visual_splice_samples": len(pairs),
            "unique_record_ids": len(pairs), "matched_pairs_groups": len(pairs),
        },
        "ela": {
            "jpeg_quality": ELA_JPEG_QUALITY, "pixel_threshold": ELA_THRESHOLD,
            "genuine_aggregate": aggregate_ela(genuine_ela),
            "visual_splice_aggregate": aggregate_ela(splice_ela),
            "caveat": "ELA is a PNG-to-JPEG diagnostic visualization, not proof of real-world JPEG forensic behavior.",
        },
        "classifier": {"feature_names": feature_names, "metrics": full_metrics,
                       "majority_baseline_accuracy": 0.5, "train_test": {key: value for key, value in evaluation.items() if key not in {"test_labels", "test_scores"}}},
        "global_shortcut_classifier": {"feature_names": [feature_names[index] for index in global_indices], "metrics": shortcut_metrics,
                                        "interpretation": interpretation},
        "warnings": ["Read-only diagnostic: no dataset files are modified.",
                     "Small pilot results are descriptive and should not be treated as statistical proof."],
    }
    with (output_dir / "results.json").open("w") as handle:
        json.dump(results, handle, indent=2)
    save_plots(output_dir, genuine_ela, splice_ela, full_metrics, evaluation)

    text = [
        "CrossVerify visual_splice sanity check", "=" * 40,
        f"Command: {results['command']}", f"Dataset: {dataset_dir}", f"Seed: {SEED}",
        f"Selected: {len(pairs)} genuine + {len(pairs)} visual_splice across {len(pairs)} matched record groups.",
        "", "ELA (PNG-to-JPEG diagnostic only; not proof of real-world JPEG forensic behavior):",
        f"  Genuine mean ELA error: {results['ela']['genuine_aggregate']['mean_error']:.3f}",
        f"  Splice mean ELA error: {results['ela']['visual_splice_aggregate']['mean_error']:.3f}",
        "", "Image-statistics LogisticRegression (grouped 70/30 record-id split):",
    ]
    for name, value in full_metrics.items():
        text.append(f"  {name}: {value}")
    text.extend(["", "Global-statistics-only LogisticRegression:"])
    for name, value in shortcut_metrics.items():
        text.append(f"  {name}: {value}")
    text.extend(["", "Interpretation:", interpretation,
                 "ELA differences are preliminary image-level diagnostics; this pilot does not establish a real copy-move forensic signature."])
    (output_dir / "results.txt").write_text("\n".join(text) + "\n")

    print("\n".join(text))


if __name__ == "__main__":
    main()
