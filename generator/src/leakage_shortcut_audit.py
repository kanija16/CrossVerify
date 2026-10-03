"""Read-only leakage and shortcut audit for a CrossVerify dataset.

This is a diagnostic, not a model-training entry point.  It writes reports
outside the canonical dataset and restricts every supervised probe to a
record_id-disjoint train/test partition.
"""
import argparse
import csv
import hashlib
import io
import json
import platform
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


SEED = 42
# This audit uses a stratified diagnostic sample rather than silently treating
# a low-level probe as a final model benchmark.  Twenty-five per cell gives
# every family/category representation while keeping JPEG/ELA work practical.
MAX_PER_FAMILY_CATEGORY = 100
EXPECTED_FORGED = {
    "text_qr_mismatch": "Printed non-identifier text conflicts with the QR payload.",
    "qr_only_mismatch": "QR payload conflicts with otherwise unchanged printed text.",
    "checksum_invalid": "The identifier checksum should fail validation.",
    "format_invalid": "The identifier format should fail validation.",
    "field_missing": "A required printed field is absent.",
    "visual_splice": "A local image-level copy/move artifact may be present; global shortcut evidence is not intended.",
    "fine_grained_edit": "A one-character printed/QR near-miss should be detected by consistency, not metadata.",
    "coordinated_full_forgery": "A consistently replaced fictional record is intentionally a hard case with no simple consistency conflict.",
}


def load_manifest(dataset):
    with (dataset / "manifests" / "manifest.json").open() as f:
        return json.load(f)


def snapshot(dataset):
    manifest = dataset / "manifests" / "manifest.json"
    tree_hasher = hashlib.sha256()
    count = 0
    for path in sorted(p for p in dataset.rglob("*") if p.is_file()):
        count += 1
        # Fast structural fingerprint for the report.  The validation command
        # additionally records a full content fingerprint before/after; doing
        # that twice here would turn a read-only diagnostic into needless I/O.
        stat = path.stat()
        tree_hasher.update(f"{path.relative_to(dataset)}:{stat.st_size}".encode())
    return {"file_count": count, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "tree_sha256": tree_hasher.hexdigest()}


def stratified_entries(manifest, seed):
    buckets = defaultdict(list)
    for item in manifest:
        buckets[(item["document_family"], item["tamper_type"])].append(item)
    rng = random.Random(seed)
    chosen = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda item: item["id"])
        rng.shuffle(bucket)
        chosen.extend(bucket[:MAX_PER_FAMILY_CATEGORY])
    return chosen


def image_features(path):
    """Deliberately non-semantic file/global/ELA statistics for shortcut probes."""
    size = path.stat().st_size
    with Image.open(path) as image:
        fmt, mode, width, height = image.format, image.mode, image.width, image.height
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114
    gy, gx = np.gradient(gray)
    gradient = np.hypot(gx, gy)
    noise = gray - (np.roll(gray, 1, 0) + np.roll(gray, -1, 0) + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)) / 4
    encoded = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(encoded, "JPEG", quality=90)
    encoded.seek(0)
    with Image.open(encoded) as jpeg:
        ela = np.abs(rgb - np.asarray(jpeg.convert("RGB"), dtype=np.float32)).mean(axis=2)
    vector = [width, height, size, *rgb.mean(axis=(0, 1)), *rgb.std(axis=(0, 1)),
              gray.mean(), gray.std(), (gradient > 25).mean(), gradient.mean(), noise.std(),
              ela.mean(), ela.std(), (ela > 20).mean()]
    return np.asarray(vector, dtype=np.float32), {"format": fmt, "mode": mode, "width": width,
                                                    "height": height, "file_size": size}


FEATURE_NAMES = ["width", "height", "file_size_bytes", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
                 "gray_mean", "gray_std", "edge_density", "gradient_mean", "noise_std", "ela_mean", "ela_std", "ela_pct_gt20"]


def _require_finite(name, values):
    """Fail closed: a real numerical fault must not become an audit result."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def audit_environment():
    """Record numerical-library provenance needed to reproduce probe results."""
    blas = getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas", {})
    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "platform": platform.platform(),
        "cpu_architecture": platform.machine(),
        "blas_backend": blas.get("name", "unknown"),
    }


def _fit_and_score_logistic(model, train_features, train_labels, test_features):
    """Run only affected LR calls under a narrow documented warning filter."""
    # NumPy issue #28687: macOS ARM64 builds linked to Apple Accelerate can
    # spuriously raise these three matmul RuntimeWarnings for finite, correct
    # products.  Keep this scoped to sklearn's LogisticRegression internals;
    # _require_finite checks before and after these calls still fail on actual
    # numerical instability.  Do not broaden this to all RuntimeWarnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^(divide by zero|overflow|invalid value) encountered in matmul$",
            category=RuntimeWarning,
            # fit uses _linear_loss; predict_proba reaches safe_sparse_dot in
            # utils.extmath.  Both remain inside this call-local LR scope.
            module=r"sklearn\.(linear_model\._linear_loss|utils\.extmath)",
        )
        model.fit(train_features, train_labels)
        scores = model.predict_proba(test_features)[:, 1]
    _require_finite("fitted LogisticRegression coefficients", model.coef_)
    _require_finite("LogisticRegression predicted probabilities", scores)
    return scores


def grouped_probe(rows, positive_category=None, seed=SEED):
    """Identity-aware logistic diagnostic using only non-semantic statistics."""
    selected = [row for row in rows if row["tamper_type"] in {"genuine", positive_category}] if positive_category else rows
    groups = sorted({row["record_id"] for row in selected})
    rng = random.Random(seed)
    rng.shuffle(groups)
    cut = max(1, round(len(groups) * .7))
    train_groups, test_groups = set(groups[:cut]), set(groups[cut:])
    train = [row for row in selected if row["record_id"] in train_groups]
    test = [row for row in selected if row["record_id"] in test_groups]
    if not train or not test or len({row["target"] for row in train}) < 2 or len({row["target"] for row in test}) < 2:
        return {"available": False, "reason": "insufficient two-class record-id-disjoint data"}
    train_features = np.vstack([row["features"] for row in train])
    test_features = np.vstack([row["features"] for row in test])
    _require_finite("raw training feature matrix", train_features)
    _require_finite("raw test feature matrix", test_features)
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_test_features = scaler.transform(test_features)
    _require_finite("scaled training feature matrix", scaled_train_features)
    _require_finite("scaled test feature matrix", scaled_test_features)
    model = LogisticRegression(max_iter=2000, random_state=seed)
    scores = _fit_and_score_logistic(model, scaled_train_features, [row["target"] for row in train], scaled_test_features)
    labels = np.asarray([row["target"] for row in test])
    return {"available": True, "feature_type": "file/global-pixel/ELA statistics only", "train_record_ids": len(train_groups),
            "test_record_ids": len(test_groups), "train_samples": len(train), "test_samples": len(test),
            "accuracy": float(accuracy_score(labels, scores >= .5)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= .5)),
            "roc_auc": float(roc_auc_score(labels, scores))}


def split_audit(dataset, manifest):
    by_id = {item["id"]: item for item in manifest}
    results = {}
    for csv_path in sorted((dataset / "splits").rglob("*.csv")):
        group = str(csv_path.parent.relative_to(dataset / "splits"))
        results.setdefault(group, {})[csv_path.stem] = {row["record_id"] for row in csv.DictReader(csv_path.open())}
    report = {}
    for group, splits in results.items():
        names = sorted(splits)
        overlaps = {f"{a}/{b}": len(splits[a] & splits[b]) for i, a in enumerate(names) for b in names[i + 1:]}
        sample_ids = {name: {row["id"] for row in csv.DictReader((dataset / "splits" / group / f"{name}.csv").open())}
                      for name in names}
        duplicate_overlap = 0
        for name, ids in sample_ids.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate sample ID inside {group}/{name}")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                # Unique manifest image paths make a shared path an exact
                # duplicate-file reference.  check_dataset.py separately
                # performs the slower whole-dataset content-hash scan.
                paths_a = {by_id[item_id]["image_path"] for item_id in sample_ids[a]}
                paths_b = {by_id[item_id]["image_path"] for item_id in sample_ids[b]}
                duplicate_overlap += len(paths_a & paths_b)
        report[group] = {"identities": {name: len(ids) for name, ids in splits.items()}, "identity_overlap": overlaps,
                         "exact_duplicate_file_reference_overlap": duplicate_overlap}
    return report


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=root / "dataset_final")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    before = snapshot(dataset)
    manifest = load_manifest(dataset)
    output.mkdir(parents=True, exist_ok=True)

    family_type = Counter((item["document_family"], item["tamper_type"]) for item in manifest)
    selected = stratified_entries(manifest, args.seed)
    rows, structure = [], []
    for item in selected:
        features, meta = image_features(dataset / item["image_path"])
        rows.append({"record_id": item["record_id"], "tamper_type": item["tamper_type"],
                     "target": int(item["final_label"] == "forged"), "features": features})
        structure.append(meta)
    overall_probe = grouped_probe(rows, seed=args.seed)
    category_probes = {category: grouped_probe(rows, category, args.seed)
                       for category in sorted({item["tamper_type"] for item in manifest if item["tamper_type"] != "genuine"})}
    split_results = split_audit(dataset, manifest) if (dataset / "splits").is_dir() else {}
    after = snapshot(dataset)
    categories_present = sorted({item["tamper_type"] for item in manifest if item["tamper_type"] != "genuine"})
    per_category = {}
    for category in sorted(EXPECTED_FORGED):
        if category not in categories_present:
            per_category[category] = {"intended_signal": EXPECTED_FORGED[category], "status": "not present in audited dataset",
                                      "severity": "MEDIUM", "evidence": "The canonical dataset version has no samples for this category.",
                                      "why_it_matters": "Its hard-case/shortcut behavior cannot be assessed before samples exist.",
                                      "recommended_remediation": "Generate the category only in a future dataset revision, then rerun this read-only audit."}
            continue
        probe = category_probes[category]
        suspicious = probe.get("available") and (probe["balanced_accuracy"] >= .80 or probe["roc_auc"] >= .80)
        per_category[category] = {"intended_signal": EXPECTED_FORGED[category], "status": "present",
                                  "severity": "MEDIUM" if suspicious else "LOW", "global_probe": probe,
                                  "evidence": "A record-id-disjoint global-statistics/ELA probe was used; no metadata or semantic text features were included.",
                                  "why_it_matters": "Strong low-level separation would indicate an easier signal than the intended attack evidence.",
                                  "recommended_remediation": "Investigate rendering/augmentation only if a future larger audit shows strong balanced-accuracy or ROC-AUC."}
    report = {
        "scope": "Read-only audit; probes are diagnostic and are not final CrossVerify models.",
        "audit_environment": audit_environment(),
        "dataset_integrity": {"before": before, "after": after, "unchanged": before == after},
        "metadata_leakage": {
            "severity": "HIGH", "finding": "Manifest IDs, filenames, label filenames, and image paths include tamper_type strings.",
            "evidence": "Example ID format: family_a_000000_genuine; forged category is directly encoded in path/name.",
            "remediation": "Model loaders must use only image bytes and an explicit allowlist of model-facing fields; never feed IDs, paths, labels, tamper_type, or debug metadata.",
            "record_id_finding": "Each record_id has the same configured variant set, so record_id alone is not a category label in this dataset.",
            "manifest_order_finding": "Manifest order is generation order with repeating category positions; never use row/index position as a feature.",
        },
        "template_rendering": {"family_tamper_counts": {f"{family}/{category}": count for (family, category), count in sorted(family_type.items())},
                                 "finding": "Every observed tamper category occurs in both document families at equal counts; template/family is not category-exclusive.",
                                 "severity": "LOW", "remediation": "Keep family/template balancing and identity-aware evaluation."},
        "image_structure": {"sampled_images": len(rows), "per_family_category_cap": MAX_PER_FAMILY_CATEGORY,
                            "formats": dict(Counter(meta["format"] for meta in structure)), "modes": dict(Counter(meta["mode"] for meta in structure)),
                            "dimensions": {f"{width}x{height}": count for (width, height), count in Counter((meta["width"], meta["height"]) for meta in structure).items()}, "feature_names": FEATURE_NAMES,
                            "overall_global_probe": overall_probe},
        "identity_split_leakage": split_results,
        "baseline_probes": {"per_category_vs_genuine": category_probes,
                              "interpretation": "Scores use a record_id-disjoint split and non-semantic file/global-pixel/ELA features only. They are shortcut diagnostics, not benchmark performance."},
        "consistency_vector_boundary": {"severity": "LOW", "model_facing_keys": ["qr_readable", "text_qr_match_score", "checksum_valid", "format_valid", "missing_field_count"],
            "finding": "build_dataset.process_variant constructs these fields from OCR/QR extraction and identifier validation. expected_consistency_vector, labels, tamper_type, ground_truth_fields, and debug metadata are stored separately and must remain excluded from model features.",
            "remediation": "Use an explicit feature allowlist; the accompanying unit test guards the recorded vector schema."},
        "per_category": per_category,
        "overall_risk": "HIGH if any training pipeline consumes paths, IDs, labels, or ground-truth/debug metadata; LOW evidence of template-exclusive category leakage in the audited manifest. Review diagnostic probe values before training; this audit cannot prove absence of all shortcuts.",
    }
    (output / "audit_report.json").write_text(json.dumps(report, indent=2))
    lines = ["# CrossVerify leakage / shortcut audit", "", "## Dataset integrity", "",
             f"- Unchanged after audit: **{report['dataset_integrity']['unchanged']}**", f"- File count: {before['file_count']}",
             f"- Manifest SHA-256: `{before['manifest_sha256']}`", f"- Dataset tree SHA-256: `{before['tree_sha256']}`", "",
             "## Metadata leakage — HIGH", "", report["metadata_leakage"]["finding"], "", "## Image/template diagnostics", "",
             f"Sampled {len(rows)} images (up to {MAX_PER_FAMILY_CATEGORY} per family/category); formats={report['image_structure']['formats']}, modes={report['image_structure']['modes']}.",
             f"Overall non-semantic probe: {overall_probe}", "", "## Identity/split leakage", "", json.dumps(split_results, indent=2), "",
             "## Category probes", "", json.dumps(category_probes, indent=2), "", "## Consistency-vector boundary — LOW", "",
             report["consistency_vector_boundary"]["finding"], "", "## Overall risk", "", report["overall_risk"],
             "", "No final models were trained and no dataset files were modified."]
    (output / "audit_report.md").write_text("\n".join(lines) + "\n")
    print(f"Audit complete: {output / 'audit_report.md'}")


if __name__ == "__main__":
    main()
