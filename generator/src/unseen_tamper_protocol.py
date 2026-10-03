"""Read-only, identity-aware unseen-tamper protocol for CrossVerify."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CANONICAL_FORGED_TYPES = ("text_qr_mismatch", "qr_only_mismatch", "checksum_invalid", "format_invalid", "field_missing", "visual_splice", "coordinated_full_forgery", "fine_grained_edit")
MODEL_FACING_FEATURE_KEYS = ("qr_readable", "text_qr_match_score", "checksum_valid", "format_valid", "missing_field_count")
PROHIBITED_MODEL_FIELDS = {"tamper_type", "held_out_category", "final_label", "expected_consistency_vector", "ground_truth_fields", "record_id", "id", "image_path", "label_path", "filename", "debug", "generator_metadata"}

def load_manifest(dataset_dir):
    return json.loads((Path(dataset_dir) / "manifests" / "manifest.json").read_text())

def dataset_snapshot(dataset_dir):
    """Deterministic whole-tree fingerprint; never modifies the dataset."""
    root, digest = Path(dataset_dir), hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    manifest = root / "manifests" / "manifest.json"
    return {"file_count": len(files), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "tree_sha256": digest.hexdigest()}

def _groups(manifest):
    result = defaultdict(list)
    for entry in manifest: result[entry["record_id"]].append(entry)
    return result

def _held_out_ids(groups, category, seed, fraction):
    by_family = defaultdict(list)
    for record_id, entries in groups.items():
        types = {x["tamper_type"] for x in entries}
        if {"genuine", category} <= types:
            family = next(x["document_family"] for x in entries if x["tamper_type"] == category)
            by_family[family].append(record_id)
    selected = []
    for family in sorted(by_family):
        candidates = sorted(by_family[family])
        random.Random(f"{seed}:{category}:{family}").shuffle(candidates)
        selected += candidates[:max(1, int(len(candidates) * fraction))]
    return set(selected)

def make_unseen_tamper_split(manifest, holdout_type, seed=42, test_identity_fraction=0.2):
    """Test has held-out forgery + genuine controls; train excludes the category."""
    if holdout_type not in CANONICAL_FORGED_TYPES: raise ValueError(f"Unsupported category: {holdout_type}")
    groups = _groups(manifest)
    test_ids = _held_out_ids(groups, holdout_type, seed, test_identity_fraction)
    if not test_ids: raise ValueError(f"{holdout_type}: no genuine/held-out identity pairs")
    train, test = [], []
    for record_id in sorted(groups):
        entries = sorted(groups[record_id], key=lambda x: x["id"])
        if record_id in test_ids:
            test.extend(x for x in entries if x["tamper_type"] in {"genuine", holdout_type})
        else:
            train.extend(x for x in entries if x["tamper_type"] != holdout_type)
    return {"train": train, "test": test}

def _counts(items): return dict(sorted(Counter(x["tamper_type"] for x in items).items()))
def _image_hashes(dataset_dir, items):
    root = Path(dataset_dir)
    return {x["id"]: hashlib.sha256((root / x["image_path"]).read_bytes()).hexdigest() for x in items}

def validate_split(dataset_dir, split, category, seed, manifest):
    train, test = split["train"], split["test"]
    train_records, test_records = {x["record_id"] for x in train}, {x["record_id"] for x in test}
    train_images, test_images = {x["id"] for x in train}, {x["id"] for x in test}
    train_hashes, test_hashes = _image_hashes(dataset_dir, train), _image_hashes(dataset_dir, test)
    tc, ec = _counts(train), _counts(test)
    repeat = make_unseen_tamper_split(manifest, category, seed)
    alternate = make_unseen_tamper_split(manifest, category, seed + 1)
    available_other = [x for x in CANONICAL_FORGED_TYPES if x != category and any(m["tamper_type"] == x for m in manifest)]
    missing = [x for x in available_other if not tc.get(x)]
    checks = {"held_out_training_count_zero": tc.get(category, 0) == 0, "held_out_test_count_positive": ec.get(category, 0) > 0, "genuine_train_positive": tc.get("genuine", 0) > 0, "genuine_test_positive": ec.get("genuine", 0) > 0, "record_id_overlap_zero": not(train_records & test_records), "image_id_overlap_zero": not(train_images & test_images), "exact_duplicate_image_overlap_zero": not(set(train_hashes.values()) & set(test_hashes.values())), "same_seed_identical": split == repeat, "different_seed_changes_split": {x["id"] for x in test} != {x["id"] for x in alternate["test"]}, "remaining_available_categories_in_train": not missing}
    family_presence = {f: sum(1 for x in manifest if x["document_family"] == f and x["tamper_type"] == category) for f in sorted({x["document_family"] for x in manifest})}
    return {"held_out_category": category, "training_categories": sorted(tc), "test_categories": sorted(ec), "random_seed": seed, "train_sample_count": len(train), "test_sample_count": len(test), "genuine_train_count": tc.get("genuine", 0), "genuine_test_count": ec.get("genuine", 0), "forged_train_count": len(train)-tc.get("genuine", 0), "forged_test_count": len(test)-ec.get("genuine", 0), "per_category_train_counts": tc, "per_category_test_counts": ec, "train_identity_count": len(train_records), "test_identity_count": len(test_records), "identity_overlap": len(train_records & test_records), "image_id_overlap": len(train_images & test_images), "duplicate_image_overlap": len(set(train_hashes.values()) & set(test_hashes.values())), "family_holdout_presence": family_presence, "missing_remaining_training_categories": missing, "validation": checks, "valid": all(checks.values())}

def _feature_boundary(dataset_dir, manifest):
    invalid = []
    for entry in manifest:
        label = json.loads((Path(dataset_dir) / entry["label_path"]).read_text())
        if set(label.get("consistency_vector", {})) != set(MODEL_FACING_FEATURE_KEYS): invalid.append(entry["id"])
    return {"model_facing_feature_allowlist": list(MODEL_FACING_FEATURE_KEYS), "prohibited_fields": sorted(PROHIBITED_MODEL_FIELDS), "labels_with_nonconforming_consistency_vector": invalid, "passes": not invalid}

def write_protocol(dataset_dir, output_dir, seed=42, test_identity_fraction=0.2):
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    before, manifest = dataset_snapshot(dataset_dir), load_manifest(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True); reports = []
    for category in CANONICAL_FORGED_TYPES:
        if not any(x["tamper_type"] == category for x in manifest):
            reports.append({"held_out_category": category, "status": "NOT AVAILABLE IN CURRENT CANONICAL DATASET"}); continue
        split = make_unseen_tamper_split(manifest, category, seed, test_identity_fraction)
        report = validate_split(dataset_dir, split, category, seed, manifest); report["status"] = "VALID" if report["valid"] else "INVALID"
        directory = output_dir / category; directory.mkdir(exist_ok=True)
        for name in ("train", "test"): (directory / f"{name}_audit_manifest.json").write_text(json.dumps(split[name], indent=2) + "\n")
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n"); reports.append(report)
    after = dataset_snapshot(dataset_dir)
    protocol = {"protocol": "CrossVerify unseen-tamper evaluation (split validation only; no model training or metrics)", "dataset": str(dataset_dir.resolve()), "seed": seed, "test_identity_fraction": test_identity_fraction, "dataset_integrity_before": before, "dataset_integrity_after": after, "dataset_unchanged": before == after, "feature_boundary": _feature_boundary(dataset_dir, manifest), "experiments": reports}
    (output_dir / "protocol_report.json").write_text(json.dumps(protocol, indent=2) + "\n")
    lines = ["# CrossVerify unseen-tamper protocol report", "", "No model was trained and no performance metrics are reported.", "", f"- Seed: `{seed}`", f"- Dataset unchanged: `{before == after}`", f"- Feature-boundary allowlist passes: `{protocol['feature_boundary']['passes']}`", "", "| Held-out category | Status | Train / test | Identity overlap | Duplicate overlap |", "|---|---:|---:|---:|---:|"]
    for report in reports:
        lines.append(f"| {report['held_out_category']} | {report['status']} | {report.get('train_sample_count', '—')} / {report.get('test_sample_count', '—')} | {report.get('identity_overlap', '—')} | {report.get('duplicate_image_overlap', '—')} |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    return protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate read-only unseen-tamper audit splits.")
    parser.add_argument("--dataset", default="dataset_final"); parser.add_argument("--output", default="unseen_tamper_protocol"); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--test-identity-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.test_identity_fraction < 1: parser.error("--test-identity-fraction must be between 0 and 1")
    result = write_protocol(args.dataset, args.output, args.seed, args.test_identity_fraction)
    if not result["dataset_unchanged"] or not result["feature_boundary"]["passes"] or any(x.get("status") == "INVALID" for x in result["experiments"]): raise SystemExit("Protocol validation failed; see protocol_report.json")
    print(f"Protocol validation complete: {args.output}")
