"""Deterministic, read-only imbalanced evaluation subsets for CrossVerify.

The canonical dataset is never copied or modified.  This utility writes only
small JSON manifests referencing the selected canonical image/label paths.
When a training split is supplied, its record_ids are checked against the
evaluation candidate split before any sampling occurs.  Sampling individual
*evaluation* variants is then safe: no selected evaluation identity can occur
in training, while stratification prevents a large category (for example,
visual_splice) from dominating the forged pool.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SUPPORTED_RATIOS = ((1, 1), (1, 3), (1, 5), (1, 10))


def load_manifest(dataset_dir):
    path = Path(dataset_dir) / "manifests" / "manifest.json"
    with path.open() as f:
        return json.load(f)


def read_split_ids(path):
    with Path(path).open(newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def _stratified_sample(items, count, rng, key):
    """Sample nearly equally from deterministic strata, without duplicates."""
    buckets = defaultdict(list)
    for item in items:
        buckets[key(item)].append(item)
    strata = sorted(buckets)
    if count > len(items):
        raise ValueError(f"Requested {count} items from a pool of {len(items)}")

    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Round-robin allocation deliberately favours category coverage before
    # adding another item to any one category/family stratum.
    selected = []
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            if buckets[stratum] and len(selected) < count:
                selected.append(buckets[stratum].pop())
                progressed = True
        if not progressed:
            raise ValueError("Sampling pool exhausted before target was reached")
    return selected


def _ratio_name(forged, genuine):
    return f"{forged}:{genuine}"


def make_imbalanced_subset(manifest, ratio, seed, candidate_ids=None, train_record_ids=None):
    """Return one exact forged:genuine evaluation subset and its report.

    ``candidate_ids`` limits the source to an existing evaluation split.  If
    ``train_record_ids`` is given, a non-empty identity intersection is a hard
    error rather than a warning, protecting all train/test experiments.
    """
    forged_unit, genuine_units = ratio
    if ratio not in SUPPORTED_RATIOS:
        raise ValueError(f"Unsupported forged:genuine ratio: {_ratio_name(*ratio)}")
    by_id = {item["id"]: item for item in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("Manifest contains duplicate sample IDs")
    # Filter in manifest order.  ``candidate_ids`` is usually a set read from
    # CSV; iterating that set would make a same-seed run depend on Python's
    # per-process hash randomisation.
    candidates = [item for item in manifest if item["id"] in candidate_ids] if candidate_ids is not None else list(manifest)
    candidate_record_ids = {item["record_id"] for item in candidates}
    train_record_ids = set(train_record_ids or ())
    overlap = candidate_record_ids & train_record_ids
    if overlap:
        raise ValueError(f"Identity leakage: {len(overlap)} record_id(s) overlap train and evaluation candidates")

    genuine_pool = [item for item in candidates if item["final_label"] == "genuine"]
    forged_pool = [item for item in candidates if item["final_label"] == "forged"]
    max_forged = min(len(forged_pool) // forged_unit, len(genuine_pool) // genuine_units)
    if max_forged == 0:
        raise ValueError(f"Insufficient candidates for forged:genuine {_ratio_name(*ratio)}")
    forged_count = max_forged * forged_unit
    genuine_count = max_forged * genuine_units

    # Independent per-ratio seed derivation keeps a ratio stable even when
    # other requested ratios are added/removed from the CLI invocation.
    ratio_seed = f"{seed}:{_ratio_name(*ratio)}"
    forged_rng = random.Random(ratio_seed + ":forged")
    genuine_rng = random.Random(ratio_seed + ":genuine")
    forged = _stratified_sample(
        forged_pool, forged_count, forged_rng,
        key=lambda item: (item["document_family"], item["tamper_type"]),
    )
    genuine = _stratified_sample(
        genuine_pool, genuine_count, genuine_rng,
        key=lambda item: item["document_family"],
    )
    selected = forged + genuine
    random.Random(ratio_seed + ":order").shuffle(selected)

    ids = [item["id"] for item in selected]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate sample ID created during sampling")
    per_category = dict(sorted(Counter(item["tamper_type"] for item in forged).items()))
    report = {
        "ratio_forged_to_genuine": _ratio_name(*ratio),
        "seed": seed,
        "forged_samples": len(forged),
        "genuine_samples": len(genuine),
        "total_samples": len(selected),
        "forged_category_counts": per_category,
        "identities_represented": len({item["record_id"] for item in selected}),
        "candidate_identities": len(candidate_record_ids),
        "train_evaluation_identity_overlap": len(overlap),
        "duplicate_sample_ids": 0,
    }
    return selected, report


def dataset_snapshot(dataset_dir):
    root = Path(dataset_dir)
    manifest = root / "manifests" / "manifest.json"
    return {
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def write_protocol(dataset_dir, output_dir, ratios, seed, evaluation_csv=None, train_csv=None):
    """Write external subset manifests and a report; never writes dataset_dir."""
    manifest = load_manifest(dataset_dir)
    candidate_ids = read_split_ids(evaluation_csv) if evaluation_csv else None
    train_ids = read_split_ids(train_csv) if train_csv else set()
    by_id = {item["id"]: item for item in manifest}
    if candidate_ids is not None:
        unknown = candidate_ids - set(by_id)
        if unknown:
            raise ValueError(f"Evaluation split has {len(unknown)} ID(s) absent from manifest")
    if train_ids:
        unknown = train_ids - set(by_id)
        if unknown:
            raise ValueError(f"Training split has {len(unknown)} ID(s) absent from manifest")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for ratio in ratios:
        selected, report = make_imbalanced_subset(
            manifest, ratio, seed, candidate_ids=candidate_ids,
            train_record_ids={by_id[item_id]["record_id"] for item_id in train_ids},
        )
        filename = "ratio_" + _ratio_name(*ratio).replace(":", "_") + ".json"
        with (output / filename).open("w") as f:
            json.dump(selected, f, indent=2)
        report["subset_manifest"] = filename
        reports.append(report)

    protocol = {
        "protocol": "CrossVerify deterministic imbalanced evaluation subset",
        "dataset": str(Path(dataset_dir).resolve()),
        "dataset_integrity_snapshot": dataset_snapshot(dataset_dir),
        "seed": seed,
        "evaluation_csv": str(Path(evaluation_csv).resolve()) if evaluation_csv else None,
        "train_csv": str(Path(train_csv).resolve()) if train_csv else None,
        "sampling": "forged is stratified by document_family and tamper_type; genuine is stratified by document_family",
        "subsets": reports,
    }
    with (output / "protocol_report.json").open("w") as f:
        json.dump(protocol, f, indent=2)
    return protocol


def _parse_ratios(values):
    parsed = []
    for value in values:
        try:
            forged, genuine = (int(part) for part in value.split(":"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid ratio {value!r}; use forged:genuine") from exc
        parsed.append((forged, genuine))
    return parsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write read-only, deterministic imbalanced evaluation manifests.")
    ap.add_argument("--dataset", required=True, help="Canonical dataset directory (read-only).")
    ap.add_argument("--output", required=True, help="Directory for external subset manifests/reports.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", nargs="+", default=["1:1", "1:3", "1:5", "1:10"], help="Forged:genuine ratios.")
    ap.add_argument("--evaluation-csv", help="Existing evaluation/test split CSV; limits candidate samples.")
    ap.add_argument("--train-csv", help="Existing training split CSV; checked for record_id overlap.")
    args = ap.parse_args()
    protocol = write_protocol(args.dataset, args.output, _parse_ratios(args.ratios), args.seed,
                              evaluation_csv=args.evaluation_csv, train_csv=args.train_csv)
    for report in protocol["subsets"]:
        print(f"{report['ratio_forged_to_genuine']}: {report['forged_samples']} forged, "
              f"{report['genuine_samples']} genuine, {report['total_samples']} total; "
              f"categories={report['forged_category_counts']}")
