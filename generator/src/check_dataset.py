"""
Sanity/QA checks over a CrossVerify-built dataset:
  - per-family, per-tamper-type counts
  - checksum_valid / qr_readable rate per tamper type (does it match expectation?)
  - OCR field-empty rate
  - identity-level (record_id) leakage check across any splits present
"""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict

from PIL import Image
from rapidfuzz import fuzz

from family_config import get_family


def _get_live_qr_decoder():
    """Return pyzbar's decoder, or a reason why a live audit cannot run.

    Labels retain the extraction result captured during generation.  That is
    useful provenance, but it must not be mistaken for a fresh QR-readability
    test when the generator ran without libzbar installed.
    """
    try:
        from pyzbar.pyzbar import decode
        return decode, None
    except ImportError as exc:
        return None, str(exc)


def audit_live_qr_fields(dataset_dir, manifest):
    """Decode each rendered image directly for QA without changing labels.

    Returns ``None, reason`` if the local pyzbar/libzbar runtime is absent;
    otherwise returns an ID -> parsed-fields-or-None mapping.  A QR is counted
    as readable as soon as pyzbar finds a QR symbol.  JSON parsing is reported
    separately because payload-format validity is a different property.
    """
    decoder, reason = _get_live_qr_decoder()
    if decoder is None:
        return None, reason

    results = {}
    for entry in manifest:
        image_path = os.path.join(dataset_dir, entry["image_path"])
        try:
            with Image.open(image_path) as img:
                symbols = decoder(img)
        except Exception as exc:  # Report an image/decoder fault as unreadable.
            results[entry["id"]] = {"readable": False, "fields": None, "error": str(exc)}
            continue

        qr_symbols = [symbol for symbol in symbols if symbol.type == "QRCODE"]
        if not qr_symbols:
            results[entry["id"]] = {"readable": False, "fields": None, "error": None}
            continue
        try:
            fields = json.loads(qr_symbols[0].data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            fields = None
        results[entry["id"]] = {"readable": True, "fields": fields, "error": None}
    return results, None


def _field_match_score(ocr_fields, qr_fields, config):
    if not qr_fields:
        return None
    scores = []
    for field in config["fields"]:
        ocr_value = str(ocr_fields.get(field, "")).strip()
        qr_value = str(qr_fields.get(field, "")).strip()
        if ocr_value or qr_value:
            scores.append(fuzz.token_sort_ratio(ocr_value, qr_value) / 100.0)
    return sum(scores) / len(scores) if scores else None


def load_manifest(dataset_dir):
    with open(os.path.join(dataset_dir, "manifests", "manifest.json")) as f:
        return json.load(f)


def load_all_labels(dataset_dir, manifest):
    labels = []
    for entry in manifest:
        with open(os.path.join(dataset_dir, entry["label_path"])) as f:
            labels.append(json.load(f))
    return labels


def report_counts(manifest):
    by_family_type = defaultdict(int)
    for m in manifest:
        by_family_type[(m["document_family"], m["tamper_type"])] += 1

    print("=== Sample counts (family x tamper_type) ===")
    families = sorted({k[0] for k in by_family_type})
    for fam in families:
        print(f"\n{fam}:")
        total = 0
        for (f, t), n in sorted(by_family_type.items()):
            if f == fam:
                print(f"  {t:22s} {n}")
                total += n
        print(f"  {'TOTAL':22s} {total}")


def report_signal_quality(dataset_dir, manifest, labels):
    """Report image-derived signals, using a fresh QR audit when possible."""
    live_qr, unavailable_reason = audit_live_qr_fields(dataset_dir, manifest)
    by_key = defaultdict(list)
    for entry, label in zip(manifest, labels):
        by_key[(label["document_family"], label["tamper_type"])].append((entry, label))

    print("\n=== Signal quality (family x tamper_type) ===")
    if unavailable_reason:
        print(f"Live QR audit unavailable (pyzbar/libzbar): {unavailable_reason}")
    print(f"{'family':10s} {'tamper_type':20s} {'n':>4s} {'checksum_valid':>15s} {'qr_readable':>12s} {'ocr_id':>8s} {'avg_match':>10s}")
    for (fam, t), items in sorted(by_key.items()):
        n = len(items)
        checksum_rate = sum(1 for _, i in items if i["consistency_vector"]["checksum_valid"]) / n
        ocr_id_rate = sum(
            1 for _, i in items
            if i["ocr_extracted_fields"].get(i["identifier_field"], "").strip()
        ) / n
        if live_qr is None:
            qr_rate_text = "n/a"
            scores = []
        else:
            audit_items = [live_qr[entry["id"]] for entry, _ in items]
            qr_rate_text = f"{sum(item['readable'] for item in audit_items) / n:.2f}"
            config = get_family(fam)
            scores = [
                score for (_, label), audit in zip(items, audit_items)
                if (score := _field_match_score(label["ocr_extracted_fields"], audit["fields"], config)) is not None
            ]
        avg_score = sum(scores) / len(scores) if scores else float("nan")
        print(f"{fam:10s} {t:20s} {n:4d} {checksum_rate:15.2f} {qr_rate_text:>12s} {ocr_id_rate:8.2f} {avg_score:10.2f}")

    empty_id_ocr = sum(
        1 for l in labels
        if not l["ocr_extracted_fields"].get(l["identifier_field"], "").strip()
    )
    print(f"\nSamples with empty OCR identifier field: {empty_id_ocr}/{len(labels)} "
          f"({empty_id_ocr/len(labels):.1%})")


def report_variant_contracts(dataset_dir, manifest, labels):
    """QA-only checks for the fixed per-identity variant contract."""
    by_record = defaultdict(list)
    for label in labels:
        by_record[label["record_id"]].append(label)

    record_sizes = defaultdict(int)
    splice_counts = defaultdict(int)
    for record_id, items in by_record.items():
        record_sizes[len(items)] += 1
        splice_counts[sum(i["tamper_type"] == "visual_splice" for i in items)] += 1

    format_invalid = [i for i in labels if i["tamper_type"] == "format_invalid"]
    format_length_ok = sum(
        len(i["ground_truth_fields"][i["identifier_field"]])
        == get_family(i["document_family"])["identifier_length"]
        for i in format_invalid
    )

    hashes = defaultdict(list)
    for entry in manifest:
        image_path = os.path.join(dataset_dir, entry["image_path"])
        with open(image_path, "rb") as f:
            hashes[hashlib.sha256(f.read()).hexdigest()].append(entry["id"])
    duplicate_groups = [ids for ids in hashes.values() if len(ids) > 1]

    print("\n=== Variant and image integrity ===")
    print(f"Record variant counts: {dict(sorted(record_sizes.items()))}")
    print(f"Visual-splice counts per record: {dict(sorted(splice_counts.items()))}")
    print(f"Length-preserving format_invalid: {format_length_ok}/{len(format_invalid)}")
    print(f"Duplicate image groups: {len(duplicate_groups)}")


def _read_record_ids(csv_path):
    ids = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ids.add(row["record_id"])
    return ids


def check_split_leakage(dataset_dir):
    """For every leaf directory under splits/ containing .csv files, verify
    no record_id appears in more than one of those files. Each leaf
    directory (e.g. splits/random/, splits/cross_family/train_both/) is
    checked independently -- files in different experiments aren't compared
    against each other since they represent different train/test contracts."""
    splits_dir = os.path.join(dataset_dir, "splits")
    if not os.path.isdir(splits_dir):
        print("\nNo splits/ directory found yet -- skipping leakage check.")
        return

    print("\n=== Identity-level leakage check ===")
    for root, dirs, files in os.walk(splits_dir):
        csvs = [f for f in files if f.endswith(".csv")]
        if not csvs:
            continue
        rel = os.path.relpath(root, splits_dir)
        record_ids_by_split = {f.replace(".csv", ""): _read_record_ids(os.path.join(root, f)) for f in csvs}

        splits = list(record_ids_by_split.items())
        leaks_found = False
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                name_i, ids_i = splits[i]
                name_j, ids_j = splits[j]
                overlap = ids_i & ids_j
                if overlap:
                    leaks_found = True
                    print(f"  [{rel}] LEAK: {len(overlap)} record_ids shared between "
                          f"{name_i} and {name_j}")
        if not leaks_found:
            print(f"  [{rel}] OK -- no record_id overlap across {list(record_ids_by_split.keys())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="../dataset")
    args = ap.parse_args()

    manifest = load_manifest(args.dataset)
    labels = load_all_labels(args.dataset, manifest)

    report_counts(manifest)
    report_signal_quality(args.dataset, manifest, labels)
    report_variant_contracts(args.dataset, manifest, labels)
    check_split_leakage(args.dataset)
