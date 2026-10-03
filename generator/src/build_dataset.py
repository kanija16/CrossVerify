"""
CrossVerify dataset builder.

ONE pipeline, driven entirely by family_config.py, that generates samples
for whichever document_family(s) are requested. No per-family branching in
the pipeline logic itself -- only in the config lookups.

Usage:
    python build_dataset.py --n_records 200 --families family_a family_b --out ../dataset --seed 42

Output layout:
    dataset/
        images/family_a/*.png
        images/family_b/*.png
        labels/*.json                (flat, filename encodes family+id+tamper_type)
        manifests/manifest.json      (all samples, all families)

IMPORTANT (data-leakage prevention): the "consistency_vector" written into
each label is computed from REAL OCR/QR extraction off the rendered image --
never from ground truth. "expected_consistency_vector" is the ground-truth
version, kept ONLY for evaluation/debugging and must never be fed to a
classifier as an input feature. See README.md section on leakage.
"""
import argparse
import json
import os
import random
import sys

from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from family_config import get_family, all_family_keys
from generate_records import generate_records
from render_card import render_card, get_informative_regions
from tamper import TAMPER_FUNCS, make_genuine, make_visual_splice_variants
from augment import copy_move_patch, realistic_noise
from extract import ocr_extract_fields, qr_decode_fields
from schema import validate_checksum, is_format_valid


def field_match_score(ocr_fields: dict, qr_fields: dict, config: dict):
    """Fuzzy match printed(OCR) fields against QR-decoded fields, generic
    over whichever fields the family config defines. Returns
    (score_or_None, qr_readable_bool). An unreadable QR is a distinct
    condition from "fields disagree" and must not be silently scored as a
    0.0 mismatch."""
    if not qr_fields or "_error" in qr_fields:
        return None, False

    scores = []
    for k in config["fields"]:
        ocr_val = str(ocr_fields.get(k, "")).strip()
        qr_val = str(qr_fields.get(k, "")).strip()
        if not ocr_val and not qr_val:
            continue
        scores.append(fuzz.token_sort_ratio(ocr_val, qr_val) / 100.0)
    score = sum(scores) / len(scores) if scores else None
    return score, True


def process_variant(record: dict, config: dict, template_variant: str, rng: random.Random,
                    used_splice_signatures=None):
    """Render, augment, extract, and compute the final consistency vector
    for one record variant (genuine or tampered). This function has zero
    family-specific branching -- everything family-specific comes from
    `config`."""
    img = render_card(
        record,
        config,
        field_overrides=record.get("render_field_overrides", {}),
        qr_payload_override=record.get("render_qr_payload"),
        template_variant=template_variant,
    )

    splice_signature = None
    splice_metadata = None
    if record.get("apply_visual_splice"):
        img, splice_signature, splice_metadata = copy_move_patch(
            img,
            get_informative_regions(config),
            rng=rng,
            excluded_signatures=used_splice_signatures,
        )

    img = realistic_noise(img, rng=rng)

    ocr_fields = ocr_extract_fields(img, config)
    qr_fields = qr_decode_fields(img)

    match_score, qr_readable = field_match_score(ocr_fields, qr_fields, config)
    id_field = config["identifier_field"]
    missing_field_count = sum(
        1 for field in config["fields"]
        if field != id_field and not ocr_fields.get(field, "").strip()
    )
    ocr_id = ocr_fields.get(id_field, "").strip()
    checksum_valid = validate_checksum(config, ocr_id) if ocr_id else False
    format_valid = is_format_valid(config, ocr_id) if ocr_id else False

    label = {
        "document_family": config["key"],
        "record_id": record["record_id"],
        "template_variant": template_variant,
        "tamper_type": record["tamper_type"],
        "identifier_field": id_field,
        "ground_truth_fields": {f: record[f] for f in config["fields"]},
        "ocr_extracted_fields": ocr_fields,
        "qr_decoded_fields": qr_fields,
        # --- computed from the image, safe to use as model input ---
        "consistency_vector": {
            "qr_readable": qr_readable,
            "text_qr_match_score": round(match_score, 3) if match_score is not None else None,
            "checksum_valid": checksum_valid,
            "format_valid": format_valid,
            "missing_field_count": missing_field_count,
        },
        # --- ground truth, evaluation/debugging ONLY, NEVER a model input ---
        "expected_consistency_vector": record["consistency_vector"],
        "cnn_label": record["cnn_label"],
        "final_label": record["final_label"],
        # Ground-truth localization/audit information only.  These values are
        # deliberately separate from consistency_vector and model labels.
        "splice_region": splice_metadata["splice_region"] if splice_metadata else None,
        "splice_variant": record.get("splice_variant") if splice_metadata else None,
        "splice_mask_seed": splice_metadata["splice_mask_seed"] if splice_metadata else None,
        "splice_mask_parameters": splice_metadata["splice_mask_parameters"] if splice_metadata else None,
        "splice_mask_sha256": splice_metadata["splice_mask_sha256"] if splice_metadata else None,
        # Audit metadata only; never part of consistency_vector/model input.
        "fine_grained_edit_metadata": record.get("fine_grained_edit_metadata"),
    }
    return img, label, splice_signature


def build_family(config: dict, n_records: int, out_dir: str, rng: random.Random):
    img_dir = os.path.join(out_dir, "images", config["key"])
    label_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    records = generate_records(config, n_records, seed=rng.randint(0, 2**31))

    manifest_entries = []
    idx = 0
    for base_record in records:
        variants = [make_genuine(base_record, config)]
        for fn in TAMPER_FUNCS:
            variants.append(fn(base_record, config, rng=rng))
        variants.extend(make_visual_splice_variants(base_record, config, rng=rng))

        template_variant = rng.choice(config["template_variants"])

        used_splice_signatures = set()
        for variant in variants:
            img, label, splice_signature = process_variant(
                variant, config, template_variant, rng, used_splice_signatures
            )
            if splice_signature is not None:
                used_splice_signatures.add(splice_signature)
            fname = f"{config['key']}_{idx:06d}_{variant['tamper_type']}"
            img.save(os.path.join(img_dir, fname + ".png"))
            with open(os.path.join(label_dir, fname + ".json"), "w") as f:
                json.dump(label, f, indent=2)
            manifest_entries.append({
                "id": fname,
                "document_family": config["key"],
                "record_id": label["record_id"],
                "template_variant": template_variant,
                "tamper_type": label["tamper_type"],
                "final_label": label["final_label"],
                "cnn_label": label["cnn_label"],
                "image_path": os.path.relpath(os.path.join(img_dir, fname + ".png"), out_dir),
                "label_path": os.path.relpath(os.path.join(label_dir, fname + ".json"), out_dir),
            })
            idx += 1

    return manifest_entries


def build(n_records: int, families: list, out_dir: str, seed: int = 42):
    rng = random.Random(seed)
    manifest_dir = os.path.join(out_dir, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    full_manifest = []
    for fam_key in families:
        config = get_family(fam_key)
        entries = build_family(config, n_records, out_dir, rng)
        full_manifest.extend(entries)
        print(f"Family {fam_key}: {len(entries)} images from {n_records} base records")

    with open(os.path.join(manifest_dir, "manifest.json"), "w") as f:
        json.dump(full_manifest, f, indent=2)

    print(f"Total: {len(full_manifest)} images -> {out_dir}")
    return full_manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_records", type=int, default=50,
                     help="base records PER FAMILY (each expands to 11 samples)")
    ap.add_argument("--families", nargs="+", default=all_family_keys(),
                     choices=all_family_keys())
    ap.add_argument("--out", type=str, default="../dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.n_records, args.families, args.out, args.seed)
