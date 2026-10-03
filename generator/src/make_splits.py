"""
Create splits from a CrossVerify manifest.

All splits are IDENTITY-LEVEL: every variant of the same synthetic record
(genuine + its 6 tampered siblings) shares the same record_id and is always
kept together in one split. This is enforced by grouping on record_id
before splitting, never by shuffling individual samples.

Modes:
  random          70/15/15 train/val/test, identity-level, within whichever
                  --families were requested (defaults to all).
  unseen_tamper   One tamper category (--holdout_type) is entirely excluded
                  from train/val and appears only in test, to measure
                  generalization to an unseen forgery pattern.
  cross_family    Three experiments in one call:
                    train_a_test_b: train on family_a, test on family_b
                    train_b_test_a: train on family_b, test on family_a
                    train_both:     train on both, test on held-out identities
                                    from both (identity-level held out, not
                                    seen in train_both's train set)
  unseen_template Train on template_variant "v1" only, test on "v2" only,
                  per family -- tests generalization to an unseen visual
                  template/skin without changing the forgery logic.

Output: dataset/splits/<mode>/<split_name>.csv, columns:
  id, document_family, record_id, template_variant, tamper_type, final_label, cnn_label
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict


def load_manifest(dataset_dir):
    with open(os.path.join(dataset_dir, "manifests", "manifest.json")) as f:
        return json.load(f)


def _group_by_record(manifest):
    groups = defaultdict(list)
    for m in manifest:
        groups[m["record_id"]].append(m)
    return groups


def _write_csv(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["id", "document_family", "record_id", "template_variant", "tamper_type", "final_label", "cnn_label"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in items:
            w.writerow({k: m[k] for k in fields})


def random_split(manifest, families=None, seed=42, ratios=(0.7, 0.15, 0.15)):
    if families:
        manifest = [m for m in manifest if m["document_family"] in families]
    groups = _group_by_record(manifest)
    record_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(record_ids)

    n = len(record_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_ids = set(record_ids[:n_train])
    val_ids = set(record_ids[n_train:n_train + n_val])
    test_ids = set(record_ids[n_train + n_val:])

    def flatten(ids):
        out = []
        for rid in ids:
            out.extend(groups[rid])
        return out

    return {"train": flatten(train_ids), "val": flatten(val_ids), "test": flatten(test_ids)}


def unseen_tamper_split(manifest, holdout_type, families=None, seed=42, val_frac=0.15, test_identity_frac=0.2):
    """Identity-level AND tamper-level separation:
      - a set of identities is held out entirely for test
      - test only contains their holdout_type variant (the unseen forgery pattern)
      - train/val use the remaining identities, with the holdout_type variant
        excluded from those identities too (so the model never sees that
        tamper type during training, on ANY identity)
    This avoids the same synthetic identity appearing in both train and
    test via a different tamper variant, which would leak identity-specific
    cues (exact name spelling, address format, etc.) across the split."""
    if families:
        manifest = [m for m in manifest if m["document_family"] in families]

    groups = _group_by_record(manifest)
    record_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(record_ids)

    n_test_identities = max(1, int(len(record_ids) * test_identity_frac))
    test_ids = set(record_ids[:n_test_identities])
    remaining_ids = record_ids[n_test_identities:]
    n_val = int(len(remaining_ids) * val_frac)
    val_ids = set(remaining_ids[:n_val])
    train_ids = set(remaining_ids[n_val:])

    train = [m for rid in train_ids for m in groups[rid] if m["tamper_type"] != holdout_type]
    val = [m for rid in val_ids for m in groups[rid] if m["tamper_type"] != holdout_type]
    # Held-out identities contribute a genuine control plus their only forged
    # test category; their other variants never enter train/val.
    test = [m for rid in test_ids for m in groups[rid]
            if m["tamper_type"] in {"genuine", holdout_type}]

    return {"train": train, "val": val, "test": test}


def cross_family_split(manifest, seed=42, holdout_frac=0.2):
    fam_a = [m for m in manifest if m["document_family"] == "family_a"]
    fam_b = [m for m in manifest if m["document_family"] == "family_b"]

    # Experiment 1 & 2: full cross-family (no identity overlap possible,
    # different families never share record_id namespaces).
    exp1 = {"train": fam_a, "test": fam_b}          # train_a_test_b
    exp2 = {"train": fam_b, "test": fam_a}          # train_b_test_a

    # Experiment 3: combined train, held-out identities from BOTH families.
    groups_a = _group_by_record(fam_a)
    groups_b = _group_by_record(fam_b)
    rng = random.Random(seed)

    def held_out_split(groups):
        ids = list(groups.keys())
        rng.shuffle(ids)
        n_hold = max(1, int(len(ids) * holdout_frac))
        hold_ids = set(ids[:n_hold])
        train_ids = set(ids[n_hold:])
        train = [m for rid in train_ids for m in groups[rid]]
        test = [m for rid in hold_ids for m in groups[rid]]
        return train, test

    train_a, test_a = held_out_split(groups_a)
    train_b, test_b = held_out_split(groups_b)
    exp3 = {"train": train_a + train_b, "test": test_a + test_b}

    return {
        "train_a_test_b": exp1,
        "train_b_test_a": exp2,
        "train_both": exp3,
    }


def unseen_template_split(manifest, families=None):
    if families:
        manifest = [m for m in manifest if m["document_family"] in families]
    train = [m for m in manifest if m["template_variant"] == "v1"]
    test = [m for m in manifest if m["template_variant"] == "v2"]
    return {"train": train, "test": test}


def write_splits(splits, out_dir, mode):
    mode_dir = os.path.join(out_dir, "splits", mode)
    for name, items in splits.items():
        path = os.path.join(mode_dir, f"{name}.csv")
        _write_csv(items, path)
        print(f"[{mode}] {name}: {len(items)} samples -> {path}")


def write_cross_family(experiments, out_dir):
    mode_dir = os.path.join(out_dir, "splits", "cross_family")
    for exp_name, split in experiments.items():
        for split_name, items in split.items():
            path = os.path.join(mode_dir, exp_name, f"{split_name}.csv")
            _write_csv(items, path)
            print(f"[cross_family/{exp_name}] {split_name}: {len(items)} samples -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="../dataset")
    ap.add_argument("--mode", choices=["random", "unseen_tamper", "cross_family", "unseen_template", "all"],
                     default="all")
    ap.add_argument("--holdout_type", type=str, default="visual_splice")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = load_manifest(args.dataset)
    modes = ["random", "unseen_tamper", "cross_family", "unseen_template"] if args.mode == "all" else [args.mode]

    if "random" in modes:
        splits = random_split(manifest, families=args.families, seed=args.seed)
        write_splits(splits, args.dataset, "random")

    if "unseen_tamper" in modes:
        splits = unseen_tamper_split(manifest, args.holdout_type, families=args.families, seed=args.seed)
        write_splits(splits, args.dataset, "unseen_tamper")

    if "cross_family" in modes:
        experiments = cross_family_split(manifest, seed=args.seed)
        write_cross_family(experiments, args.dataset)

    if "unseen_template" in modes:
        splits = unseen_template_split(manifest, families=args.families)
        write_splits(splits, args.dataset, "unseen_template")
