# Building the CrossVerify dataset

The generator is deterministic: the same seed, in the same environment, produces
byte-identical images and labels. (Verified: two builds with `--seed 42` gave identical
SHA-256 hashes for all images; `--seed 43` gave different images.)

## Environment

- Python 3.10, macOS (the released dataset was generated on macOS)
- `pip install -r ../requirements/generator.txt`
- System packages: Tesseract OCR, zbar

Rendering uses system fonts (Arial on macOS, DejaVu Sans on Linux; see `_font()` in
`src/render_card.py`). Building on a different OS, or with different library versions,
produces visually equivalent but not byte-identical images. **The released dataset
archive is the reference copy.**

## Command used for the released dataset

```bash
cd generator/src
python build_dataset.py --n_records 500 --families family_a family_b --seed 42 --out ../../dataset
```

Confirmed by the dataset author: the released dataset (`dataset_final_v2`) was built with
`--n_records 500 --seed 42`. `final_check.log` confirms 500 base records per family
(11,000 images).

This produces `images/`, `labels/` and `manifests/manifest.json`.

## Splits

The canonical split used in the paper is in `data/splits/` (`train.csv`, `val.csv`,
`test.csv`, `record_id_split_map.csv`). It is identity-grouped and **stratified by
document family**: 350 / 75 / 75 identities per family (7,700 / 1,650 / 1,650 documents).

`src/make_splits.py --mode random --seed 42` is the generator's splitting routine, but it
pools both families before shuffling, so it produces the same sizes with a different
assignment of identities. **The released `data/splits/` files are the reference split.**
These are byte-for-byte the files the visual branch (M2) was trained on, and the same
files are used by M3, the fusion model and the end-to-end baseline.

> Note: the script that produced the family-stratified assignment is not included.
> Use the released `data/splits/` files (also shipped inside the Zenodo archive) to
> reproduce the paper's results.

## Reproducibility check

Regenerating with `--n_records 500 --seed 42` (Linux, Faker 40.40.0) reproduces all
1,000 `record_id`s in the same order, with the same template variants, as the released
dataset. Image pixels differ across operating systems because of fonts (see above).

## Quality audit

`final_check.log` is the post-build QA report: per-family/tamper counts, checksum
validity, QR readability (pyzbar) and OCR identifier recovery per tamper type,
variant-count integrity and duplicate-image check. Repeated zbar decoder warnings
were removed from the log for readability.
