# CrossVerify: Multi-Family Fictional Document Forgery Benchmark

**Version B** — one centralized, configurable pipeline supporting two
fictional document families, extended from the Version A single-family
prototype (preserved at `../id-forgery-dataset-v1-backup/`).

## Research motivation

Real identity and registration documents contain sensitive personal
information and cannot simply be collected and redistributed as a research
benchmark. That makes it hard to build or evaluate forgery-detection systems
without either (a) using real people's real documents, which is both
unethical and typically illegal to collect at scale, or (b) training only on
whatever small, inconsistent set of forgery examples an organization has
internally.

CrossVerify sidesteps this by generating a **fully synthetic benchmark**
built from **fictional document schemes** — invented issuers, invented
identifier formats, invented checksum algorithms, invented QR payload
structures, invented visual layouts. Nothing here reproduces a real
government document's layout, identifiers, checksum algorithm, QR schema,
logos, or branding. This gives:

- controlled generation of genuine documents and specific, labeled
  inconsistency types
- known ground truth for every sample (down to which field was altered)
- multiple document families, so a detector's generality can be tested
  rather than its ability to memorize one layout
- controlled, reproducible attack conditions

**Research question this benchmark is built to support:**

> Can cross-modal consistency between textual, encoded, and structural
> information improve identity-document forgery detection across
> heterogeneous document types?

The fictional documents are an experimental benchmark, **not** replicas of
any real government document. This project does not claim, and this README
does not claim, that generating this dataset scientifically validates the
research question above — that requires the actual model training, baseline
comparisons, ablations, and statistical evaluation, which are a separate
later step (see "What's not done yet" below).

## Architecture

One pipeline, two document families, selected via configuration:

```
        FICTIONAL DOCUMENT
               |
      Document Family Config
     (family_config.py: fields,
      identifier rules, checksum
      algo, QR schema, layout)
               |
   +-----------+-----------+
   |           |           |
  OCR      QR Decoder   Visual Model
   |           |           |
   +-----------+-----------+
               |
   Validation / Consistency
   (schema.py + build_dataset.py,
    generic over config)
               |
        Feature Fusion
      (consistency vector,
       family-agnostic keys)
               |
          CrossVerify
               |
        Genuine / Forged
```

There is **one** `build_dataset.py`, **one** `tamper.py`, **one**
`extract.py`, **one** `render_card.py` (with a small per-family layout
registry inside it), and **one** `schema.py` (with a small per-family
checksum-algorithm registry inside it). Adding a third document family
means adding one entry to `family_config.py`, one field-value generator
function, and one layout function — nothing else in the pipeline changes.

## Document families

| | Family A | Family B |
|---|---|---|
| Concept | Fictional national identity card | Fictional business/entity registration certificate |
| Issuer | "Fictional Identity Authority" | "Fictional Revenue & Registry Bureau" |
| Identifier | 12-digit, Verhoeff checksum | 10-digit, weighted mod-11 checksum (ISBN-10-style) |
| Fields | name, dob, gender, id_number, address | full_name, registry_id, entity_type, jurisdiction_code, registration_date |
| Layout | portrait card, photo box, bottom-right QR | landscape ledger/table, no photo, top-right QR |
| Visual grammar | blue/purple header, card-style | green/olive header, table-row style |

Both checksum algorithms are standard, generic, publicly documented
check-digit techniques (Verhoeff's algorithm; a positional-weighted mod-11
scheme in the same family as ISBN-10) applied with our own constants and
field layout — not any real government ID's actual algorithm.

## Tamper categories (identical set, both families)

All eight tamper categories apply directly to both families with no special-casing
needed, since each operates on concepts every family config defines
(an identifier field, a checksum rule, a format rule, other printed fields,
and a QR payload):

| Type | What's altered | Which module should catch it |
|---|---|---|
| `genuine` | nothing | — |
| `text_qr_mismatch` | a printed non-identifier field edited, QR keeps old value | consistency vector (text/QR match score) |
| `qr_only_mismatch` | QR payload edited, printed fields keep old value | consistency vector |
| `checksum_invalid` | identifier's check digit corrupted | checksum validator |
| `format_invalid` | identifier malformed (length / leading-digit rule) | format validator |
| `field_missing` | a required non-identifier field blanked on the printed card | field-presence flags |
| `fine_grained_edit` | one character in a configured printed non-identifier field changed; QR keeps the original | consistency vector (near-miss text/QR conflict) |
| `coordinated_full_forgery` | every field and QR payload replaced consistently with a fictional forged record | final-label hard case; no intentional visual artifact |
| `visual_splice` | pure image-level copy-move patch; text/QR stay fully consistent | **CNN module only** — the text/QR pipeline structurally cannot catch this |

## Data-leakage prevention

This is enforced structurally, not just by convention:

- `build_dataset.py`'s `consistency_vector` is computed **only** from real
  OCR output and real QR-decode output run against the rendered (and
  noise-augmented) image — never from the record's ground-truth fields.
- The record's ground-truth-derived vector is stored separately as
  `expected_consistency_vector`, alongside `tamper_type`, `cnn_label`, and
  `final_label` — these exist for evaluation/debugging and **must never be
  used as classifier input features**. Nothing in the repo currently trains
  a classifier, so there's no code path where this could happen by
  accident, but this rule must hold when that code is written.
- Every sample generated from render → augment → OCR/QR-extract →
  compute-consistency happens in `process_variant()` in `build_dataset.py`,
  which only receives the image and the family config at the point it
  computes features — ground truth is attached to the label afterward, not
  fed forward into feature computation.

## Splits and identity-level separation

**Every split groups by `record_id` before splitting** (`make_splits.py`),
never by shuffling individual sample rows. Each synthetic identity's 11
variants (genuine + 7 non-splice tamper types + 3 visual-splice variants)
always stay together as a unit unless a
split's specific design calls for separating a tamper type from its
identity (see `unseen_tamper` below) — and even there, the held-out
identities' *other* variants are excluded from train/val entirely, not
scattered into it.

- **`random`** — 70/15/15 train/val/test, identity-level, across whichever
  families are included.
- **`unseen_tamper`** — a fraction of identities are held out entirely;
  their `--holdout_type` variant plus a genuine control goes to test. The
  remaining identities go to train/val with that tamper type excluded
  entirely (the model never sees that forgery pattern, on any identity,
  during training). No identity appears in both train/val and test.
- **`cross_family`** — three experiments in one call:
  - `train_a_test_b`: train on Family A, test on Family B
  - `train_b_test_a`: train on Family B, test on Family A
  - `train_both`: train on both families combined, test on identities held
    out from both (identity-level, not seen in `train_both`'s train set)
- **`unseen_template`** — train on `template_variant="v1"` only, test on
  `"v2"` only, per family. Tests generalization to an unseen visual
  skin/layout variant without changing any forgery logic.

`check_dataset.py` runs an automated leakage check across every split mode
present (including the nested `cross_family` experiment subfolders) and
reports any `record_id` that appears in more than one split within the same
experiment. All four split modes currently pass this check with zero
overlaps (see validation run below).

## Unseen-tamper evaluation protocol

Unseen-tamper evaluation asks whether a future model can detect a forgery
category that was wholly withheld during training. It is a split-validation
protocol only: an implemented protocol is **not** a claim of generalization
performance, which will be measured only after model training.

```bash
python src/unseen_tamper_protocol.py --dataset dataset_final \
  --output unseen_tamper_protocol --seed 42
```

For each supported forged category, the protocol deterministically holds out
identities per family. Test receives only each held-out identity's genuine
sample and the held-out forged category; train receives all variants of every
other identity except that category. It verifies disjoint record/image IDs,
exact image hashes, category absence from train, genuine samples on both
sides, seed reproducibility, and canonical-dataset tree fingerprints.

Generated files are audit manifests and reports, not model-facing data.
Inputs remain restricted to the image-derived `consistency_vector` allowlist;
category, labels, IDs, paths, ground truth, and debug metadata are prohibited.
If a canonical category is absent (currently expected for
`coordinated_full_forgery` or `fine_grained_edit`), the report states
`NOT AVAILABLE IN CURRENT CANONICAL DATASET` and no split is fabricated.

## Baselines this dataset is intended to support

The dataset/label structure is designed to be compatible with:

1. Visual CNN baseline (image → genuine/forged, no text/QR features)
2. ELA + CNN baseline
3. ViT / document-forgery transformer baseline
4. Consistency-only baseline (just the `consistency_vector` fields, no image)
5. CrossVerify fusion model (consistency vector + CNN forensic score)

**No performance numbers are generated or claimed by this repo.** Model
training, baseline comparison, ablations, and statistical evaluation are a
separate, later step — see "What's not done yet."

## Directory structure

```
dataset/
    images/
        family_a/*.png
        family_b/*.png
    labels/*.json                      # one per sample, full metadata
    manifests/manifest.json            # flat index of every sample
    splits/
        random/{train,val,test}.csv
        unseen_tamper/{train,val,test}.csv
        cross_family/
            train_a_test_b/{train,test}.csv
            train_b_test_a/{train,test}.csv
            train_both/{train,test}.csv
        unseen_template/{train,test}.csv
```

Each label JSON contains: `document_family`, `record_id`,
`template_variant`, `tamper_type`, `identifier_field`,
`ground_truth_fields`, `ocr_extracted_fields` (real OCR output, with real
OCR noise), `qr_decoded_fields` (real QR-decode output), the computed
`consistency_vector` (image-derived, safe as model input), the
`expected_consistency_vector` (ground-truth, evaluation-only), and both a
`cnn_label` and `final_label` — these intentionally diverge for several
tamper types (e.g. `checksum_invalid` looks visually clean but fails
consistency checks), which is what makes a fusion classifier worth having
over either signal alone.

## Usage

```bash
pip install -r requirements.txt
cd src

# Build both families (n_records base identities PER family; each expands
# to 11 samples: 8 non-splice categories plus 3 visual-splice variants).
# OCR is the bottleneck — budget roughly 0.3-0.5s/image.
python build_dataset.py --n_records 200 --families family_a family_b --out ../dataset --seed 42

# All four split modes in one call
python make_splits.py --dataset ../dataset --mode all --holdout_type visual_splice

# QA report: per-family/tamper-type counts, checksum/QR-read rates,
# OCR failure rate, and identity-leakage check across every split
python check_dataset.py --dataset ../dataset
```

## Imbalance evaluation protocol

Real screening settings commonly contain far more genuine documents than
forged ones.  Accuracy alone can therefore be misleading: a system that
always predicts "genuine" can look strong when forgeries are rare while
failing at the actual detection task.  CrossVerify provides a deterministic,
read-only protocol for forged:genuine ratios of **1:1, 1:3, 1:5, and 1:10**.

The protocol writes only external JSON subset manifests; it never copies,
edits, or regenerates the canonical images and labels.  Give it existing
identity-level train and test CSVs so it first rejects any overlapping
`record_id`.  It then samples only from the test candidates, stratifying
forged samples by `(document_family, tamper_type)` and genuine samples by
family.  This avoids accidental domination by the more numerous
`visual_splice` variants while retaining coverage across forged categories
when the requested size permits it.

```bash
# Run from the repository root. Output is outside the canonical dataset.
python src/imbalance_protocol.py \
  --dataset dataset_final \
  --evaluation-csv dataset_final/splits/random/test.csv \
  --train-csv dataset_final/splits/random/train.csv \
  --output ../imbalance_protocol_pilot \
  --seed 42
```

Each external subset manifest contains the selected canonical manifest rows;
`protocol_report.json` records the exact ratio, seed, sample and identity
counts, per-category forged counts, and a canonical-dataset file-count and
manifest-hash snapshot.  Repeating the same dataset, configuration, ratio,
and seed yields the same IDs; a different seed selects another valid subset.

This protocol creates no model metrics.  Future model evaluations at every
ratio should report precision, recall, F1, balanced accuracy, ROC-AUC where
appropriate, and especially PR-AUC / average precision for severe imbalance.

## Validation run (n_records=20 per family, seed=42)

Ran the full pipeline end-to-end on a small sample before calling Version B
complete, per the checklist this build was scoped against:

- **Family A documents:** 180 (20 identities × 9 samples)
- **Family B documents:** 180 (20 identities × 9 samples)
- **Genuine samples:** 40 (20 per family)
- **Samples per non-splice tamper type:** 40 each (20 per family); **visual
  splice samples:** 120 (three per identity). All categories are balanced by
  construction at their intended rates.
- **OCR identifier-field read failure rate:** 3.3% (12/360) — real Tesseract
  noise, not simulated; occasional full misses are expected and are exactly
  the kind of noise the fuzzy-match consistency module needs to be robust
  to
- **QR decode success rate:** 70–100% for Family A, 55–90% for Family B,
  varying by tamper type. Family B runs consistently a bit lower — its QR
  payload is denser (more/longer field names → higher QR version → smaller
  effective module size after the shared augmentation pipeline's blur/JPEG
  step) even after sizing the QR render dynamically off payload density.
  This is a real, currently-unresolved characteristic of Family B rather
  than a bug in the harness, and should be accounted for when comparing
  the two families' downstream results (e.g. don't expect QR-uptime parity
  between families) or optionally revisited if it needs tightening.
- **Validation success (checksum/format):** 0% checksum-valid on
  `checksum_invalid`/`format_invalid` samples (correctly rejected), ~90-100%
  on all other categories (matches expectation; the small shortfall from
  100% is a byproduct of occasional OCR misreads on the identifier digits
  themselves, which is realistic behavior)
- **Consistency feature availability:** `text_qr_match_score` is `null`
  exactly when `qr_readable=False`, and populated otherwise — verified no
  case where an unreadable QR was silently scored as a 0.0 mismatch
- **Train/test identity leakage check:** zero overlaps across all four
  split modes (`random`, `unseen_tamper`, `cross_family` × 3 experiments,
  `unseen_template`) — see `check_dataset.py` output above
- **Cross-family splits:** confirmed both single-direction experiments
  (`train_a_test_b`, `train_b_test_a`) and the combined `train_both`
  experiment produce correctly partitioned, non-overlapping sample sets
- **Unseen-tamper split:** confirmed held-out identities appear only via
  their `visual_splice` variant in test, and are entirely absent from
  train/val's other 6 variants
- Representative images from both families were visually inspected
  (`render_card.py` output) — layouts are clearly distinct from each other
  and are original designs, not modeled on any real document

## What's not done yet

Per the original scope, this delivers the **dataset + extraction +
consistency pipeline**, validated as above. The following are explicitly
out of scope for this deliverable and have not been built or run:

- The CNN forensic module, the fusion classifier, and any of the five
  listed baselines
- Any training run, performance number, ablation, or statistical comparison
- Unseen-template generalization is supported structurally (the split
  exists and passes leakage checks) but has not been evaluated with an
  actual model
- `identifier_min_first_digit` and the two checksum algorithms are
  reasonable generic choices but have not been stress-tested against
  adversarial identifier-construction attempts beyond what `tamper.py`
  already generates

Building the dataset generator working correctly is not the same claim as
the research method being validated — that requires the model
training/evaluation step that comes next.
