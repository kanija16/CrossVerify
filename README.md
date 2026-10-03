# CrossVerify

Cross-modal identity-document forgery detection: an image-only CNN branch (visual-splice
detector) and a cross-modal OCR / QR / checksum consistency branch (M3), combined by a
leakage-controlled stacked logistic-regression fusion model. Evaluated on a synthetic,
identity-grouped benchmark of 11,000 documents (1,000 fictional identities × 11 variants,
two document families, eight forgery mechanisms).

> Paper: *CrossVerify* (IEEE Access, under review). Citation to be added.

## Repository layout

| Path | Contents |
|---|---|
| `generator/` | Synthetic dataset generator (rendering, forgery mechanisms, augmentation, QA). See `generator/BUILD.md` |
| `data/splits/` | Canonical identity-grouped 70/15/15 split (`train.csv`, `val.csv`, `test.csv`) |
| `src/baseline/` | Visual branch (ResNet-18). `train_cnn.py`, `train_cnn_oof.py`, `eval_cnn_test.py`; end-to-end baseline `train_cnn_e2e.py` |
| `src/m3_crossmodal/` | Cross-modal branch: live OCR/QR extraction, consistency features, Random Forest |
| `src/m3_crossmodal/fusion/` | Out-of-fold stacking, fusion training, threshold selection, locked test evaluation |
| `src/m3_crossmodal/forensics/` | Forensic investigations for coordinated full forgery (paper Section IX) |
| `src/evaluation/` | Metrics and bootstrap confidence intervals |
| `src/analysis/` | Revision analyses (McNemar, prevalence, decoder comparison, etc.) |
| `outputs/features/` | Extracted M3 features for all splits |
| `outputs/fusion/` | Out-of-fold predictions, fusion models, validation and test predictions |
| `models/cnn/` | CNN validation and test predictions (see note on CNN weights below) |
| `requirements/` | Pinned environments per component |

## Dataset

The 11,000 images, label JSONs, manifest and the canonical split are released separately on
**Zenodo: [10.5281/zenodo.23136723](https://doi.org/10.5281/zenodo.23136723)** (CC BY 4.0).

The archive is split into parts because of its size. Download all `crossverify_dataset_v1.zip.part_*`
files into one folder, join them and unzip:

```bash
cat crossverify_dataset_v1.zip.part_* > crossverify_dataset_v1.zip   # Windows: copy /b crossverify_dataset_v1.zip.part_* crossverify_dataset_v1.zip
unzip crossverify_dataset_v1.zip                                      # creates dataset_crossverify/
export DATASET_ROOT=$PWD/dataset_crossverify                          # or rename the folder to ./dataset
```

The MD5 of the joined zip is listed on the Zenodo record. Layout:

```
dataset_crossverify/
  README.txt
  images/family_a/*.png      (5,500)
  images/family_b/*.png      (5,500)
  labels/*.json              (11,000)
  manifests/manifest.json
  splits/{train,val,test}.csv, record_id_split_map.csv   (same files as data/splits/ here)
```

The dataset is fully synthetic. Identities are generated with Faker; identifiers are random
numbers that satisfy each family's format and check-digit rule (Family A: 12 digits,
Verhoeff check digit; Family B: 10 digits, weighted mod-11). They are not derived from any real
person's data. To regenerate the dataset instead, see `generator/BUILD.md`.

## Reproducing the paper's numbers (no images needed)

Every reported result can be recomputed from the prediction and feature files in this repo.
Run from the repository root after `pip install -r requirements/fusion.txt`.

**A. Evaluate the frozen models** (checks artifact hashes, then scores the test split):

```bash
# Primary results: leakage-controlled (out-of-fold) fusion
python -m src.m3_crossmodal.fusion.run_corrected_final_test_evaluation

# Ablation: naive in-sample stacking baseline
python -m src.m3_crossmodal.fusion.run_final_test_evaluation

# Bootstrap 95% confidence intervals (CNN, M3, fusion)
python -m src.evaluation.bootstrap_ci

# Revision analyses (McNemar, coordinated-forgery chance test, read failures,
# simple fusion baselines, prevalence-adjusted precision, Brier)
python -m src.analysis.crossverify_audit

# End-to-end CNN baseline: per-mechanism table (paper Section VIII), from the saved predictions
python -m src.analysis.e2e_per_type
```

Expected: fusion `σ(3.536043 · pCNN + 6.615993 · pM3 − 1.469315) ≥ 0.82`,
test ROC-AUC 0.9388, F1 0.9336.

**B. Refit the stacking stage from scratch.** These scripts rebuild the models and
overwrite the frozen artifacts with numerically identical ones (same weights and
thresholds). The re-saved `.joblib` files are not byte-identical, so the hash checks in A
fail afterwards: run B in a separate clone, or `git checkout -- models outputs` after it.

```bash
# M3 out-of-fold predictions on the training split (reproduces all 7,700 values exactly)
python -m src.m3_crossmodal.fusion.generate_m3_oof

# Leakage-controlled fusion: fit on out-of-fold predictions, select threshold on validation
python -m src.m3_crossmodal.fusion.train_corrected_fusion

# Naive in-sample stacking baseline (weights 3.470976, 7.209759, −1.808732; threshold 0.78)
python -m src.m3_crossmodal.fusion.train_and_freeze_fusion
```

## Full pipeline (needs the dataset images)

```bash
export DATASET_ROOT=$PWD/dataset

# Visual branch (requirements/cnn.txt)
python -m src.baseline.train_cnn          # deployment CNN, target cnn_label (visual splice)
python -m src.baseline.train_cnn_oof      # 5 out-of-fold CNNs
python -m src.baseline.eval_cnn_test

# End-to-end image-only baseline, target final_label (genuine vs all forgeries)
python -m src.baseline.train_cnn_e2e               # 15-epoch budget -> outputs/baselines/cnn_e2e/
python -m src.baseline.train_cnn_e2e --epochs 30   # 30-epoch budget (paper headline) -> outputs/baselines/cnn_e2e_ep30/
```

The cross-modal feature extraction (`src/m3_crossmodal/run_part4.py` and related scripts)
re-runs Tesseract OCR and OpenCV QR decoding on every image; the extracted features used in the
paper are provided in `outputs/features/`.

## Notes

- **CNN weights.** The trained weights of the deployment CNN and the five out-of-fold CNNs
  were not retained. All CNN-derived results are computed from the released prediction files
  (`models/cnn/*_predictions.csv`, `outputs/fusion/oof/cnn_oof_predictions.csv`). Retraining
  with the provided code and seed gives close but not bit-identical weights.
- **QR decoding.** M3 decodes QR codes with OpenCV's `QRCodeDetector`; the generator used pyzbar.
  `python -m src.analysis.qr_decoder_comparison --split test` compares the two on the images.
- **scikit-learn versions.** The frozen M3 models were saved with scikit-learn 1.6.1.
  `src/m3_crossmodal/model_io.load_m3_model` applies the compatibility patch needed to load them
  under newer versions (numerically equivalent).
- **Line endings.** Reference SHA-256 hashes were recorded on Windows; hash checks normalise
  text-file line endings before comparing.
- **Evaluation history.** The in-sample (naive) fusion model was evaluated on the test split
  before the leakage-controlled out-of-fold pipeline was built; both use the same architecture,
  hyperparameters and validation-only threshold rule, and both results are reported.
  Reports for each stage are kept in `outputs/`.

## Environments

The work used several environments (see `requirements/`): generator (Python 3.10), deployment
CNN (Python 3.9.13, PyTorch 2.8.0), out-of-fold CNNs (Python 3.12.10, PyTorch 2.5.1),
M3 (Python 3.13.5, scikit-learn 1.6.1), fusion and analyses (Python 3.12, scikit-learn 1.9.1).

## License

Code: MIT License (see `LICENSE`).
Dataset: CC BY 4.0, https://doi.org/10.5281/zenodo.23136723
