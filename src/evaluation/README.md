# Evaluation

This directory contains the evaluation and analysis code for the CrossVerify project.

## Purpose

The evaluation module compares the performance of the three main approaches:

1. CNN
2. ELA + CNN
3. Proposed Cross-Modal Fusion

The evaluation must be performed on the same held-out test set for all models.

## Evaluation Metrics

The following metrics will be calculated:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Analysis

The evaluation module will perform:

### Overall Model Comparison

Compare CNN, ELA + CNN, and the proposed fusion model using the selected evaluation metrics.

### Attack-wise Analysis

Measure performance separately for each forgery type:

- Face inpainting
- Text-field redraw
- QR-region regeneration
- Full-card synthesis
- Hybrid edits

### Confusion Matrix

Generate confusion matrices for the major models to identify:

- True positives
- True negatives
- False positives
- False negatives

### ROC Analysis

Generate ROC curves and calculate ROC-AUC using model probability scores.

### Ablation Study

Evaluate the contribution of individual components by comparing:

- CNN only
- OCR + QR + checksum
- CNN + OCR + QR + checksum
- CNN + OCR + QR without checksum

### Error Analysis

Identify and study false positives and false negatives.

## Input

The evaluation scripts will read model prediction files containing information such as:

- sample_id
- document_type
- true_label
- predicted_label
- probability
- attack_type

The fusion model may additionally provide:

- cnn_score
- ocr_qr_match
- checksum_valid
- format_valid
- field_presence
- fusion_probability
- reason

## Output

The evaluation module will generate:

- Overall performance tables
- Attack-wise performance tables
- Confusion matrices
- ROC curves
- Ablation results
- Error-analysis results
- Figures for the research paper

## Reproducibility

All models must be evaluated using the same test set and the same evaluation definitions.

Final paper results must always be based on actual measured experimental results.
