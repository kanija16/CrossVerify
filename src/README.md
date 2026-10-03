# Source Code

This directory contains the source code for the CrossVerify project.

CrossVerify is a cross-modal identity-document forgery detection system that combines visual and structural evidence.

## Project Components

### baseline/
Contains the traditional image-based forgery detection methods:

- CNN
- Error Level Analysis (ELA)
- ELA + CNN

### cross_modal/
Contains the proposed cross-modal verification components:

- OCR
- QR decoding
- Checksum validation
- Format validation
- OCR-QR consistency checking
- Feature fusion

### evaluation/
Contains the experimental evaluation code:

- Performance metrics
- Model comparison
- Attack-wise analysis
- Confusion matrices
- ROC-AUC analysis
- Ablation study
- Error analysis

## Important Rule

All models must use the same dataset split and the same test set so that the final comparison is fair and reproducible.
