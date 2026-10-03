# Generator-Artifact Gating Diagnostic Report

**Dataset Split:** TRAIN ONLY (`train.csv`, $N=7,700$ total, $N=1,400$ analyzed)
**Genuine Count:** 700
**Coordinated Full Forgery Count:** 700

---

## 1. Feature / Statistic Comparison Table

| Feature / Statistic | Genuine Distribution ($N=700$) | Coordinated Forgery Distribution ($N=700$) | Suspicious Separation? | Gating Verdict |
| :--- | :--- | :--- | :---: | :---: |
| **Image Dimensions $(W \times H)$** | {(1150, 520), (1000, 640)} | {(1150, 520), (1000, 640)} | **NO** (Identical) | **Allowed** (Physical constraint) |
| **Encoding Mode & Format** | {('RGB', 'PNG')} | {('RGB', 'PNG')} | **NO** (Identical) | **Allowed** (Physical constraint) |
| **Metadata / EXIF Info Chunks** | 0 keys | 0 keys | **NO** (Both 0) | **Allowed** (Zero metadata present) |
| **File Size (Bytes)** | Mean: 421849.5 ± 445163.0<br>[30800, 1246780] | Mean: 434140.7 ± 455463.1<br>[31128, 1246142] | **NO** (Broad overlapping distribution) | **REJECT AS DIRECT FEATURE** (Avoid file container bias) |
| **Global Pixel Entropy** | Mean: 3.421 ± 1.089<br>[1.786, 5.658] | Mean: 3.451 ± 1.130<br>[1.779, 5.714] | **NO** (Continuous and heavily overlapping) | **Allowed** (As regional intra-doc relative stat) |

---

## 2. Gating Analysis & Conclusions

1. **No Generator Metadata Leakage:** Both genuine and coordinated full forgery files are stripped of auxiliary PNG text chunks, EXIF tags, and software stamps.
2. **Format & Dimension Parity:** Both classes strictly use RGB PNG with identical dimensions per document family (1000x640 for Family A, 1150x520 for Family B).
3. **File Size Policy:** File size exhibits no discrete or bimodal separation between genuine and forged documents. However, to strictly follow best forensic practices, **file size, container headers, and file paths are FORBIDDEN as ML features**.
4. **Pixel Forensics Approval:** All features must operate strictly on decoded pixel arrays using document-relative regional anomaly statistics (intra-document variance, residual differences, and texture consistency across header/body/footer/QR).
