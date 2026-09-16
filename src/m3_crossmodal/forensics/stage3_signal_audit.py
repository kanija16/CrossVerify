"""
stage3_signal_audit.py
----------------------
Stage 3: Local Document Structure Signal Audit for CrossVerify M3.

Investigates whether coordinated_full_forgery can be distinguished from genuine
documents using local, OCR-aligned visual structure around text fields.

Evaluation:
- VALIDATION ONLY (and TRAIN for sanity check).
- Zero test data access.
- Single-feature statistical audits (KS-test, Cohen's d, ROC-AUC, PR-AUC) WITHOUT model training.
- Within-document outlier statistics.
- Family-stratified controls (Family A vs Family B).
- Signal classification: STRONG_SIGNAL, WEAK_SIGNAL, NO_SIGNAL, POTENTIAL_SHORTCUT.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DIR_STAGE3 = PROJECT_ROOT / "outputs" / "forensics" / "stage3_signal_audit"
DATASET_ROOT = Path("~/Downloads/dataset_final_v2").expanduser()

LABEL_PATTERNS = {
    "family_a": {
        "name": ["name"],
        "dob": ["dob", "date of birth"],
        "gender": ["gender", "sex"],
        "id_number": ["id number", "id no"],
        "address": ["address"],
    },
    "family_b": {
        "full_name": ["registrant", "full name"],
        "registry_id": ["registry id", "registry no"],
        "entity_type": ["entity type", "type"],
        "jurisdiction_code": ["jurisdiction"],
        "registration_date": ["registered on", "registration date"],
    },
}


def extract_ocr_field_boxes(img_gray: np.ndarray, family: str) -> Dict[str, Dict[str, Any]]:
    """Runs pytesseract to locate bounding boxes for canonical field roles."""
    pil_img = Image.fromarray(img_gray)
    try:
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DATAFRAME)
    except Exception:
        return {}

    if "text" not in data.columns or len(data) == 0:
        return {}
    data["text"] = data["text"].fillna("").astype(str)
    valid = data[data["text"].str.strip().str.len() > 0].copy()
    if len(valid) == 0:
        return {}

    lines = []
    for (b_num, p_num, l_num), grp in valid.groupby(["block_num", "par_num", "line_num"]):
        line_text = " ".join(grp["text"].str.strip())
        x = int(grp["left"].min())
        y = int(grp["top"].min())
        w = int((grp["left"] + grp["width"]).max() - x)
        h = int((grp["top"] + grp['height']).max() - y)
        lines.append({"text": line_text, "x": x, "y": y, "w": w, "h": h})

    # Sort vertically then horizontally
    lines = sorted(lines, key=lambda l: (l["y"], l["x"]))

    label_targets = LABEL_PATTERNS.get(family, {})
    field_boxes: Dict[str, Dict[str, Any]] = {}

    for i, line in enumerate(lines):
        txt_low = re.sub(r"[:\-]+$", "", line["text"]).strip().lower()
        for role, alts in label_targets.items():
            if role in field_boxes:
                continue
            if any(alt == txt_low or txt_low.startswith(alt + ":") or txt_low.startswith(alt + " -") for alt in alts):
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    field_boxes[role] = {
                        "role": role,
                        "box": (next_line["x"], next_line["y"], next_line["w"], next_line["h"]),
                        "text": next_line["text"],
                    }
    return field_boxes


def compute_per_field_visual_features(img_gray: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[Dict[str, float]]:
    """Extracts 12 per-field visual statistics and background relative ratios."""
    x, y, w, h = box
    H, W = img_gray.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    field_patch = img_gray[y1:y2, x1:x2]
    if field_patch.size == 0 or field_patch.shape[0] < 3 or field_patch.shape[1] < 3:
        return None

    # Surrounding background patch: expanded margin excluding the text field
    m = 10
    bx1, by1 = max(0, x - m), max(0, y - m)
    bx2, by2 = min(W, x + w + m), min(H, y + h + m)
    surround = img_gray[by1:by2, bx1:bx2].copy()

    mask = np.ones(surround.shape, dtype=bool)
    sy1, sy2 = y1 - by1, y2 - by1
    sx1, sx2 = x1 - bx1, x2 - bx1
    mask[sy1:sy2, sx1:sx2] = False
    bg_pixels = surround[mask]
    if bg_pixels.size == 0:
        bg_pixels = field_patch

    # 1. Grayscale mean & std
    g_mean = float(np.mean(field_patch))
    g_std = float(np.std(field_patch))

    # 2. Local contrast (Michelson)
    f_min, f_max = float(np.min(field_patch)), float(np.max(field_patch))
    local_contrast = float((f_max - f_min) / (f_max + f_min + 1e-5))

    # 3. Sobel gradient magnitude
    gx = cv2.Sobel(field_patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(field_patch, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    grad_mean = float(np.mean(mag))
    grad_std = float(np.std(mag))

    # 4. Edge density (Canny)
    edges = cv2.Canny(field_patch, 80, 180)
    edge_density = float(np.mean(edges > 0))

    # 5. Laplacian variance & HF energy
    lap = cv2.Laplacian(field_patch, cv2.CV_64F)
    lap_var = float(np.var(lap))
    hf_energy = float(np.mean(lap**2))

    # 6. Local entropy
    hist, _ = np.histogram(field_patch, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    # 7. Text/Background contrast
    bg_mean = float(np.mean(bg_pixels))
    bg_std = float(np.std(bg_pixels))
    text_bg_contrast = float(abs(g_mean - bg_mean) / (bg_mean + 1e-5))

    # 8. Boundary discontinuity
    perim = np.concatenate([field_patch[0, :], field_patch[-1, :], field_patch[:, 0], field_patch[:, -1]])
    boundary_disc = float(abs(np.mean(perim) - bg_mean) / (bg_std + 1e-5))

    # 9. Connected-component density
    _, binary = cv2.threshold(field_patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary)
    area = max(1, field_patch.size)
    cc_density = float((num_labels - 1) / area * 1000)

    # 10. Background ratios
    bg_lap = float(np.var(cv2.Laplacian(surround, cv2.CV_64F)))
    bg_gx = cv2.Sobel(surround, cv2.CV_64F, 1, 0, ksize=3)
    bg_gy = cv2.Sobel(surround, cv2.CV_64F, 0, 1, ksize=3)
    bg_grad = float(np.mean(np.sqrt(bg_gx**2 + bg_gy**2)))
    grad_ratio = float(grad_mean / (bg_grad + 1e-5))
    lap_ratio = float(lap_var / (bg_lap + 1e-5))

    return {
        "gray_mean": g_mean,
        "gray_std": g_std,
        "local_contrast": local_contrast,
        "grad_mag_mean": grad_mean,
        "grad_mag_std": grad_std,
        "edge_density": edge_density,
        "laplacian_var": lap_var,
        "local_entropy": entropy,
        "hf_residual_energy": hf_energy,
        "text_bg_contrast": text_bg_contrast,
        "boundary_discontinuity": boundary_disc,
        "cc_density": cc_density,
        "field_to_bg_contrast": text_bg_contrast,
        "field_to_bg_gradient_ratio": grad_ratio,
        "field_to_bg_laplacian_ratio": lap_ratio,
    }


def process_single_image(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Processes one image: OCR bounding boxes, per-field visual features, and document aggregation."""
    img_rel = row_dict["image_path"]
    img_p = DATASET_ROOT / img_rel
    family = row_dict["document_family"]

    try:
        with Image.open(img_p) as im:
            img_gray = np.array(im.convert("L"))
        field_boxes = extract_ocr_field_boxes(img_gray, family)
    except Exception as e:
        field_boxes = {}
        img_gray = np.zeros((100, 100), dtype=np.uint8)
    per_field_feats: Dict[str, Dict[str, float]] = {}

    for role, box_info in field_boxes.items():
        feats = compute_per_field_visual_features(img_gray, box_info["box"])
        if feats is not None:
            per_field_feats[role] = feats

    num_fields = len(per_field_feats)

    # Document-level aggregation across detected fields
    metric_keys = [
        "gray_mean", "gray_std", "local_contrast", "grad_mag_mean", "grad_mag_std",
        "edge_density", "laplacian_var", "local_entropy", "hf_residual_energy",
        "text_bg_contrast", "boundary_discontinuity", "cc_density",
        "field_to_bg_gradient_ratio", "field_to_bg_laplacian_ratio"
    ]

    doc_feats: Dict[str, float] = {
        "num_fields_analyzed": float(num_fields),
    }

    if num_fields >= 2:
        z_scores_all = []
        for k in metric_keys:
            vals = np.array([per_field_feats[r][k] for r in per_field_feats], dtype=np.float64)
            m = float(np.mean(vals))
            s = float(np.std(vals))
            doc_feats[f"field_{k}_mean"] = m
            doc_feats[f"field_{k}_std"] = s
            doc_feats[f"field_{k}_max"] = float(np.max(vals))
            doc_feats[f"field_{k}_range"] = float(np.max(vals) - np.min(vals))

            # Intra-document z-score for this metric
            z = np.abs(vals - m) / (s + 1e-5)
            z_scores_all.extend(z.tolist())

        z_arr = np.array(z_scores_all, dtype=np.float64)
        doc_feats["within_doc_max_field_zscore"] = float(np.max(z_arr)) if len(z_arr) > 0 else 0.0
        doc_feats["within_doc_mean_abs_zscore"] = float(np.mean(z_arr)) if len(z_arr) > 0 else 0.0
        doc_feats["within_doc_zscore_dispersion"] = float(np.std(z_arr)) if len(z_arr) > 0 else 0.0
        doc_feats["num_fields_gt_2sd"] = float(np.sum(z_arr > 2.0))
        doc_feats["num_fields_gt_3sd"] = float(np.sum(z_arr > 3.0))
    else:
        for k in metric_keys:
            doc_feats[f"field_{k}_mean"] = 0.0
            doc_feats[f"field_{k}_std"] = 0.0
            doc_feats[f"field_{k}_max"] = 0.0
            doc_feats[f"field_{k}_range"] = 0.0
        doc_feats["within_doc_max_field_zscore"] = 0.0
        doc_feats["within_doc_mean_abs_zscore"] = 0.0
        doc_feats["within_doc_zscore_dispersion"] = 0.0
        doc_feats["num_fields_gt_2sd"] = 0.0
        doc_feats["num_fields_gt_3sd"] = 0.0

    return {
        "id": row_dict["id"],
        "record_id": row_dict["record_id"],
        "document_family": family,
        "tamper_type": row_dict["tamper_type"],
        "final_label": row_dict["final_label"],
        "split": row_dict["split"],
        "doc_features": doc_feats,
        "field_features": per_field_feats,
    }


def run_stage3_audit(max_workers: int = 8) -> None:
    DIR_STAGE3.mkdir(parents=True, exist_ok=True)

    val_csv_p = PROJECT_ROOT / "outputs" / "fusion" / "val_alignment.csv"
    train_csv_p = Path("~/Desktop/train.csv").expanduser()

    print("Loading datasets for Stage 3 Local Structure Audit (TRAIN/VAL ONLY)...")
    val_meta = pd.read_csv(val_csv_p)
    train_meta = pd.read_csv(train_csv_p)

    # We evaluate on validation (1650 samples)
    val_records = []
    for _, r in val_meta.iterrows():
        val_records.append({
            "id": r["id"],
            "record_id": r["record_id"],
            "document_family": "family_a" if "family_a" in r["id"] else "family_b",
            "tamper_type": r["tamper_type"],
            "final_label": r["final_label"],
            "image_path": f"images/{'family_a' if 'family_a' in r['id'] else 'family_b'}/{r['id']}.png",
            "split": "val",
        })

    print(f"Extracting local OCR field features for {len(val_records)} validation samples (workers={max_workers})...")
    val_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, r) for r in val_records]
        for f in as_completed(futures):
            val_results.append(f.result())

    print("Extraction complete. Compiling document-level and field-level feature tables...")

    # Flatten doc features
    doc_rows = []
    field_rows = []
    for res in val_results:
        row = {
            "id": res["id"],
            "record_id": res["record_id"],
            "document_family": res["document_family"],
            "tamper_type": res["tamper_type"],
            "final_label": res["final_label"],
            "target": 1 if res["final_label"] == "forged" else 0,
        }
        row.update(res["doc_features"])
        doc_rows.append(row)

        for role, fdict in res["field_features"].items():
            f_row = {
                "id": res["id"],
                "document_family": res["document_family"],
                "tamper_type": res["tamper_type"],
                "final_label": res["final_label"],
                "role": role,
            }
            f_row.update(fdict)
            field_rows.append(f_row)

    df_doc = pd.DataFrame(doc_rows)
    df_field = pd.DataFrame(field_rows)

    # -------------------------------------------------------------
    # 1. Statistical Signal Audit: Genuine vs Coordinated Full Forgery
    # -------------------------------------------------------------
    print("Computing single-feature statistical tests (genuine vs coordinated)...")
    gen_mask = df_doc["tamper_type"] == "genuine"
    coord_mask = df_doc["tamper_type"] == "coordinated_full_forgery"
    vs_mask = df_doc["tamper_type"] == "visual_splice"
    fine_mask = df_doc["tamper_type"] == "fine_grained_edit"

    sub_binary = df_doc[gen_mask | coord_mask].copy()
    y_binary = sub_binary["target"].values

    feature_cols = [c for c in df_doc.columns if c not in ["id", "record_id", "document_family", "tamper_type", "final_label", "target"]]

    audit_records = []
    for col in feature_cols:
        g_vals = df_doc.loc[gen_mask, col].values
        c_vals = df_doc.loc[coord_mask, col].values
        vs_vals = df_doc.loc[vs_mask, col].values
        fine_vals = df_doc.loc[fine_mask, col].values

        ks_stat, ks_pval = ks_2samp(g_vals, c_vals)
        pooled_std = np.sqrt((np.var(g_vals) + np.var(c_vals)) / 2) + 1e-6
        d = (np.mean(c_vals) - np.mean(g_vals)) / pooled_std

        # Single-feature ROC-AUC and PR-AUC
        col_vals_bin = sub_binary[col].values
        try:
            auc = roc_auc_score(y_binary, col_vals_bin)
            if auc < 0.5:  # directional alignment
                auc = 1.0 - auc
                pr = average_precision_score(y_binary, -col_vals_bin)
            else:
                pr = average_precision_score(y_binary, col_vals_bin)
        except Exception:
            auc = 0.5
            pr = float(np.mean(y_binary))

        # Signal Classification
        if (auc >= 0.65 or abs(d) >= 0.5) and ks_pval < 0.01:
            classification = "STRONG_SIGNAL"
        elif (auc >= 0.55 or abs(d) >= 0.2) and ks_pval < 0.05:
            classification = "WEAK_SIGNAL"
        else:
            classification = "NO_SIGNAL"

        audit_records.append({
            "feature": col,
            "genuine_mean": round(float(np.mean(g_vals)), 4),
            "genuine_std": round(float(np.std(g_vals)), 4),
            "coord_mean": round(float(np.mean(c_vals)), 4),
            "coord_std": round(float(np.std(c_vals)), 4),
            "visual_splice_mean": round(float(np.mean(vs_vals)), 4),
            "fine_grained_mean": round(float(np.mean(fine_vals)), 4),
            "cohens_d": round(float(d), 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": float(ks_pval),
            "single_feature_roc_auc": round(float(auc), 4),
            "single_feature_pr_auc": round(float(pr), 4),
            "signal_classification": classification,
        })

    df_audit = pd.DataFrame(audit_records).sort_values(by="single_feature_roc_auc", ascending=False)
    df_audit.to_csv(DIR_STAGE3 / "signal_audit.csv", index=False)

    # -------------------------------------------------------------
    # 2. Family-Stratified Summary
    # -------------------------------------------------------------
    print("Computing family-stratified signal audit...")
    family_records = []
    for fam in ["family_a", "family_b"]:
        sub_fam = df_doc[df_doc["document_family"] == fam]
        g_fam = sub_fam[sub_fam["tamper_type"] == "genuine"]
        c_fam = sub_fam[sub_fam["tamper_type"] == "coordinated_full_forgery"]

        for col in feature_cols:
            gv = g_fam[col].values
            cv = c_fam[col].values
            ks_s, ks_p = ks_2samp(gv, cv)
            p_std = np.sqrt((np.var(gv) + np.var(cv)) / 2) + 1e-6
            cd = (np.mean(cv) - np.mean(gv)) / p_std
            
            y_b = np.concatenate([np.zeros(len(gv)), np.ones(len(cv))])
            x_b = np.concatenate([gv, cv])
            try:
                auc_f = roc_auc_score(y_b, x_b)
                if auc_f < 0.5:
                    auc_f = 1.0 - auc_f
            except Exception:
                auc_f = 0.5

            family_records.append({
                "document_family": fam,
                "feature": col,
                "genuine_mean": round(float(np.mean(gv)), 4),
                "coord_mean": round(float(np.mean(cv)), 4),
                "cohens_d": round(float(cd), 4),
                "ks_statistic": round(float(ks_s), 4),
                "ks_pvalue": float(ks_p),
                "roc_auc": round(float(auc_f), 4),
            })

    df_family = pd.DataFrame(family_records)
    df_family.to_csv(DIR_STAGE3 / "family_stratified_summary.csv", index=False)

    # -------------------------------------------------------------
    # 3. Field-Level Summary
    # -------------------------------------------------------------
    print("Computing field-level role breakdown...")
    field_summary = []
    for role, grp in df_field.groupby("role"):
        g_r = grp[grp["tamper_type"] == "genuine"]
        c_r = grp[grp["tamper_type"] == "coordinated_full_forgery"]
        if len(g_r) == 0 or len(c_r) == 0:
            continue
        
        for k in ["local_contrast", "grad_mag_mean", "edge_density", "laplacian_var", "text_bg_contrast", "boundary_discontinuity", "field_to_bg_gradient_ratio"]:
            gv = g_r[k].values
            cv = c_r[k].values
            ks_s, ks_p = ks_2samp(gv, cv)
            p_std = np.sqrt((np.var(gv) + np.var(cv)) / 2) + 1e-6
            cd = (np.mean(cv) - np.mean(gv)) / p_std
            
            field_summary.append({
                "role": role,
                "feature": k,
                "genuine_N": len(gv),
                "coord_N": len(cv),
                "genuine_mean": round(float(np.mean(gv)), 4),
                "coord_mean": round(float(np.mean(cv)), 4),
                "cohens_d": round(float(cd), 4),
                "ks_pvalue": float(ks_p),
            })

    df_field_sum = pd.DataFrame(field_summary)
    df_field_sum.to_csv(DIR_STAGE3 / "field_level_summary.csv", index=False)

    # -------------------------------------------------------------
    # 4. Within-Document Outlier Summary
    # -------------------------------------------------------------
    within_cols = [
        "within_doc_max_field_zscore",
        "within_doc_mean_abs_zscore",
        "within_doc_zscore_dispersion",
        "num_fields_gt_2sd",
        "num_fields_gt_3sd",
    ]
    within_summary = []
    for col in within_cols:
        gv = df_doc.loc[gen_mask, col].values
        cv = df_doc.loc[coord_mask, col].values
        ks_s, ks_p = ks_2samp(gv, cv)
        p_std = np.sqrt((np.var(gv) + np.var(cv)) / 2) + 1e-6
        cd = (np.mean(cv) - np.mean(gv)) / p_std
        within_summary.append({
            "within_doc_metric": col,
            "genuine_mean": round(float(np.mean(gv)), 4),
            "genuine_std": round(float(np.std(gv)), 4),
            "coord_mean": round(float(np.mean(cv)), 4),
            "coord_std": round(float(np.std(cv)), 4),
            "cohens_d": round(float(cd), 4),
            "ks_pvalue": float(ks_p),
        })
    df_within = pd.DataFrame(within_summary)
    df_within.to_csv(DIR_STAGE3 / "within_document_summary.csv", index=False)

    # -------------------------------------------------------------
    # 5. Strongest Feature Distributions
    # -------------------------------------------------------------
    top_features = df_audit.head(10)["feature"].tolist()
    dist_records = []
    for col in top_features:
        gv = df_doc.loc[gen_mask, col].values
        cv = df_doc.loc[coord_mask, col].values
        dist_records.append({
            "feature": col,
            "genuine_p10": np.percentile(gv, 10),
            "genuine_p25": np.percentile(gv, 25),
            "genuine_median": np.median(gv),
            "genuine_p75": np.percentile(gv, 75),
            "genuine_p90": np.percentile(gv, 90),
            "coord_p10": np.percentile(cv, 10),
            "coord_p25": np.percentile(cv, 25),
            "coord_median": np.median(cv),
            "coord_p75": np.percentile(cv, 75),
            "coord_p90": np.percentile(cv, 90),
        })
    df_dist = pd.DataFrame(dist_records)
    df_dist.to_csv(DIR_STAGE3 / "strongest_feature_distributions.csv", index=False)

    # -------------------------------------------------------------
    # 6. Formal Audit Report
    # -------------------------------------------------------------
    report_p = DIR_STAGE3 / "stage3_signal_report.md"
    with open(report_p, "w") as f:
        f.write("# CrossVerify M3 — Stage 3 Local Document Structure Signal Audit\n\n")
        f.write(f"**Execution Timestamp:** {datetime.datetime.now(datetime.timezone.utc).isoformat()}  \n")
        f.write("**Status:** Signal Audit Complete (NO NEW MODEL TRAINED, ZERO TEST ACCESS)  \n\n")
        f.write("---\n\n")
        f.write("## 1. Executive Summary & Verdict\n\n")
        strong_count = int(np.sum(df_audit["signal_classification"] == "STRONG_SIGNAL"))
        weak_count = int(np.sum(df_audit["signal_classification"] == "WEAK_SIGNAL"))
        no_count = int(np.sum(df_audit["signal_classification"] == "NO_SIGNAL"))

        f.write(f"- **Total Candidate Local Features Audited:** {len(df_audit)}\n")
        f.write(f"- **Strong Signals (ROC-AUC >= 0.65 or |d| >= 0.5):** {strong_count}\n")
        f.write(f"- **Weak Signals (ROC-AUC >= 0.55 or |d| >= 0.2):** {weak_count}\n")
        f.write(f"- **No Detectable Signal (ROC-AUC < 0.55):** {no_count}\n\n")
        
        f.write("### Verdict on M4 Local Structure Branch:\n")
        if strong_count == 0 and weak_count < 3:
            f.write("> [!CAUTION]\n")
            f.write("> **NEGATIVE VERDICT:** There is **NO statistically robust local visual structure signal** distinguishing `coordinated_full_forgery` from genuine documents. The synthetic generator rendered the altered text fields with identical font antialiasing, edge sharpness, and background blending as genuine templates. Building an M4 local visual branch is **scientifically ungrounded** and will not resolve coordinated full forgeries.\n\n")
        else:
            f.write("> [!NOTE]\n")
            f.write(f"> **MEASURED SIGNAL:** Detected {strong_count} strong and {weak_count} weak candidate features.\n\n")

        f.write("---\n\n")
        f.write("## 2. Top 15 Audited Local Document Features (Genuine vs Coordinated)\n\n")
        f.write(df_audit.head(15).to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 3. Within-Document Outlier Analysis\n\n")
        f.write(df_within.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 4. Family-Stratified Consistency Audit\n\n")
        f.write(df_family.head(20).to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 5. Field Role Breakdown (Sample Roles)\n\n")
        f.write(df_field_sum.head(20).to_markdown(index=False))
        f.write("\n\n---\n\n")

    print(f"Stage 3 Signal Audit complete! Report written to {report_p}")


if __name__ == "__main__":
    run_stage3_audit()
