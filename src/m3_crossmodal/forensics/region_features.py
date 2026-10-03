"""
region_features.py
------------------
Candidate I: Document-Region Anomaly Features for CrossVerify M3.

Extracts format-agnostic, intra-document statistical consistency features
across spatial document regions (header, body, footer, QR):

1. High-frequency residual statistics (Laplacian std, energy).
2. Local gradient magnitude statistics (Sobel).
3. Edge density statistics (Canny edge proportion).
4. Local entropy statistics.
5. Document-relative regional anomaly metrics:
   - Mean across regions
   - Dispersion (std) across regions
   - Coefficient of variation (CV) across regions
   - Maximum standardized regional deviation (z-score)
   - Cross-regional contrast ratios (QR vs Body)
   - Count of anomalous regions (|z| > 1.5)

Operates strictly on decoded RGB pixel arrays.
Rejects all forbidden labels and metadata.
"""

from typing import Any, Dict, List, Optional, Set
import cv2
import numpy as np

# Canonical feature list (16 features)
FORENSIC_FEATURE_NAMES: List[str] = [
    "forensic_reg_lap_std_mean",
    "forensic_reg_lap_std_dispersion",
    "forensic_reg_lap_std_cv",
    "forensic_reg_lap_std_max_z",
    "forensic_reg_grad_mag_mean",
    "forensic_reg_grad_mag_dispersion",
    "forensic_reg_grad_mag_cv",
    "forensic_reg_grad_mag_max_z",
    "forensic_reg_edge_density_mean",
    "forensic_reg_edge_density_dispersion",
    "forensic_reg_edge_density_max_z",
    "forensic_reg_entropy_mean",
    "forensic_reg_entropy_dispersion",
    "forensic_reg_qr_vs_body_grad_ratio",
    "forensic_reg_qr_vs_body_lap_ratio",
    "forensic_reg_anomalous_regions_count",
]

FORBIDDEN_FORENSIC_COLUMNS: Set[str] = {
    "id",
    "record_id",
    "filename",
    "path",
    "image_path",
    "label_path",
    "tamper_type",
    "final_label",
    "cnn_label",
    "ground_truth_fields",
    "expected_consistency_vector",
    "target",
    "file_size",
}


def assert_clean_forensic_features(feature_names: List[str]) -> None:
    violating = [c for c in feature_names if c in FORBIDDEN_FORENSIC_COLUMNS or c.endswith(("_path", "_label"))]
    if violating:
        raise ValueError(f"LEAKAGE VIOLATION: Forbidden columns in forensic features: {violating}")


def _compute_entropy(img_gray: np.ndarray) -> float:
    if img_gray.size == 0:
        return 0.0
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def extract_document_regions(
    img_gray: np.ndarray,
    document_family: str,
) -> Dict[str, np.ndarray]:
    """
    Slices the image into 4 canonical document regions according to family layout constants:
    - Header
    - Body text
    - QR code area
    - Footer
    """
    h, w = img_gray.shape[:2]
    regions: Dict[str, np.ndarray] = {}

    if document_family == "family_a":
        # Family A (1000x640): QR is bottom right
        regions["header"] = img_gray[0 : max(1, int(h * 0.15)), 0:w]
        regions["body"] = img_gray[max(1, int(h * 0.15)) : int(h * 0.85), 0 : int(w * 0.65)]
        regions["qr"] = img_gray[int(h * 0.55) : h, int(w * 0.65) : w]
        regions["footer"] = img_gray[int(h * 0.85) : h, 0 : int(w * 0.65)]
    else:
        # Family B (1150x520): QR is top right
        regions["header"] = img_gray[0 : max(1, int(h * 0.20)), 0 : int(w * 0.65)]
        regions["qr"] = img_gray[0 : int(h * 0.55), int(w * 0.65) : w]
        regions["body"] = img_gray[max(1, int(h * 0.20)) : int(h * 0.85), 0 : int(w * 0.65)]
        regions["footer"] = img_gray[int(h * 0.85) : h, 0:w]

    return regions


def compute_region_statistics(region: np.ndarray) -> Dict[str, float]:
    """Computes Laplacian, Sobel gradient, Canny edges, and entropy for a single region."""
    if region.size == 0:
        return {"lap_std": 0.0, "grad_mag": 0.0, "edge_density": 0.0, "entropy": 0.0}

    # Laplacian
    lap = cv2.Laplacian(region, cv2.CV_64F)
    lap_std = float(np.std(lap))

    # Sobel Gradient Magnitude
    gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(gx**2 + gy**2)))

    # Canny Edge Density
    edges = cv2.Canny(region, 80, 180)
    edge_density = float(np.mean(edges > 0))

    # Entropy
    entropy = _compute_entropy(region)

    return {
        "lap_std": lap_std,
        "grad_mag": grad_mag,
        "edge_density": edge_density,
        "entropy": entropy,
    }


def extract_region_anomaly_features(
    image_array: np.ndarray,
    document_family: str,
) -> Dict[str, float]:
    """
    Extracts 16 intra-document regional anomaly features from an image array.
    Guaranteed to return all FORENSIC_FEATURE_NAMES with finite float values.
    """
    if image_array.ndim == 3:
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array

    regions = extract_document_regions(img_gray, document_family)
    reg_stats = {name: compute_region_statistics(patch) for name, patch in regions.items()}

    # Collect per-metric arrays across regions
    lap_stds = [s["lap_std"] for s in reg_stats.values()]
    grad_mags = [s["grad_mag"] for s in reg_stats.values()]
    edge_densities = [s["edge_density"] for s in reg_stats.values()]
    entropies = [s["entropy"] for s in reg_stats.values()]

    def _intra_stats(vals: List[float], eps: float = 1e-5):
        arr = np.array(vals, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        cv_val = std_val / (mean_val + eps)
        z_scores = np.abs(arr - mean_val) / (std_val + eps)
        max_z = float(np.max(z_scores))
        anom_count = int(np.sum(z_scores > 1.5))
        return mean_val, std_val, cv_val, max_z, anom_count

    lap_mean, lap_std, lap_cv, lap_max_z, lap_anom = _intra_stats(lap_stds)
    grad_mean, grad_std, grad_cv, grad_max_z, grad_anom = _intra_stats(grad_mags)
    edge_mean, edge_std, edge_cv, edge_max_z, edge_anom = _intra_stats(edge_densities)
    ent_mean, ent_std, ent_cv, ent_max_z, _ = _intra_stats(entropies)

    # Cross-regional ratios (QR vs Body)
    qr_stats = reg_stats.get("qr", {})
    body_stats = reg_stats.get("body", {})
    qr_grad = qr_stats.get("grad_mag", 0.0)
    body_grad = body_stats.get("grad_mag", 0.0)
    qr_lap = qr_stats.get("lap_std", 0.0)
    body_lap = body_stats.get("lap_std", 0.0)

    grad_ratio = float(qr_grad / (body_grad + 1e-4))
    lap_ratio = float(qr_lap / (body_lap + 1e-4))

    total_anom = int(lap_anom + grad_anom + edge_anom)

    feature_dict: Dict[str, float] = {
        "forensic_reg_lap_std_mean": lap_mean,
        "forensic_reg_lap_std_dispersion": lap_std,
        "forensic_reg_lap_std_cv": lap_cv,
        "forensic_reg_lap_std_max_z": lap_max_z,
        "forensic_reg_grad_mag_mean": grad_mean,
        "forensic_reg_grad_mag_dispersion": grad_std,
        "forensic_reg_grad_mag_cv": grad_cv,
        "forensic_reg_grad_mag_max_z": grad_max_z,
        "forensic_reg_edge_density_mean": edge_mean,
        "forensic_reg_edge_density_dispersion": edge_std,
        "forensic_reg_edge_density_max_z": edge_max_z,
        "forensic_reg_entropy_mean": ent_mean,
        "forensic_reg_entropy_dispersion": ent_std,
        "forensic_reg_qr_vs_body_grad_ratio": grad_ratio,
        "forensic_reg_qr_vs_body_lap_ratio": lap_ratio,
        "forensic_reg_anomalous_regions_count": float(total_anom),
    }

    assert_clean_forensic_features(list(feature_dict.keys()))
    return feature_dict
