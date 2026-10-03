"""
live_qr.py
----------
Live QR detection and decoding from document images.

Pipeline step:
    image -> raw QR payload string -> best-effort structured field dictionary

Operates ONLY on image pixels. Never reads M1's precomputed fields,
ground truth, or metadata. QR failure is represented cleanly as
unreadable evidence (qr_readable=False) and is never treated as forgery.
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .image_utils import ImageLoadError, load_image_as_array


class QRBackendError(Exception):
    """Raised by a QR backend when detection or decoding fails."""


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class QRBackend:
    def decode(self, image_array: np.ndarray) -> Optional[str]:
        """Returns the decoded payload string, or None if no QR was decoded."""
        raise NotImplementedError


class OpenCVQRBackend(QRBackend):
    """Default backend: OpenCV QRCodeDetector."""

    def __init__(self):
        try:
            import cv2
        except ImportError as e:
            raise QRBackendError(
                "opencv-python (or opencv-python-headless) is not installed."
            ) from e
        self._cv2 = cv2
        self._detector = cv2.QRCodeDetector()

    def decode(self, image_array: np.ndarray) -> Optional[str]:
        try:
            bgr = self._cv2.cvtColor(image_array, self._cv2.COLOR_RGB2BGR)
            data, points, _ = self._detector.detectAndDecode(bgr)
        except Exception as e:
            raise QRBackendError(f"OpenCV QR detection failed: {e}") from e

        if not data:
            return None
        return data


_default_backend: Optional[QRBackend] = None


def get_default_qr_backend() -> QRBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = OpenCVQRBackend()
    return _default_backend


def set_default_qr_backend(backend: QRBackend) -> None:
    """Allows tests or callers to swap the backend."""
    global _default_backend
    _default_backend = backend


# ---------------------------------------------------------------------------
# Known QR region hints per family (benchmark layout facts, not metadata)
# ---------------------------------------------------------------------------

QR_REGION_HINTS = {
    "family_a": {"corner": "bottom_right", "image_width": 1000, "image_height": 640},
    "family_b": {"corner": "top_right", "image_width": 1150, "image_height": 520},
}

_CROP_FRACTION = 0.4
_CROP_UPSCALE_ATTEMPTS = [3.0, 4.0, 2.0]


def _crop_corner(image_array: np.ndarray, corner: str, frac: float = _CROP_FRACTION) -> np.ndarray:
    h, w = image_array.shape[:2]
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
    if corner == "bottom_right":
        return image_array[h - ch:h, w - cw:w]
    if corner == "top_right":
        return image_array[0:ch, w - cw:w]
    if corner == "top_left":
        return image_array[0:ch, 0:cw]
    if corner == "bottom_left":
        return image_array[h - ch:h, 0:cw]
    return image_array


def _upscale(image_array: np.ndarray, factor: float) -> np.ndarray:
    import cv2
    h, w = image_array.shape[:2]
    return cv2.resize(image_array, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Low-level QR call
# ---------------------------------------------------------------------------

@dataclass
class QRResult:
    readable: bool
    raw_payload: Optional[str]
    error: Optional[str]


def run_qr_detection(
    image_path: str,
    document_family: Optional[str] = None,
    backend: Optional[QRBackend] = None,
) -> QRResult:
    """
    Attempts whole-image QR detection first. If that fails and document_family
    is supplied, falls back to the known corner region crop and upscale.
    Never raises exceptions — failure returns QRResult(readable=False, ...).
    """
    try:
        backend = backend or get_default_qr_backend()
    except QRBackendError as e:
        return QRResult(readable=False, raw_payload=None, error=str(e))

    try:
        image_array = load_image_as_array(image_path)
    except ImageLoadError as e:
        return QRResult(readable=False, raw_payload=None, error=str(e))

    try:
        payload = backend.decode(image_array)
    except QRBackendError as e:
        return QRResult(readable=False, raw_payload=None, error=str(e))

    if payload is not None:
        return QRResult(readable=True, raw_payload=payload, error=None)

    # Fallback to corner crop + upscale hint
    hint = QR_REGION_HINTS.get(document_family) if document_family else None
    if hint is not None:
        raw_crop = _crop_corner(image_array, hint["corner"])
        for factor in _CROP_UPSCALE_ATTEMPTS:
            try:
                payload = backend.decode(_upscale(raw_crop, factor))
            except QRBackendError:
                payload = None
            if payload is not None:
                return QRResult(readable=True, raw_payload=payload, error=None)

    return QRResult(readable=False, raw_payload=None, error="No QR code detected/decoded")


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

def parse_qr_payload(raw_payload: Optional[str], document_family: str) -> Dict[str, str]:
    """
    Parses decoded raw QR payload string into a structured field dictionary.
    Supports JSON and delimited key:value / key=value formats.
    """
    if not raw_payload:
        return {}

    # 1. Try JSON first
    try:
        parsed = json.loads(raw_payload)
        if isinstance(parsed, dict):
            return {str(k): ("" if v is None else str(v)) for k, v in parsed.items()}
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Try delimited key:value / key=value pairs
    segments = [s.strip() for s in raw_payload.replace("|", ";").replace("\n", ";").split(";") if s.strip()]
    fields: Dict[str, str] = {}
    for segment in segments:
        for sep in (":", "="):
            if sep in segment:
                key, value = segment.split(sep, 1)
                fields[key.strip()] = value.strip()
                break

    return fields


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def extract_qr_fields(
    image_path: str,
    document_family: str,
    backend: Optional[QRBackend] = None,
) -> Dict[str, object]:
    """
    Runs QR detection on image_path and returns:
        {"qr_readable": bool, "qr_decoded_fields": dict}
    """
    qr_result = run_qr_detection(image_path, document_family=document_family, backend=backend)
    if not qr_result.readable:
        return {"qr_readable": False, "qr_decoded_fields": {}}

    fields = parse_qr_payload(qr_result.raw_payload, document_family)
    return {"qr_readable": True, "qr_decoded_fields": fields}
