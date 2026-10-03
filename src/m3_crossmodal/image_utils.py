"""
image_utils.py
---------------
Shared image loading helper for live OCR and live QR modules.
"""

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageLoadError(Exception):
    """Raised when an image path is missing or the file cannot be read as an image."""


def load_image_as_array(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and returns it as an RGB numpy array.
    Raises ImageLoadError if missing, corrupt, or unsupported.
    """
    path = Path(image_path)
    if not path.exists():
        raise ImageLoadError(f"Image not found: {path}")
    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    except (UnidentifiedImageError, OSError) as e:
        raise ImageLoadError(f"Could not read image at {path}: {e}") from e
