"""
Run real OCR (pytesseract) and real QR decoding (pyzbar) on rendered card
images so the dataset contains REAL extraction noise, not idealized ground
truth. This is what your fuzzy-match consistency module should actually be
tuned against.
"""
import re

import pytesseract
from PIL import Image

try:
    from pyzbar.pyzbar import decode as zbar_decode
    _HAS_ZBAR = True
except ImportError:
    _HAS_ZBAR = False


def ocr_extract_text(img: Image.Image, config: dict) -> str:
    # The registry layout places a dense QR code in a separate right-hand
    # column.  Exclude that column from text OCR so Tesseract's page
    # segmentation does not treat QR modules as document glyphs.
    if config["layout"] == "registry_v1":
        img = img.crop((0, 0, round(img.width * 0.65), img.height))
    return pytesseract.image_to_string(img)


def ocr_extract_fields(img: Image.Image, config: dict) -> dict:
    """Label:value line parser, generic over document family via
    config['field_labels'] (field_key -> printed label text). Real pipelines
    would use layout-aware OCR; this is enough to inject genuine OCR noise
    into the dataset regardless of which family's layout was rendered."""
    text = ocr_extract_text(img, config)
    lines = [ln.strip() for ln in text.splitlines()]
    label_lookup = {label: field for field, label in config["field_labels"].items()}

    found = []
    for i, line in enumerate(lines):
        stripped = line.rstrip(":").strip()
        if stripped in label_lookup:
            found.append((i, label_lookup[stripped]))
    found.sort()

    fields = {}
    for pos, (line_idx, field_key) in enumerate(found):
        next_idx = found[pos + 1][0] if pos + 1 < len(found) else len(lines)
        value_lines = [ln for ln in lines[line_idx + 1:next_idx] if ln]
        fields[field_key] = " ".join(value_lines).strip()

    for field_key in config["field_labels"]:
        fields.setdefault(field_key, "")

    fields["_raw_ocr_text"] = text
    return fields


def qr_decode_fields(img: Image.Image) -> dict:
    """Decode the QR code from the image. Falls back to empty dict if pyzbar
    isn't available or no QR is found (e.g. it's been fully removed as a
    tamper type)."""
    if not _HAS_ZBAR:
        return {"_error": "pyzbar not installed"}
    results = zbar_decode(img)
    if not results:
        return {}
    import json

    try:
        return json.loads(results[0].data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"_error": "undecodable_payload"}
