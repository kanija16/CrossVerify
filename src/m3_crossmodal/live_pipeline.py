"""
live_pipeline.py
-----------------
Live cross-modal pipeline for Member 3 (M3).

Flow:
    IMAGE -> live OCR + live QR -> compute_consistency_vector() [Part 2]

This module contains NO consistency, matching, or checksum logic itself;
all consistency evaluation is delegated to Part 2's existing engine.

LEAKAGE NOTE:
document_family is a required argument supplied by the caller.
This module never infers family from filenames, paths, or metadata.
Image pixels are the only evidence source used for feature extraction.
"""

from typing import Dict, Optional

from .live_ocr import extract_ocr_fields, OCRBackend
from .live_qr import extract_qr_fields, QRBackend


def extract_live_modal_fields(
    image_path: str,
    document_family: str,
    ocr_backend: Optional[OCRBackend] = None,
    qr_backend: Optional[QRBackend] = None,
) -> Dict[str, object]:
    """
    Runs live OCR and live QR extraction on one image and returns:
        {
            "ocr_extracted_fields": dict,   # all-None values on OCR failure
            "qr_decoded_fields": dict,      # {} when QR unreadable/unparseable
            "qr_readable": bool,
        }
    """
    ocr_extracted_fields = extract_ocr_fields(image_path, document_family, backend=ocr_backend)
    qr_result = extract_qr_fields(image_path, document_family, backend=qr_backend)

    return {
        "ocr_extracted_fields": ocr_extracted_fields,
        "qr_decoded_fields": qr_result["qr_decoded_fields"],
        "qr_readable": qr_result["qr_readable"],
    }


def run_live_crossmodal(
    image_path: str,
    document_family: str,
    ocr_backend: Optional[OCRBackend] = None,
    qr_backend: Optional[QRBackend] = None,
) -> Dict[str, object]:
    """
    Main Part 3 entry point:
        1. Loads image
        2. Runs live OCR -> ocr_extracted_fields
        3. Runs live QR decoding -> qr_decoded_fields
        4. Passes outputs into Part 2's compute_consistency_vector()
        5. Returns structured results
    """
    from .consistency import compute_consistency_vector

    live_fields = extract_live_modal_fields(
        image_path, document_family, ocr_backend=ocr_backend, qr_backend=qr_backend
    )

    # When OCR produces no extracted text / all-None fields (e.g. missing image,
    # engine failure, or QR-only image without text), pass an empty dict to Part 2
    # so text_qr_match_score cleanly evaluates to None (evidence unavailable).
    raw_ocr = live_fields["ocr_extracted_fields"]
    ocr_for_consistency = (
        {} if all(v is None for v in raw_ocr.values())
        else raw_ocr
    )

    consistency_vector = compute_consistency_vector(
        document_family=document_family,
        ocr_fields=ocr_for_consistency,
        qr_fields=live_fields["qr_decoded_fields"],
    )

    return {
        "ocr_extracted_fields": live_fields["ocr_extracted_fields"],
        "qr_decoded_fields": live_fields["qr_decoded_fields"],
        "consistency_vector": consistency_vector,
    }
