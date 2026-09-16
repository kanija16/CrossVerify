"""
test_live_pipeline.py
----------------------
Tests for the live cross-modal pipeline and integration with Part 2.
"""

import json

import m3_crossmodal.live_pipeline as live_pipeline
import m3_crossmodal.live_ocr as live_ocr
import m3_crossmodal.live_qr as live_qr
from m3_crossmodal.live_pipeline import extract_live_modal_fields, run_live_crossmodal
from m3_crossmodal.consistency import compute_consistency_vector

EXPECTED_CONSISTENCY_KEYS = {
    "qr_readable",
    "text_qr_match_score",
    "checksum_valid",
    "format_valid",
    "missing_field_count",
}


def test_live_ocr_module_imports_successfully():
    assert hasattr(live_ocr, "extract_ocr_fields")


def test_live_qr_module_imports_successfully():
    assert hasattr(live_qr, "extract_qr_fields")


def test_live_pipeline_module_imports_successfully():
    assert hasattr(live_pipeline, "run_live_crossmodal")


def test_run_live_crossmodal_invalid_path_does_not_raise():
    result = run_live_crossmodal("/nonexistent/does_not_exist.png", "family_b")
    assert set(result.keys()) == {"ocr_extracted_fields", "qr_decoded_fields", "consistency_vector"}
    assert set(result["consistency_vector"].keys()) == EXPECTED_CONSISTENCY_KEYS


def test_run_live_crossmodal_on_missing_image_reports_unavailable_not_forged():
    result = run_live_crossmodal("/nonexistent/does_not_exist.png", "family_a")
    cv = result["consistency_vector"]
    assert cv["qr_readable"] is False
    assert cv["text_qr_match_score"] is None
    assert cv["format_valid"] is False
    assert cv["checksum_valid"] is False


def test_extract_live_modal_fields_shape(text_image_factory):
    path = text_image_factory(["Name: Jane Doe"])
    result = extract_live_modal_fields(path, "family_a")
    assert set(result.keys()) == {"ocr_extracted_fields", "qr_decoded_fields", "qr_readable"}
    assert isinstance(result["ocr_extracted_fields"], dict)
    assert isinstance(result["qr_decoded_fields"], dict)
    assert isinstance(result["qr_readable"], bool)


def test_run_live_crossmodal_output_shape(text_image_factory):
    path = text_image_factory(["Name: Jane Doe", "DOB: 1990-01-01"])
    result = run_live_crossmodal(path, "family_a")
    assert set(result.keys()) == {"ocr_extracted_fields", "qr_decoded_fields", "consistency_vector"}
    assert set(result["consistency_vector"].keys()) == EXPECTED_CONSISTENCY_KEYS


def test_run_live_crossmodal_no_qr_gives_unreadable_and_none_score(text_image_factory):
    path = text_image_factory(["Name: Jane Doe", "DOB: 1990-01-01"])
    result = run_live_crossmodal(path, "family_a")
    cv = result["consistency_vector"]
    assert cv["qr_readable"] is False
    assert cv["text_qr_match_score"] is None
    assert result["ocr_extracted_fields"]["name"] == "Jane Doe"


def test_live_output_can_be_passed_into_part2_consistency_engine_directly(text_image_factory):
    path = text_image_factory(["Name: Jane Doe", "DOB: 1990-01-01"])
    live_fields = extract_live_modal_fields(path, "family_a")

    consistency_vector = compute_consistency_vector(
        document_family="family_a",
        ocr_fields=live_fields["ocr_extracted_fields"],
        qr_fields=live_fields["qr_decoded_fields"],
    )

    assert set(consistency_vector.keys()) == EXPECTED_CONSISTENCY_KEYS
    assert consistency_vector["qr_readable"] is False
    assert consistency_vector["text_qr_match_score"] is None


def test_run_live_crossmodal_end_to_end_with_real_qr(qr_image_factory):
    payload = json.dumps({"name": "Jane Doe"})
    path = qr_image_factory(payload)

    result = run_live_crossmodal(path, "family_a")
    cv = result["consistency_vector"]

    assert set(result.keys()) == {"ocr_extracted_fields", "qr_decoded_fields", "consistency_vector"}
    assert cv["qr_readable"] is True
    assert result["qr_decoded_fields"] == {"name": "Jane Doe"}
    assert cv["text_qr_match_score"] is None
