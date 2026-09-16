"""
test_live_ocr.py
-----------------
Tests for image loading, live OCR interface, failure handling, and per-family parsing.
"""

from m3_crossmodal.image_utils import ImageLoadError, load_image_as_array
from m3_crossmodal.live_ocr import (
    run_ocr,
    extract_ocr_fields,
    extract_ocr_debug,
    parse_family_a_fields,
    parse_family_b_fields,
    FAMILY_A_FIELDS,
    FAMILY_B_FIELDS,
)


def test_load_image_missing_file_raises():
    import pytest
    with pytest.raises(ImageLoadError):
        load_image_as_array("/nonexistent/path/does_not_exist.png")


def test_load_image_valid_file(text_image_factory):
    path = text_image_factory(["hello"])
    array = load_image_as_array(path)
    assert array.shape[2] == 3  # RGB


def test_run_ocr_missing_image_reports_failure_not_exception():
    result = run_ocr("/nonexistent/path/does_not_exist.png")
    assert result.success is False
    assert result.raw_text is None
    assert result.error is not None


def test_run_ocr_on_rendered_text_image_succeeds(text_image_factory):
    path = text_image_factory(["Name: Jane Doe", "DOB: 1990-01-01"])
    result = run_ocr(path)
    assert result.success is True
    assert isinstance(result.raw_text, str)


def test_parse_family_a_fields_extracts_known_labels():
    text = "Name: Jane Doe\nDOB: 1990-01-01\nGender: F\nID Number: 234567890123\nAddress: 12 Main St"
    fields = parse_family_a_fields(text)
    assert fields["name"] == "Jane Doe"
    assert fields["dob"] == "1990-01-01"
    assert fields["gender"] == "F"
    assert fields["id_number"] == "234567890123"
    assert fields["address"] == "12 Main St"


def test_parse_family_a_fields_missing_lines_are_none():
    text = "Name: Jane Doe"
    fields = parse_family_a_fields(text)
    assert fields["name"] == "Jane Doe"
    assert fields["dob"] is None
    assert fields["address"] is None


def test_parse_family_a_fields_empty_text_returns_all_none():
    fields = parse_family_a_fields("")
    assert set(fields.keys()) == set(FAMILY_A_FIELDS)
    assert all(v is None for v in fields.values())


def test_parse_family_b_fields_extracts_known_labels():
    text = (
        "Full Name: Acme Corp\n"
        "Registry ID: 1234567895\n"
        "Entity Type: Corporation\n"
        "Jurisdiction Code: US-DE\n"
        "Registration Date: 2010-05-01"
    )
    fields = parse_family_b_fields(text)
    assert fields["full_name"] == "Acme Corp"
    assert fields["registry_id"] == "1234567895"
    assert fields["entity_type"] == "Corporation"
    assert fields["jurisdiction_code"] == "US-DE"
    assert fields["registration_date"] == "2010-05-01"


def test_parse_family_b_fields_empty_text_returns_all_none():
    fields = parse_family_b_fields(None)
    assert set(fields.keys()) == set(FAMILY_B_FIELDS)
    assert all(v is None for v in fields.values())


def test_extract_ocr_fields_shape_family_a(text_image_factory):
    path = text_image_factory(["Name: Jane Doe"])
    fields = extract_ocr_fields(path, "family_a")
    assert set(fields.keys()) == set(FAMILY_A_FIELDS)


def test_extract_ocr_fields_shape_family_b(text_image_factory):
    path = text_image_factory(["Full Name: Acme Corp"])
    fields = extract_ocr_fields(path, "family_b")
    assert set(fields.keys()) == set(FAMILY_B_FIELDS)


def test_extract_ocr_fields_on_missing_image_returns_all_none():
    fields = extract_ocr_fields("/nonexistent/does_not_exist.png", "family_a")
    assert set(fields.keys()) == set(FAMILY_A_FIELDS)
    assert all(v is None for v in fields.values())


def test_extract_ocr_fields_unknown_family_raises():
    import pytest
    with pytest.raises(ValueError):
        extract_ocr_fields("/nonexistent/does_not_exist.png", "family_z")


def test_family_a_parser_output_has_expected_field_names():
    fields = parse_family_a_fields("Name: Jane Doe")
    assert set(fields.keys()) == {"name", "dob", "gender", "id_number", "address"}


def test_family_b_parser_output_has_expected_field_names():
    fields = parse_family_b_fields("Full Name: Acme Corp")
    assert set(fields.keys()) == {
        "full_name", "registry_id", "entity_type", "jurisdiction_code", "registration_date",
    }


def test_extract_ocr_fields_never_contains_raw_text_or_internal_keys(text_image_factory):
    path = text_image_factory(["Name: Jane Doe"])
    fields = extract_ocr_fields(path, "family_a")
    assert "_raw_ocr_text" not in fields
    assert "raw_text" not in fields
    assert "ocr_success" not in fields
    assert set(fields.keys()) == set(FAMILY_A_FIELDS)


def test_extract_ocr_debug_exposes_raw_text_separately(text_image_factory):
    path = text_image_factory(["Name: Jane Doe"])
    debug = extract_ocr_debug(path, "family_a")
    assert set(debug.keys()) == {"fields", "ocr_success", "raw_text", "error"}
    assert set(debug["fields"].keys()) == set(FAMILY_A_FIELDS)
    assert isinstance(debug["raw_text"], str)
    assert "_raw_ocr_text" not in debug["fields"]
