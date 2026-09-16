"""
test_live_qr.py
----------------
Tests for the live QR interface, unreadable handling, and payload parsing.
"""

import json

from m3_crossmodal.live_qr import (
    run_qr_detection,
    extract_qr_fields,
    parse_qr_payload,
)


def test_run_qr_detection_missing_image_reports_unreadable_not_exception():
    result = run_qr_detection("/nonexistent/does_not_exist.png")
    assert result.readable is False
    assert result.raw_payload is None
    assert result.error is not None


def test_run_qr_detection_on_image_with_no_qr_is_unreadable(blank_image):
    result = run_qr_detection(blank_image)
    assert result.readable is False
    assert result.raw_payload is None


def test_run_qr_detection_on_real_qr_code_succeeds(qr_image_factory):
    payload = json.dumps({"name": "Jane Doe", "dob": "1990-01-01"})
    path = qr_image_factory(payload)
    result = run_qr_detection(path)
    assert result.readable is True
    assert result.raw_payload == payload


def test_parse_qr_payload_json():
    payload = json.dumps({"name": "Jane Doe", "dob": "1990-01-01"})
    fields = parse_qr_payload(payload, "family_a")
    assert fields == {"name": "Jane Doe", "dob": "1990-01-01"}


def test_parse_qr_payload_delimited_key_value():
    payload = "name:Jane Doe;dob:1990-01-01"
    fields = parse_qr_payload(payload, "family_a")
    assert fields == {"name": "Jane Doe", "dob": "1990-01-01"}


def test_parse_qr_payload_delimited_with_equals():
    payload = "name=Jane Doe|dob=1990-01-01"
    fields = parse_qr_payload(payload, "family_a")
    assert fields == {"name": "Jane Doe", "dob": "1990-01-01"}


def test_parse_qr_payload_unparseable_returns_empty_dict():
    fields = parse_qr_payload("just some plain text with no structure", "family_a")
    assert fields == {}


def test_parse_qr_payload_empty_returns_empty_dict():
    assert parse_qr_payload(None, "family_a") == {}
    assert parse_qr_payload("", "family_a") == {}


def test_extract_qr_fields_shape_on_real_qr(qr_image_factory):
    payload = json.dumps({"full_name": "Acme Corp"})
    path = qr_image_factory(payload)
    result = extract_qr_fields(path, "family_b")
    assert set(result.keys()) == {"qr_readable", "qr_decoded_fields"}
    assert result["qr_readable"] is True
    assert result["qr_decoded_fields"] == {"full_name": "Acme Corp"}


def test_extract_qr_fields_shape_on_missing_image():
    result = extract_qr_fields("/nonexistent/does_not_exist.png", "family_a")
    assert result == {"qr_readable": False, "qr_decoded_fields": {}}


def test_extract_qr_fields_shape_on_no_qr_present(blank_image):
    result = extract_qr_fields(blank_image, "family_a")
    assert result == {"qr_readable": False, "qr_decoded_fields": {}}


def test_region_hint_recovers_small_qr_that_whole_image_detection_misses(tmp_path):
    from PIL import Image
    import qrcode

    qr_size = 55
    canvas = Image.new("RGB", (1000, 640), color="white")
    payload = json.dumps({"name": "Jane Doe"})
    qr_img = qrcode.make(payload).resize((qr_size, qr_size))
    canvas.paste(qr_img, (1000 - qr_size - 10, 640 - qr_size - 10))
    path = tmp_path / "small_qr_bottom_right.png"
    canvas.save(path)

    without_hint = run_qr_detection(str(path), document_family=None)
    with_hint = run_qr_detection(str(path), document_family="family_a")

    assert without_hint.readable is False
    assert with_hint.readable is True
    assert with_hint.raw_payload == payload
