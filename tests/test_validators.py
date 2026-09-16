"""
test_validators.py
-------------------
Unit tests for the fictional Family A (Verhoeff-based) and
Family B (weighted Mod-11) checksum + format validators.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m3_crossmodal.validators import (
    VerhoeffChecksum,
    Mod11WeightedChecksum,
    MOD11_WEIGHTS_FAMILY_B,
    family_a_checksum_is_valid,
    family_a_format_is_valid,
    family_b_checksum_is_valid,
    family_b_format_is_valid,
    validate_format,
    validate_checksum,
    get_identifier_field,
)


def make_family_a_id(body_11_digits: str) -> str:
    check = VerhoeffChecksum.generate_check_digit(body_11_digits)
    return body_11_digits + check


def make_family_b_id(body_9_digits: str) -> str:
    check = Mod11WeightedChecksum.generate_check_digit(body_9_digits, MOD11_WEIGHTS_FAMILY_B)
    return body_9_digits + check


# ---------------------------------------------------------------------------
# Family A
# ---------------------------------------------------------------------------

def test_family_a_valid_identifier_passes_checksum():
    identifier = make_family_a_id("23456789012")
    assert len(identifier) == 12
    assert family_a_checksum_is_valid(identifier) is True


def test_family_a_invalid_checksum_fails():
    identifier = make_family_a_id("23456789012")
    tampered = identifier[:-1] + str((int(identifier[-1]) + 1) % 10)
    assert family_a_checksum_is_valid(tampered) is False


def test_family_a_format_valid():
    identifier = make_family_a_id("23456789012")
    assert family_a_format_is_valid(identifier) is True


def test_family_a_format_invalid_wrong_length():
    assert family_a_format_is_valid("23456789") is False
    assert family_a_format_is_valid("2345678901234") is False


def test_family_a_format_invalid_non_digit():
    identifier = make_family_a_id("23456789012")
    bad = "A" + identifier[1:]
    assert family_a_format_is_valid(bad) is False


def test_family_a_format_invalid_first_digit():
    # First digit must be 2-9; 0 and 1 should fail format
    identifier0 = make_family_a_id("03456789012")
    assert family_a_format_is_valid(identifier0) is False
    # Checksum can still be mathematically valid independently
    assert family_a_checksum_is_valid(identifier0) is True

    identifier1 = make_family_a_id("13456789012")
    assert family_a_format_is_valid(identifier1) is False
    assert family_a_checksum_is_valid(identifier1) is True


# ---------------------------------------------------------------------------
# Family B
# ---------------------------------------------------------------------------

def test_family_b_valid_identifier_passes_checksum():
    identifier = make_family_b_id("123456780")
    assert len(identifier) == 10
    assert family_b_checksum_is_valid(identifier) is True


def test_family_b_invalid_checksum_fails():
    identifier = make_family_b_id("123456780")
    tampered = identifier[:-1] + str((int(identifier[-1]) + 1) % 10)
    assert family_b_checksum_is_valid(tampered) is False


def test_family_b_format_valid():
    identifier = make_family_b_id("123456780")
    assert family_b_format_is_valid(identifier) is True


def test_family_b_format_invalid_wrong_length():
    assert family_b_format_is_valid("12345") is False
    assert family_b_format_is_valid("12345678901") is False


def test_family_b_format_invalid_non_digit():
    identifier = make_family_b_id("123456780")
    bad = identifier[:-1] + "X"
    assert family_b_format_is_valid(bad) is False


def test_family_b_format_invalid_first_digit_zero():
    identifier0 = make_family_b_id("023456789")
    assert family_b_format_is_valid(identifier0) is False
    # Checksum can still be mathematically valid independently
    assert family_b_checksum_is_valid(identifier0) is True


# ---------------------------------------------------------------------------
# Family registry / dispatch helpers
# ---------------------------------------------------------------------------

def test_get_identifier_field():
    assert get_identifier_field("family_a") == "id_number"
    assert get_identifier_field("family_b") == "registry_id"


def test_validate_format_and_checksum_dispatch_family_a():
    identifier = make_family_a_id("23456789012")
    assert validate_format("family_a", identifier) is True
    assert validate_checksum("family_a", identifier) is True


def test_validate_format_and_checksum_dispatch_family_b():
    identifier = make_family_b_id("123456780")
    assert validate_format("family_b", identifier) is True
    assert validate_checksum("family_b", identifier) is True


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        validate_format("family_z", "123")
    with pytest.raises(ValueError):
        validate_checksum("family_z", "123")
    with pytest.raises(ValueError):
        get_identifier_field("family_z")
