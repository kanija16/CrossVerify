"""
validators.py
--------------
Checksum and format validators for the two fictional document families in CrossVerify:
  - Family A ("family_a"): Synthetic identity card (Aadhaar style)
  - Family B ("family_b"): Synthetic registration certificate

Algorithms:
1. Family A Checksum:
   - Identifier field: 'id_number'
   - Algorithm: Classic Verhoeff algorithm over the full 12-digit string.
   - Validation: Evaluates True if the full 12 digits satisfy the Verhoeff check table.

2. Family A Format:
   - Must be a 12-digit numeric string with the first digit in range [2-9] (no 0 or 1).

3. Family B Checksum:
   - Identifier field: 'registry_id'
   - Algorithm: Weighted Modulo-11 with weights [1, 2, 3, 4, 5, 6, 7, 8, 9] over the
     first 9 digits. The 10th digit is the check digit.
   - Validation: Evaluates True if sum(d[i] * (i + 1) for i in 0..8) % 11 == d[9].

4. Family B Format:
   - Must be a 10-digit numeric string with the first digit in range [1-9] (no 0).
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Family A: Verhoeff Checksum
# ---------------------------------------------------------------------------

class VerhoeffChecksum:
    """
    Implementation of the Verhoeff dihedral group D5 checksum algorithm.
    Used for Family A 12-digit identifiers.
    """

    _D = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ]

    _P = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
    ]

    _INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

    @classmethod
    def is_valid(cls, digits: str) -> bool:
        """
        Validate full identifier including check digit.
        Returns True if Verhoeff checksum produces 0.
        """
        if not isinstance(digits, str) or not digits.isdigit() or len(digits) == 0:
            return False
        c = 0
        for i, ch in enumerate(reversed(digits)):
            c = cls._D[c][cls._P[i % 8][int(ch)]]
        return c == 0

    @classmethod
    def generate_check_digit(cls, body: str) -> str:
        """
        Compute the check digit for an unfinalized number string.
        """
        if not isinstance(body, str) or not body.isdigit():
            raise ValueError("body must contain digits only")
        c = 0
        for i, ch in enumerate(reversed(body)):
            c = cls._D[c][cls._P[(i + 1) % 8][int(ch)]]
        return str(cls._INV[c])


def family_a_checksum_is_valid(identifier: str) -> bool:
    """
    Validate Family A checksum.
    Requires exactly 12 numeric digits. Evaluated independently of format rules.
    """
    if not isinstance(identifier, str) or len(identifier) != 12 or not identifier.isdigit():
        return False
    return VerhoeffChecksum.is_valid(identifier)


def family_a_format_is_valid(identifier: str) -> bool:
    """
    Validate Family A format: exactly 12 digits, first digit in [2-9].
    """
    return (
        isinstance(identifier, str)
        and len(identifier) == 12
        and identifier.isdigit()
        and identifier[0] in "23456789"
    )


# ---------------------------------------------------------------------------
# Family B: Weighted Mod-11 Checksum
# ---------------------------------------------------------------------------

MOD11_WEIGHTS_FAMILY_B = [1, 2, 3, 4, 5, 6, 7, 8, 9]


class Mod11WeightedChecksum:
    """
    Weighted Modulo-11 checksum for Family B 10-digit identifiers.
    Check digit: sum(d[i] * (i + 1) for i in 0..8) % 11 == d[9].
    """

    @classmethod
    def compute_check_digit(cls, body: str, weights=MOD11_WEIGHTS_FAMILY_B) -> Optional[int]:
        if not isinstance(body, str) or len(body) != len(weights) or not body.isdigit():
            return None
        total = sum(int(d) * w for d, w in zip(body, weights))
        remainder = total % 11
        if remainder == 10:
            # Remainder 10 cannot fit in a single decimal digit
            return None
        return remainder

    @classmethod
    def is_valid(cls, identifier: str, weights=MOD11_WEIGHTS_FAMILY_B) -> bool:
        if not isinstance(identifier, str) or len(identifier) != len(weights) + 1:
            return False
        if not identifier.isdigit():
            return False
        body, check_char = identifier[:-1], identifier[-1]
        expected = cls.compute_check_digit(body, weights)
        if expected is None:
            return False
        return str(expected) == check_char

    @classmethod
    def generate_check_digit(cls, body: str, weights=MOD11_WEIGHTS_FAMILY_B) -> str:
        expected = cls.compute_check_digit(body, weights)
        if expected is None:
            raise ValueError(f"Cannot generate decimal check digit for body '{body}' (remainder is 10)")
        return str(expected)


def family_b_checksum_is_valid(identifier: str) -> bool:
    """
    Validate Family B checksum.
    Requires exactly 10 numeric digits. Evaluated independently of format rules.
    """
    if not isinstance(identifier, str) or len(identifier) != 10 or not identifier.isdigit():
        return False
    return Mod11WeightedChecksum.is_valid(identifier, MOD11_WEIGHTS_FAMILY_B)


def family_b_format_is_valid(identifier: str) -> bool:
    """
    Validate Family B format: exactly 10 digits, first digit in [1-9].
    """
    return (
        isinstance(identifier, str)
        and len(identifier) == 10
        and identifier.isdigit()
        and identifier[0] in "123456789"
    )


# ---------------------------------------------------------------------------
# Family Registry & Dispatch Helpers
# ---------------------------------------------------------------------------

IDENTIFIER_FIELD = {
    "family_a": "id_number",
    "family_b": "registry_id",
}

FORMAT_VALIDATORS = {
    "family_a": family_a_format_is_valid,
    "family_b": family_b_format_is_valid,
}

CHECKSUM_VALIDATORS = {
    "family_a": family_a_checksum_is_valid,
    "family_b": family_b_checksum_is_valid,
}

EXPECTED_FAMILY_FIELDS = {
    "family_a": ["name", "dob", "gender", "id_number", "address"],
    "family_b": ["full_name", "registry_id", "entity_type", "jurisdiction_code", "registration_date"],
}


def get_identifier_field(document_family: str) -> str:
    try:
        return IDENTIFIER_FIELD[document_family]
    except KeyError as e:
        raise ValueError(f"Unknown document_family: {document_family!r}") from e


def validate_format(document_family: str, identifier: str) -> bool:
    try:
        validator = FORMAT_VALIDATORS[document_family]
    except KeyError as e:
        raise ValueError(f"Unknown document_family: {document_family!r}") from e
    return validator(identifier)


def validate_checksum(document_family: str, identifier: str) -> bool:
    try:
        validator = CHECKSUM_VALIDATORS[document_family]
    except KeyError as e:
        raise ValueError(f"Unknown document_family: {document_family!r}") from e
    return validator(identifier)


def get_expected_fields(document_family: str) -> list:
    try:
        return EXPECTED_FAMILY_FIELDS[document_family]
    except KeyError as e:
        raise ValueError(f"Unknown document_family: {document_family!r}") from e
