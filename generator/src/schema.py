"""
Fictional identifier schemes.

Two independent, generic (non-real-ID) checksum algorithms are registered
here so different document families can use different schemes:

  - "verhoeff"        : the classic public-domain Verhoeff check-digit
                         algorithm (used by Family A)
  - "weighted_mod11"  : a generic positional-weighted mod-11 check digit,
                         the same general class of scheme used by things
                         like ISBN-10 (used by Family B)

Neither is any real government ID's actual algorithm; they're standard
generic check-digit techniques used here as OUR fictional documents' rule,
selected per document_family via family_config.py.
"""
import json
import random

# ---------------------------------------------------------------------------
# Verhoeff (generic public-domain algorithm)
# ---------------------------------------------------------------------------
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


def verhoeff_checksum(base_digits: str) -> int:
    """Compute the Verhoeff check digit for a base digit string."""
    c = 0
    for i, digit in enumerate(reversed(base_digits)):
        c = _D[c][_P[(i + 1) % 8][int(digit)]]
    return _INV[c]


def verhoeff_validate(full_number: str, expected_length: int = None) -> bool:
    if not full_number.isdigit():
        return False
    if expected_length is not None and len(full_number) != expected_length:
        return False
    c = 0
    for i, digit in enumerate(reversed(full_number)):
        c = _D[c][_P[i % 8][int(digit)]]
    return c == 0


# ---------------------------------------------------------------------------
# Weighted mod-11 (generic ISBN-10-style positional weighting)
# ---------------------------------------------------------------------------
def weighted_mod11_checksum(base_digits: str):
    """Compute a mod-11 check digit with descending positional weights
    starting at len(base_digits)+1. Returns None if the natural result
    would be 10 (no single digit represents it) so the caller can
    regenerate the base instead of using a placeholder character."""
    n = len(base_digits)
    weights = list(range(n + 1, 1, -1))  # e.g. length 9 -> [10,9,...,2]
    total = sum(int(d) * w for d, w in zip(base_digits, weights))
    check = (11 - (total % 11)) % 11
    return None if check == 10 else check


def weighted_mod11_validate(full_number: str, expected_length: int = None) -> bool:
    if not full_number.isdigit():
        return False
    if expected_length is not None and len(full_number) != expected_length:
        return False
    n = len(full_number)
    weights = list(range(n, 0, -1))  # include the check digit itself at weight 1
    total = sum(int(d) * w for d, w in zip(full_number, weights))
    return total % 11 == 0


CHECKSUM_ALGOS = {
    "verhoeff": {"generate": verhoeff_checksum, "validate": verhoeff_validate},
    "weighted_mod11": {"generate": weighted_mod11_checksum, "validate": weighted_mod11_validate},
}


# ---------------------------------------------------------------------------
# Generic dispatch, driven by family config
# ---------------------------------------------------------------------------
def generate_identifier(config: dict, rng: random.Random = random) -> str:
    """Generate a syntactically + checksum-valid identifier for the given
    family config. Never starts below identifier_min_first_digit so freshly
    generated genuine identifiers always pass is_format_valid()."""
    algo = CHECKSUM_ALGOS[config["checksum_algo"]]
    length = config["identifier_length"]
    min_first = config.get("identifier_min_first_digit", 0)

    while True:
        first = str(rng.randint(min_first, 9))
        rest = "".join(str(rng.randint(0, 9)) for _ in range(length - 2))
        base = first + rest
        check = algo["generate"](base)
        if check is not None:
            return base + str(check)
        # else: (weighted_mod11 edge case, check digit would be 10) retry


def validate_checksum(config: dict, id_str: str) -> bool:
    algo = CHECKSUM_ALGOS[config["checksum_algo"]]
    return algo["validate"](id_str, config["identifier_length"])


def is_format_valid(config: dict, id_str: str) -> bool:
    """Structural format check independent of checksum validity."""
    if not isinstance(id_str, str) or not id_str.isdigit():
        return False
    if len(id_str) != config["identifier_length"]:
        return False
    min_first = config.get("identifier_min_first_digit", 0)
    return int(id_str[0]) >= min_first


def build_qr_payload(fields: dict, config: dict) -> str:
    """Build the fictional QR JSON payload for a record, generic over
    whatever fields the family config defines."""
    payload = {f: fields.get(f, "") for f in config["fields"]}
    payload["issuer"] = config["issuer_tag"]
    return json.dumps(payload, sort_keys=True)


def parse_qr_payload(payload: str) -> dict:
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return {}
