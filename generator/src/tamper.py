"""
Generate tampered variants of a genuine record, generic over document
family. Each tamper family is parameterized purely off the family's config
(identifier_field, other fields, checksum/format rules) — no family-specific
branching in the tamper logic itself.

Tamper categories (identical set for both families; see README for why each
one is logically applicable to both):
  genuine
  text_qr_mismatch    - a printed non-identifier field edited, QR unchanged
  qr_only_mismatch    - QR payload edited, printed fields unchanged
  checksum_invalid    - identifier's check digit corrupted
  format_invalid      - identifier malformed (wrong length / leading digit)
  field_missing       - a required printed field blanked out
  fine_grained_edit   - one printed non-identifier character changed, QR unchanged
  visual_splice       - pure image-level tamper; text/QR stay consistent
                         (only the CNN forensic module should catch this)
"""
import copy
import random

from schema import generate_identifier, validate_checksum, is_format_valid, build_qr_payload
from generate_records import generate_field_value


def _non_identifier_fields(config):
    return [f for f in config["fields"] if f != config["identifier_field"]]


def _different_field_value(record, config, field, rng, max_attempts=100):
    """Sample a realistic replacement that cannot turn a tamper into a no-op."""
    original = str(record[field])
    for _ in range(max_attempts):
        candidate = str(generate_field_value(config, field, rng))
        if candidate != original:
            return candidate
    raise RuntimeError(f"Could not generate a distinct replacement for {field!r}")


def _base_consistency_vector(config):
    return {
        "text_qr_match_score": 1.0,
        "checksum_valid": True,
        "format_valid": True,
        "field_presence_flags": {f: True for f in config["fields"]},
    }


def make_genuine(record, config):
    r = copy.deepcopy(record)
    r["render_field_overrides"] = {}
    r["render_qr_payload"] = record["qr_payload"]
    r["consistency_vector"] = _base_consistency_vector(config)
    r["tamper_type"] = "genuine"
    r["cnn_label"] = "genuine"
    r["final_label"] = "genuine"
    return r


def tamper_text_qr_mismatch(record, config, rng=random):
    r = copy.deepcopy(record)
    field = rng.choice(_non_identifier_fields(config))
    new_value = _different_field_value(record, config, field, rng)
    r[field] = new_value

    r["render_field_overrides"] = {field: new_value}
    r["render_qr_payload"] = record["qr_payload"]  # QR keeps OLD value

    cv = _base_consistency_vector(config)
    cv["text_qr_match_score"] = 0.2
    r["consistency_vector"] = cv
    r["tamper_type"] = "text_qr_mismatch"
    r["tampered_field"] = field
    r["cnn_label"] = "genuine"
    r["final_label"] = "forged"
    return r


def tamper_qr_only_mismatch(record, config, rng=random):
    r = copy.deepcopy(record)
    field = rng.choice(_non_identifier_fields(config))
    fake_fields = dict(record)
    fake_fields[field] = _different_field_value(record, config, field, rng)
    bad_payload = build_qr_payload(fake_fields, config)

    r["render_field_overrides"] = {}  # printed fields unchanged
    r["render_qr_payload"] = bad_payload

    cv = _base_consistency_vector(config)
    cv["text_qr_match_score"] = 0.2
    r["consistency_vector"] = cv
    r["tamper_type"] = "qr_only_mismatch"
    r["cnn_label"] = "genuine"
    r["final_label"] = "forged"
    return r


def tamper_checksum_invalid(record, config, rng=random):
    r = copy.deepcopy(record)
    id_field = config["identifier_field"]
    digits = list(r[id_field])
    digits[-1] = str((int(digits[-1]) + rng.randint(1, 9)) % 10)
    bad_id = "".join(digits)
    r[id_field] = bad_id
    r["render_field_overrides"] = {id_field: bad_id}
    r["render_qr_payload"] = record["qr_payload"]  # QR still has the OLD valid id

    cv = _base_consistency_vector(config)
    cv["checksum_valid"] = validate_checksum(config, bad_id)  # should be False
    cv["text_qr_match_score"] = 0.3
    r["consistency_vector"] = cv
    r["tamper_type"] = "checksum_invalid"
    r["cnn_label"] = "genuine"
    r["final_label"] = "forged"
    return r


def tamper_format_invalid(record, config, rng=random):
    """Break only the configured leading-digit format rule.

    The identifier keeps its original length and check digit. A non-check
    digit may be adjusted to retain checksum validity, making this a pure
    format violation rather than a disguised checksum-invalid sample.
    """
    r = copy.deepcopy(record)
    id_field = config["identifier_field"]
    min_first = config.get("identifier_min_first_digit", 0)
    bad_first = str(max(0, min_first - 1)) if min_first > 0 else "0"
    digits = list(bad_first + r[id_field][1:])

    # Preserve the original check digit. Adjust a payload digit only when
    # needed so the checksum rule remains satisfied after the leading-digit
    # format violation.
    if not validate_checksum(config, "".join(digits)):
        found_valid_payload = False
        for position in range(1, len(digits) - 1):
            original = digits[position]
            for replacement in "0123456789":
                if replacement == original:
                    continue
                digits[position] = replacement
                if validate_checksum(config, "".join(digits)):
                    found_valid_payload = True
                    break
            if found_valid_payload:
                break
            digits[position] = original
        if not found_valid_payload:
            raise RuntimeError("Could not construct checksum-valid format violation")
    bad_id = "".join(digits)

    r[id_field] = bad_id
    r["render_field_overrides"] = {id_field: bad_id}
    r["render_qr_payload"] = record["qr_payload"]

    cv = _base_consistency_vector(config)
    cv["format_valid"] = is_format_valid(config, bad_id)  # should be False
    cv["checksum_valid"] = validate_checksum(config, bad_id)
    cv["text_qr_match_score"] = 0.3
    r["consistency_vector"] = cv
    r["tamper_type"] = "format_invalid"
    r["cnn_label"] = "genuine"
    r["final_label"] = "forged"
    return r


def tamper_field_missing(record, config, rng=random):
    r = copy.deepcopy(record)
    field = rng.choice(_non_identifier_fields(config))
    r[field] = ""
    r["render_field_overrides"] = {field: ""}
    r["render_qr_payload"] = record["qr_payload"]

    cv = _base_consistency_vector(config)
    cv["field_presence_flags"][field] = False
    cv["text_qr_match_score"] = 0.4
    r["consistency_vector"] = cv
    r["tamper_type"] = "field_missing"
    r["missing_field"] = field
    r["cnn_label"] = "genuine"
    r["final_label"] = "forged"
    return r


def _single_character_substitution(value, rng):
    """Return a deterministic one-character, length-preserving edit."""
    eligible_positions = [index for index, char in enumerate(value) if char.isalnum()]
    if not eligible_positions:
        raise ValueError("Fine-grained edit requires at least one alphanumeric character")
    position = rng.choice(eligible_positions)
    original = value[position]
    alphabet = "0123456789" if original.isdigit() else "abcdefghijklmnopqrstuvwxyz"
    if original.isupper():
        alphabet = alphabet.upper()
    replacement = rng.choice([char for char in alphabet if char != original])
    return value[:position] + replacement + value[position + 1:], position


def tamper_fine_grained_edit(record, config, rng=random):
    """Create a one-character printed/QR near-miss without touching IDs."""
    r = copy.deepcopy(record)
    candidates = config.get("fine_grained_edit_fields", _non_identifier_fields(config))
    candidates = [field for field in candidates if field != config["identifier_field"]]
    if not candidates:
        raise ValueError("No non-identifier fine-grained edit fields configured")
    field = rng.choice(candidates)
    original_value = str(record[field])
    edited_value, changed_position = _single_character_substitution(original_value, rng)
    r[field] = edited_value

    r["render_field_overrides"] = {field: edited_value}
    r["render_qr_payload"] = record["qr_payload"]
    r["fine_grained_edit_metadata"] = {
        "edited_field": field,
        "original_value": original_value,
        "edited_value": edited_value,
        "edit_type": "single_character_substitution",
        "changed_position": changed_position,
    }

    cv = _base_consistency_vector(config)
    cv["text_qr_match_score"] = 0.8
    r["consistency_vector"] = cv
    r["tamper_type"] = "fine_grained_edit"
    r["cnn_label"] = "genuine"
    r["final_label"] = "forged"
    return r


def tamper_visual_splice(record, config, rng=random, splice_variant=None):
    """Marks the record for a pure image-level splice/copy-move applied at
    render time (see augment.py). Text and QR stay fully self-consistent —
    this is the family only the CNN forensic module can catch. Applicable
    identically to both document families since it operates on the
    rendered image, not on any family-specific field."""
    r = copy.deepcopy(record)
    r["render_field_overrides"] = {}
    r["render_qr_payload"] = record["qr_payload"]
    r["apply_visual_splice"] = True
    r["splice_variant"] = splice_variant

    cv = _base_consistency_vector(config)  # fully consistent on text/QR side
    r["consistency_vector"] = cv
    r["tamper_type"] = "visual_splice"
    r["cnn_label"] = "forged"
    r["final_label"] = "forged"
    return r


def tamper_coordinated_full_forgery(record, config, rng=random):
    """Forge every configured field while keeping all semantic checks valid.

    This intentionally has no visual-only artifact.  It is a hard-limit case:
    printed text, QR payload, checksum, and format all agree on a plausible
    but fabricated fictional record, while the final label remains forged.
    """
    r = copy.deepcopy(record)
    id_field = config["identifier_field"]
    forged_fields = {}
    for field in config["fields"]:
        if field == id_field:
            candidate = generate_identifier(config, rng)
            while candidate == record[field]:
                candidate = generate_identifier(config, rng)
            forged_fields[field] = candidate
        else:
            forged_fields[field] = _different_field_value(record, config, field, rng)
    r.update(forged_fields)
    r["render_field_overrides"] = dict(forged_fields)
    r["render_qr_payload"] = build_qr_payload(r, config)

    r["consistency_vector"] = _base_consistency_vector(config)
    r["tamper_type"] = "coordinated_full_forgery"
    # There is intentionally no visual forensic change for the CNN to use.
    r["cnn_label"] = "genuine"
    r["final_label"] = "forged"
    return r


def make_visual_splice_variants(record, config, rng=random, count=3):
    """Create independent copy-move samples for one identity."""
    return [tamper_visual_splice(record, config, rng=rng, splice_variant=i)
            for i in range(count)]


TAMPER_FUNCS = [
    tamper_text_qr_mismatch,
    tamper_qr_only_mismatch,
    tamper_checksum_invalid,
    tamper_format_invalid,
    tamper_field_missing,
    tamper_fine_grained_edit,
    tamper_coordinated_full_forgery,
]
