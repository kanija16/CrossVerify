"""
Document family configuration registry.

CrossVerify is one pipeline. Adding a new fictional document family means
adding an entry here (plus a matching layout function in render_card.py and
a field-value generator in generate_records.py) — the rest of the pipeline
(tamper generation, OCR/QR extraction, consistency features, splitting)
reads what it needs from this config and does not hard-code any
family-specific assumptions.

Both families below are entirely fictional: fictional issuer, fictional
identifier scheme, fictional checksum algorithm, fictional QR payload,
fictional visual layout. Neither reproduces any real government document's
layout, identifiers, checksum algorithm, QR schema, logos, or branding.
"""

FAMILY_A = {
    "key": "family_a",
    "display_name": "Fictional Identity Authority",
    "issuer_tag": "FICTIONAL-ID-AUTHORITY",
    "identifier_field": "id_number",
    "identifier_length": 12,
    "identifier_min_first_digit": 2,  # avoid leading 0/1
    "checksum_algo": "verhoeff",
    "fields": ["name", "dob", "gender", "id_number", "address"],
    # Eligible for one-character cross-modal edits; identifiers stay excluded
    # so this tamper is not a checksum or format violation.
    "fine_grained_edit_fields": ["name", "dob", "gender"],
    # OCR label text (as rendered on the card) -> field key, used to parse
    # printed fields back out of the image via regex.
    "field_labels": {
        "name": "Name",
        "dob": "DOB",
        "gender": "Gender",
        "id_number": "ID Number",
        "address": "Address",
    },
    "layout": "identity_v1",
    "template_variants": ["v1", "v2"],
}

FAMILY_B = {
    "key": "family_b",
    "display_name": "Fictional Revenue & Registry Bureau",
    "issuer_tag": "FICTIONAL-REVENUE-BUREAU",
    "identifier_field": "registry_id",
    "identifier_length": 10,
    "identifier_min_first_digit": 1,
    "checksum_algo": "weighted_mod11",
    "fields": [
        "full_name",
        "registry_id",
        "entity_type",
        "jurisdiction_code",
        "registration_date",
    ],
    "fine_grained_edit_fields": [
        "full_name", "entity_type", "jurisdiction_code", "registration_date",
    ],
    "field_labels": {
        "full_name": "Registrant",
        "registry_id": "Registry ID",
        "entity_type": "Entity Type",
        "jurisdiction_code": "Jurisdiction",
        "registration_date": "Registered On",
    },
    "layout": "registry_v1",
    "template_variants": ["v1", "v2"],
}

FAMILIES = {"family_a": FAMILY_A, "family_b": FAMILY_B}


def get_family(key: str) -> dict:
    if key not in FAMILIES:
        raise ValueError(f"Unknown document family: {key!r}. Known: {list(FAMILIES)}")
    return FAMILIES[key]


def all_family_keys():
    return list(FAMILIES.keys())
