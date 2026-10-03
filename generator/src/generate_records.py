"""
Generate genuine fictional-document records, generic over document family.

Each family has its own set of non-identifier field generators (registered
below); the identifier itself is always produced via schema.generate_identifier
using the family's configured checksum algorithm. No real people, no real ID
scheme, for either family.
"""
import random

from faker import Faker

from schema import generate_identifier, build_qr_payload

fake = Faker("en_IN")


def _family_a_field(field: str, rng: random.Random):
    if field == "name":
        return fake.name()
    if field == "dob":
        return fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%d/%m/%Y")
    if field == "gender":
        return rng.choice(["Male", "Female"])
    if field == "address":
        return fake.address().replace("\n", ", ")
    raise ValueError(f"family_a: no generator for field {field!r}")


def _family_b_field(field: str, rng: random.Random):
    if field == "full_name":
        return fake.company() if rng.random() < 0.5 else fake.name()
    if field == "entity_type":
        return rng.choice(["Individual", "Business", "Non-Profit"])
    if field == "jurisdiction_code":
        return rng.choice(["JX-01", "JX-02", "JX-03", "JX-04", "JX-05"])
    if field == "registration_date":
        return fake.date_between(start_date="-20y", end_date="today").strftime("%d/%m/%Y")
    raise ValueError(f"family_b: no generator for field {field!r}")


_FIELD_GENERATORS = {
    "family_a": _family_a_field,
    "family_b": _family_b_field,
}


def generate_field_value(config: dict, field: str, rng: random.Random = random):
    """Generate a single plausible value for a non-identifier field. Also
    used by tamper.py to produce a realistic *replacement* value when
    simulating a text-tamper (rather than corrupting the string directly)."""
    return _FIELD_GENERATORS[config["key"]](field, rng)


def generate_record(config: dict, rng: random.Random = random) -> dict:
    """Generate one genuine record for the given family config."""
    id_field = config["identifier_field"]
    fields = {
        f: generate_field_value(config, f, rng)
        for f in config["fields"]
        if f != id_field
    }
    fields[id_field] = generate_identifier(config, rng)
    # Keep the identity namespace deterministic under generate_records(seed).
    # This also gives the programmatic portrait a stable per-record key.
    fields["record_id"] = f"{rng.getrandbits(64):016x}"
    fields["document_family"] = config["key"]
    fields["qr_payload"] = build_qr_payload(fields, config)
    return fields


def generate_records(config: dict, n: int, seed: int = None) -> list:
    rng = random.Random(seed)
    Faker.seed(seed)
    return [generate_record(config, rng) for _ in range(n)]


if __name__ == "__main__":
    import json
    import sys
    from family_config import get_family

    fam_key = sys.argv[1] if len(sys.argv) > 1 else "family_a"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cfg = get_family(fam_key)
    for r in generate_records(cfg, n, seed=42):
        print(json.dumps(r, indent=2))
