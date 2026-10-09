"""JSON schema loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parent


def load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def validate(instance: dict, schema_name: str) -> tuple[bool, list[str]]:
    schema = load_schema(schema_name)
    errors = []
    validator = jsonschema.Draft7Validator(schema)
    for err in validator.iter_errors(instance):
        errors.append(err.message)
    return len(errors) == 0, errors
