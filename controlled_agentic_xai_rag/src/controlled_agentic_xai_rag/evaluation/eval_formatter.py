"""Case formatter evaluation."""

from __future__ import annotations

import json

from controlled_agentic_xai_rag.inputs.loader import get_case_profile, get_xai_card
from controlled_agentic_xai_rag.schemas import validate
from controlled_agentic_xai_rag.tools.case_formatter import format_case


def eval_formatter(case_id: str) -> dict:
    card = get_xai_card(case_id)
    profile = get_case_profile(case_id)
    a = format_case(case_id, card, profile)
    b = format_case(case_id, card, profile)
    ok, errors = validate(a, "formatted_case.json")
    return {
        "case_id": case_id,
        "json_valid": ok,
        "schema_errors": errors,
        "deterministic": a == b,
        "required_fields_present": all(k in a for k in (
            "case_id", "prediction_probability", "risk_threshold", "top_xai_contributors"
        )),
    }
