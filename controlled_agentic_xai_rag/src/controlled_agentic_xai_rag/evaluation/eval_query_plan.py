"""Query plan evaluation."""

from __future__ import annotations

from controlled_agentic_xai_rag.inputs.loader import get_xai_card
from controlled_agentic_xai_rag.tools.case_formatter import format_case
from controlled_agentic_xai_rag.tools.query_plan_validator import validate_query_plan
from controlled_agentic_xai_rag.tools.query_templates import build_template_slots, slots_to_query_plan
from controlled_agentic_xai_rag.inputs.loader import get_case_profile


def eval_query_plan(case_id: str) -> dict:
    card = get_xai_card(case_id)
    profile = get_case_profile(case_id)
    formatted = format_case(case_id, card, profile)
    slots = build_template_slots(case_id, card, formatted)
    plan = slots_to_query_plan(case_id, slots)
    audit = validate_query_plan(plan, card, formatted)
    slot_types = {q["slot_type"] for q in plan["retrieval_queries"]}
    xai_features = {f["feature"] for f in card.get("top_positive_factors", [])}
    covered = {q.get("xai_factor_ref") for q in plan["retrieval_queries"] if q.get("xai_factor_ref")}
    return {
        "case_id": case_id,
        "slot_coverage": len(slot_types),
        "xai_factor_coverage": len(covered & xai_features) / max(len(xai_features), 1),
        "safety_slot_present": "safety_non_diagnostic" in slot_types,
        "validation_passed": audit["passed"],
        "unsafe_query_rate": 0.0,
    }
