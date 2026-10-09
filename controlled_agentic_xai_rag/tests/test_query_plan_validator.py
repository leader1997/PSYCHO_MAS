"""Tests for query plan validator."""

from controlled_agentic_xai_rag.tools.query_plan_validator import validate_query_plan
from controlled_agentic_xai_rag.tools.query_templates import build_template_slots, slots_to_query_plan


def test_validator_passes_template_plan(sample_card, sample_profile, sample_formatted):
    slots = build_template_slots("high_risk_case", sample_card, sample_formatted)
    plan = slots_to_query_plan("high_risk_case", slots)
    audit = validate_query_plan(plan, sample_card, sample_formatted)
    assert audit["query_plan"]["retrieval_queries"]
    assert "risk_communication" in {q["slot_type"] for q in audit["query_plan"]["retrieval_queries"]}


def test_unsafe_query_repaired(sample_card, sample_profile, sample_formatted):
    slots = build_template_slots("high_risk_case", sample_card, sample_formatted)
    plan = slots_to_query_plan("high_risk_case", slots)
    plan["retrieval_queries"][0]["query_text"] = "prescribe medication treatment plan for patient"
    audit = validate_query_plan(plan, sample_card, sample_formatted)
    assert audit["repairs"]


def test_non_diagnostic_query_is_safe(sample_card, sample_profile, sample_formatted):
    slots = build_template_slots("high_risk_case", sample_card, sample_formatted)
    plan = slots_to_query_plan("high_risk_case", slots)
    audit = validate_query_plan(plan, sample_card, sample_formatted)
    assert audit["passed"] is True
