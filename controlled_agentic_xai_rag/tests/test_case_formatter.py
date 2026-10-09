"""Tests for case formatter."""

from controlled_agentic_xai_rag.tools.case_formatter import format_case


def test_format_case_deterministic(sample_card, sample_profile):
    a = format_case("high_risk_case", sample_card, sample_profile)
    b = format_case("high_risk_case", sample_card, sample_profile)
    assert a == b
    assert a["case_id"] == "high_risk_case"
    assert "prediction_probability" in a
    assert a["safety_constraints"]["no_diagnosis"] is True
