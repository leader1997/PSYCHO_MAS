"""Tests for XAI alignment."""

from controlled_agentic_xai_rag.validators.xai_alignment import check_xai_alignment


def test_causal_misuse_detected(sample_card):
    report = {
        "case_id": "high_risk_case",
        "xai_explanation": "Loneliness caused by academic stress causes the risk.",
        "screening_summary": "",
        "probability_interpretation": "",
        "evidence_grounded_interpretation": "",
        "non_clinical_recommendation": "",
        "limitations": "",
    }
    result = check_xai_alignment(report, sample_card)
    assert result["passed"] is False
    assert result["causal_misuse"]
