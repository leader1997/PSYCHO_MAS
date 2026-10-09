"""Pipeline smoke test with mocked LLM."""

from unittest.mock import patch

import pytest

from controlled_agentic_xai_rag.pipeline import PipelineOptions, run_case
from controlled_agentic_xai_rag.retrieval.base import RetrievalHit


SAMPLE_REPORT = {
    "case_id": "high_risk_case",
    "screening_summary": "Elevated screening signal for review.",
    "probability_interpretation": "Model estimate 0.98 — not a diagnosis.",
    "xai_explanation": "Academic impairment contributed to the model prediction.",
    "evidence_grounded_interpretation": "Literature links academic stress to screening indicators.",
    "non_clinical_recommendation": "Consider speaking with a qualified counselor.",
    "limitations": "This is a model-based screening signal and not a clinical diagnosis.",
    "citations": [],
    "safety_status": "draft_unvalidated",
}


@pytest.fixture
def mock_llm():
    mock_hits = [
        RetrievalHit(
            chunk_id="DOC_A_p1_c1",
            rank=1,
            score=1.0,
            title="Test",
            page=1,
            text="Sample evidence.",
            method="hybrid_faiss",
        )
    ]
    with patch("controlled_agentic_xai_rag.pipeline.hybrid_retrieve", return_value=([], mock_hits, {"requested_mode": "hybrid"})):
        with patch("controlled_agentic_xai_rag.agents.report_generation.structured_json", return_value=SAMPLE_REPORT):
            with patch("controlled_agentic_xai_rag.agents.query_planning.structured_json") as qp:
                qp.return_value = {
                    "report_questions": ["How should the expert interpret the screening probability?"],
                    "retrieval_queries": [],
                }
                yield


def test_pipeline_smoke(mock_llm, seeded_inputs, built_corpus):
    opts = PipelineOptions(live_prediction=False)
    result = run_case("high_risk_case", opts)
    assert result["final_report"]["case_id"] == "high_risk_case"
    assert result["final_report"]["safety_status"] == "draft_unvalidated"
    assert result.get("query_plan")
    assert result.get("validators_skipped") is True
