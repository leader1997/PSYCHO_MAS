"""Tests for live HMS prediction integration."""

import pytest

from controlled_agentic_xai_rag.prediction.hms_predictor import (
    PredictionUnavailableError,
    predict_preset_case,
)


@pytest.fixture
def phase1_artifacts():
    try:
        return predict_preset_case("high_risk_case")
    except PredictionUnavailableError as exc:
        pytest.skip(str(exc))


def test_high_risk_preset_probability(phase1_artifacts):
    assert phase1_artifacts.prediction_probability > 0.9
    assert phase1_artifacts.card["prediction_label"] == "elevated_risk"
    assert len(phase1_artifacts.card["top_positive_factors"]) >= 1


def test_profile_has_all_features(phase1_artifacts):
    prof = phase1_artifacts.profile
    assert "feature_levels" in prof
    assert "raw_values" in prof
    assert len(prof["feature_levels"]) == 25


def test_live_pipeline_smoke(phase1_artifacts):
    from controlled_agentic_xai_rag.pipeline import PipelineOptions, run_case

    from unittest.mock import patch

    from controlled_agentic_xai_rag.retrieval.base import RetrievalHit

    opts = PipelineOptions(live_prediction=True)
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
    with patch("controlled_agentic_xai_rag.agents.report_generation.structured_json") as rg:
        with patch("controlled_agentic_xai_rag.agents.query_planning.structured_json") as qp:
            with patch(
                "controlled_agentic_xai_rag.pipeline.hybrid_retrieve",
                return_value=([], mock_hits, {"requested_mode": "hybrid", "actual_mode": "hybrid_faiss"}),
            ):
                rg.return_value = {
                    "case_id": "high_risk_case",
                    "screening_summary": "Test summary.",
                    "probability_interpretation": "Test.",
                    "xai_explanation": "Test.",
                    "evidence_grounded_interpretation": "Test.",
                    "non_clinical_recommendation": "Test.",
                    "limitations": "Not a diagnosis.",
                    "citations": [],
                }
                qp.return_value = {"report_questions": ["Q?"], "retrieval_queries": []}
                result = run_case("high_risk_case", opts)
    assert result["prediction"] is not None
    assert result["inputs"]["source"] == "live_hms_prediction"
    assert result["final_report"]["case_id"] == "high_risk_case"
