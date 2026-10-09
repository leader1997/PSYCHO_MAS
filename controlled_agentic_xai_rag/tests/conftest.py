"""Shared test fixtures."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_card():
    return {
        "case_id": "high_risk_case",
        "prediction_label": "elevated_risk",
        "prediction_probability": 0.979,
        "model_name": "CatBoost",
        "calibration_status": "calibrated",
        "top_positive_factors": [
            {"feature": "aca_impa", "display_name": "Academic impairment", "contribution": 1.2,
             "direction": "increases", "interpretation": "contrib"},
            {"feature": "lonely", "display_name": "Loneliness", "contribution": 0.3,
             "direction": "increases", "interpretation": "contrib"},
        ],
        "top_protective_factors": [],
        "important_warning": "Not a diagnosis.",
    }


@pytest.fixture
def sample_profile():
    return {
        "case_id": "high_risk_case",
        "feature_levels": {"aca_impa": "high", "lonely": "medium"},
    }


@pytest.fixture
def sample_formatted(sample_card, sample_profile):
    from controlled_agentic_xai_rag.tools.case_formatter import format_case
    return format_case("high_risk_case", sample_card, sample_profile)


@pytest.fixture
def seeded_inputs():
    import subprocess
    import sys
    subprocess.run([sys.executable, str(ROOT / "scripts" / "seed_inputs.py")], check=True, cwd=ROOT)
    return ROOT / "data" / "inputs"


@pytest.fixture
def built_corpus():
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_corpus.py"), "--fallback-phase2"],
        check=True,
        cwd=ROOT,
    )
    return ROOT / "data" / "processed_chunks.csv"
