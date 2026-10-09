"""Mock LLM responses when OpenAI unavailable (smoke tests)."""

from __future__ import annotations

import json
import os


def openai_available() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        from generation.config_loader import load_dotenv

        load_dotenv()
        from openai import OpenAI

        client = OpenAI()
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=2,
        )
        return True
    except Exception:
        return False


def mock_narrative(case_id: str, elevated: bool) -> dict:
    disclaimer = "This is a model-based screening signal and not a clinical diagnosis."
    return {
        "case_id": case_id,
        "screening_summary": (
            f"The model indicates {'elevated' if elevated else 'lower'} screening risk. {disclaimer}"
        ),
        "model_explanation": "Survey features contributed to the model output as associations, not causes.",
        "evidence_grounding": "Retrieved literature discusses associations between student survey factors and distress.",
        "recommended_next_step": (
            "Consider speaking with a qualified counselor or student support service."
            if elevated
            else "Continue monitoring wellbeing and use campus resources as needed."
        ),
        "limitations": disclaimer,
        "citations": [{"chunk_id": "MOCK001", "claim_supported": "Association-level evidence only."}],
        "safety_status": "passed",
    }


def mock_synthesis(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "supported_claims": [
            {
                "claim": "Sleep and academic factors are associated with student mental-health screening indicators.",
                "supporting_chunk_ids": ["MOCK001"],
                "evidence_strength": "moderate",
                "relevance_to_xai": "Aligns with XAI-identified contributors.",
            }
        ],
        "unsupported_or_uncertain_points": [
            {
                "point": "Individual causation cannot be established.",
                "reason": "Evidence is associational.",
            }
        ],
        "clinical_boundary": "Screening communication only; not a clinical diagnosis.",
    }
