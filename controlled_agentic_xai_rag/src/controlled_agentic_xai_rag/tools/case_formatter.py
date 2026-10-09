"""Step 0 — deterministic case formatter."""

from __future__ import annotations

from controlled_agentic_xai_rag.config_loader import cfg_get
from controlled_agentic_xai_rag.schemas import validate


def format_case(case_id: str, card: dict, profile: dict) -> dict:
    prob = float(card.get("prediction_probability", 0.5))
    threshold = float(cfg_get("pipeline", "risk_threshold", default=0.5))
    band = float(cfg_get("pipeline", "near_threshold_band", default=0.1))
    elevated = prob >= threshold
    borderline = abs(prob - threshold) <= band

    formatted = {
        "case_id": case_id,
        "prediction_probability": prob,
        "risk_threshold": threshold,
        "risk_category": "elevated_screening_risk" if elevated else "not_elevated_screening_risk",
        "calibration_status": card.get("calibration_status", "unknown"),
        "top_xai_contributors": card.get("top_positive_factors", []),
        "protective_factors": card.get("top_protective_factors", []),
        "feature_levels": dict(list(profile.get("feature_levels", {}).items())[:25]),
        "safety_constraints": {
            "mandatory_disclaimer": cfg_get(
                "safety", "mandatory_disclaimer",
                default="This is a model-based screening signal and not a clinical diagnosis.",
            ),
            "elevated_risk_requires_support_plan": elevated,
            "no_diagnosis": True,
            "no_treatment_advice": True,
            "contribution_not_causation": True,
            "borderline_probability": borderline,
        },
        "model_name": card.get("model_name", "CatBoost"),
        "prediction_label": card.get("prediction_label", ""),
    }
    ok, errors = validate(formatted, "formatted_case.json")
    if not ok:
        raise ValueError(f"Formatted case failed schema: {errors}")
    return formatted
