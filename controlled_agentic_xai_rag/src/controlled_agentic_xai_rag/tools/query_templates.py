"""Mandatory deterministic query slots."""

from __future__ import annotations

from controlled_agentic_xai_rag.config_loader import cfg_get


def _needs_uncertainty_slot(prob: float, calibration: str) -> bool:
    threshold = float(cfg_get("pipeline", "risk_threshold", default=0.5))
    band = float(cfg_get("pipeline", "near_threshold_band", default=0.1))
    if abs(prob - threshold) <= band:
        return True
    return calibration.lower() in ("uncertain", "uncalibrated", "unknown")


def _level_label(levels: dict, feat: str) -> str:
    return str(levels.get(feat, "unknown"))


def build_template_slots(case_id: str, card: dict, formatted_case: dict) -> list[dict]:
    """Build mandatory query slots with query_id and slot_type."""
    slots: list[dict] = []
    idx = 1
    levels = formatted_case.get("feature_levels") or {}
    prob = float(card.get("prediction_probability", 0.5))
    risk_label = "elevated" if prob >= float(cfg_get("pipeline", "risk_threshold", default=0.5)) else "not elevated"

    def add(slot_type: str, query_text: str, keywords: list[str], xai_ref: str = "") -> None:
        nonlocal idx
        slots.append({
            "query_id": f"{case_id}_q{idx:02d}",
            "slot_type": slot_type,
            "query_text": query_text,
            "focus_keywords": keywords,
            "xai_factor_ref": xai_ref,
        })
        idx += 1

    add(
        "risk_communication",
        "Retrieve peer-reviewed evidence on how to communicate "
        f"{risk_label} depression/anxiety screening results to college students in a supportive, "
        "non-diagnostic, non-stigmatizing way, including what experts should emphasize in follow-up conversations.",
        ["screening communication", "college students", "non-diagnostic", "referral", "counseling"],
    )

    for factor in card.get("top_positive_factors", [])[:5]:
        name = factor.get("display_name", factor.get("feature", ""))
        feat = factor.get("feature", "")
        level = _level_label(levels, feat)
        contrib = factor.get("contribution", "")
        add(
            f"xai_factor_{feat}",
            f"Retrieve survey-based and epidemiological evidence linking {name} "
            f"(feature level: {level}) to depression/anxiety screening indicators, "
            f"psychological distress, or wellbeing in college student populations. "
            f"Include evidence on associated stressors and non-clinical support strategies "
            f"relevant when this factor is a strong model contributor (contribution score ~{contrib}).",
            [name, feat, level, "college students", "screening risk", "wellbeing", "support"],
            xai_ref=feat,
        )

    if card.get("top_protective_factors"):
        prot_parts = []
        for f in card["top_protective_factors"][:3]:
            pname = f.get("display_name", f.get("feature", ""))
            pfeat = f.get("feature", "")
            prot_parts.append(f"{pname} (level {_level_label(levels, pfeat)})")
        prot_names = ", ".join(prot_parts)
        add(
            "protective_factors",
            f"Retrieve evidence on protective factors ({prot_names}) that buffer "
            "depression/anxiety screening risk in college students, and how experts can "
            "reinforce these strengths in non-clinical support conversations.",
            ["protective factors", "resilience", "flourishing", "social support", "college students"],
        )

    add(
        "expert_support_pathways",
        "Retrieve evidence on campus mental-health support pathways for college students "
        "with elevated screening indicators: counseling referral, academic advising, "
        "financial/basic-needs resources, peer support, and stepped-care models — "
        "described as non-clinical guidance for professionals, not treatment protocols.",
        ["counseling referral", "campus support", "stepped care", "student services", "wellbeing"],
    )

    add(
        "safety_non_diagnostic",
        "Retrieve evidence on ethical communication, limitations of survey-based screening tools, "
        "avoiding over-interpretation of model scores, and reducing stigma when discussing "
        "mental-health screening with students.",
        ["screening limitations", "ethics", "non-diagnostic", "stigma", "false positives"],
    )

    cal = str(card.get("calibration_status", ""))
    if _needs_uncertainty_slot(prob, cal):
        add(
            "uncertainty_limitations",
            f"Retrieve evidence on interpreting borderline or uncertain screening probabilities "
            f"(probability ~{prob:.2f}) in student mental-health surveys, calibration limits, "
            "and how experts should communicate ambiguity without alarming or dismissing students.",
            ["uncertainty", "calibration", "borderline", "probability interpretation"],
        )

    return slots


def slots_to_query_plan(
    case_id: str,
    slots: list[dict],
    formatted_case: dict | None = None,
    objectives: list[str] | None = None,
) -> dict:
    internal = []
    for s in slots:
        st = s["slot_type"]
        if st.startswith("xai_factor_"):
            internal.append(
                f"What does the literature say about factor '{s.get('xai_factor_ref', st)}' "
                f"and student screening risk, and what non-clinical supports are documented?"
            )
        elif st == "expert_support_pathways":
            internal.append(
                "What campus support and referral pathways should an expert consider for this profile?"
            )
        elif st == "protective_factors":
            internal.append("What protective factors should the expert reinforce in recommendations?")
        else:
            internal.append(f"What evidence supports the report section for slot '{st}'?")

    default_objectives = [
        "Help psychological experts identify the main model contributors to this screening status",
        "Explain each contributor using survey feature levels and XAI scores (association, not causation)",
        "Provide precise, evidence-grounded non-clinical recommendations the expert can follow with the student",
        "Communicate screening probability clearly without implying diagnosis",
        "Recommend appropriate counseling or student-support referral when risk is elevated",
    ]
    if formatted_case and formatted_case.get("risk_category") == "not_elevated_screening_risk":
        default_objectives.append("Acknowledge lower screening signal while noting limitations and ongoing wellbeing")

    return {
        "case_id": case_id,
        "internal_questions": internal,
        "retrieval_queries": slots,
        "recommendation_objectives": objectives or default_objectives,
    }
