"""Template and agent-planned retrieval queries."""

from __future__ import annotations


def _factor_names(factors: list[dict], n: int = 3) -> str:
    return ", ".join(f.get("display_name", f.get("feature", "")) for f in factors[:n])


def build_case_queries(case_id: str, card: dict) -> list[dict]:
    pos = card.get("top_positive_factors", [])
    prot = card.get("top_protective_factors", [])
    pos_names = _factor_names(pos) or "survey features"
    prot_names = _factor_names(prot) or "protective survey features"

    return [
        {
            "case_id": case_id,
            "query_type": "risk_communication",
            "query_text": (
                "Retrieve evidence about how elevated depression/anxiety screening risk should be "
                "communicated in a non-diagnostic, non-clinical way to college students."
            ),
        },
        {
            "case_id": case_id,
            "query_type": "feature_specific",
            "query_text": (
                f"Retrieve evidence about how {pos_names} are associated with student "
                f"depression/anxiety screening risk in survey-based studies."
            ),
        },
        {
            "case_id": case_id,
            "query_type": "protective_factor",
            "query_text": (
                f"Retrieve evidence about protective factors such as {prot_names} "
                f"and student mental health in college populations."
            ),
        },
        {
            "case_id": case_id,
            "query_type": "safety_communication",
            "query_text": (
                "Retrieve evidence about safe, non-stigmatizing, non-clinical mental-health "
                "communication and ethical limitations of screening tools."
            ),
        },
    ]


def queries_from_plan(case_id: str, query_plan: dict) -> list[dict]:
    """Convert query-planning agent output to hybrid_retrieve query rows."""
    rows = []
    for rq in query_plan.get("retrieval_queries") or []:
        if not isinstance(rq, dict):
            continue
        text = rq.get("query_text", "").strip()
        if not text:
            continue
        keywords = rq.get("focus_keywords") or []
        if keywords:
            text = f"{text} Focus keywords: {', '.join(keywords)}."
        rows.append(
            {
                "case_id": case_id,
                "query_type": rq.get("query_type", "agent_planned"),
                "query_text": text,
                "related_xai_factor": rq.get("related_xai_factor", ""),
                "source": "query_planning_agent",
            }
        )
    return rows


def resolve_case_queries(case_id: str, card: dict, query_plan: dict | None) -> list[dict]:
    if query_plan and query_plan.get("retrieval_queries"):
        planned = queries_from_plan(case_id, query_plan)
        if planned:
            return planned
    return build_case_queries(case_id, card)
