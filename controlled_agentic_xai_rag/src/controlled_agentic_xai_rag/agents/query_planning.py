"""Agent 1 — LLM query planning: report questions + hybrid retrieval queries from XAI context."""

from __future__ import annotations

import json

from controlled_agentic_xai_rag.llm.ollama_client import structured_json
from controlled_agentic_xai_rag.tools.query_templates import build_template_slots, slots_to_query_plan

QUERY_PLANNING_SYSTEM = """You are an XAI-aware planning assistant for a non-clinical college mental-health screening RAG system.
Your audience is psychological experts reviewing model-based screening outputs.

Primary task: propose a structured list of **report_questions** — expert-oriented questions that the downstream
report agent must answer in detail using retrieved literature and the case XAI context.
These questions are the backbone of the recommendation report.

Secondary task: refine **retrieval_queries** so hybrid search finds evidence to answer those questions.
Each retrieval query must stay non-diagnostic (no "diagnose", "prescribe", "treat").

Use the CatBoost/SHAP XAI context (contributors, feature levels, probability) to make questions and queries
specific to THIS student profile. Contributors are statistical associations in the model, NOT proven causes.

Output JSON:
- case_id
- report_questions: array of 8–14 specific questions the report must answer (strings)
- internal_questions: same as report_questions or sub-questions per theme (for schema compatibility)
- retrieval_queries: array with query_id, slot_type, query_text, focus_keywords, xai_factor_ref
- recommendation_objectives: what the expert should be able to do after reading the report

Keep every mandatory slot_type from the template; do NOT remove slots."""


def _contributor_summary(formatted_case: dict) -> str:
    lines = []
    levels = formatted_case.get("feature_levels") or {}
    for f in formatted_case.get("top_xai_contributors") or []:
        feat = f.get("feature", "")
        name = f.get("display_name", feat)
        contrib = f.get("contribution", "")
        level = levels.get(feat, "unknown")
        lines.append(f"- {name} ({feat}): contribution={contrib}, survey level={level}")
    return "\n".join(lines) if lines else "- none listed"


def run_query_planning_agent(
    formatted_case: dict,
    card: dict,
    templates_only: bool = False,
) -> dict:
    case_id = formatted_case["case_id"]
    template_slots = build_template_slots(case_id, card, formatted_case)
    base_plan = slots_to_query_plan(case_id, template_slots, formatted_case)
    base_questions = base_plan.get("internal_questions") or []

    if templates_only:
        return {
            **base_plan,
            "report_questions": list(base_questions),
        }

    user = (
        f"Case ID: {case_id}\n"
        f"Prediction probability: {formatted_case['prediction_probability']}\n"
        f"Risk threshold: {formatted_case.get('risk_threshold', 0.5)}\n"
        f"Risk category: {formatted_case['risk_category']}\n"
        f"Calibration: {formatted_case.get('calibration_status', 'unknown')}\n\n"
        f"Top model contributors (XAI — associations, NOT causes):\n"
        f"{_contributor_summary(formatted_case)}\n\n"
        f"Protective factors:\n"
        f"{json.dumps(formatted_case.get('protective_factors', []), indent=2)}\n\n"
        f"Feature levels:\n"
        f"{json.dumps(dict(list((formatted_case.get('feature_levels') or {}).items())[:20]), indent=2)}\n\n"
        f"XAI card:\n{json.dumps(card, indent=2)}\n\n"
        f"Template plan (keep ALL slot_types; refine query_text for this profile):\n"
        f"{json.dumps(base_plan, indent=2)}\n\n"
        "Produce report_questions that cover: screening interpretation, each major XAI contributor, "
        "protective factors, evidence-based supports, campus pathways, communication ethics, and limitations. "
        "Align retrieval_queries so hybrid search retrieves literature to answer those questions."
    )
    refined = structured_json(QUERY_PLANNING_SYSTEM, user)
    if not refined.get("retrieval_queries") and not refined.get("report_questions"):
        return {**base_plan, "report_questions": list(base_questions)}

    report_questions = refined.get("report_questions") or refined.get("internal_questions") or base_questions
    if isinstance(report_questions, str):
        report_questions = [report_questions]

    template_by_slot = {s["slot_type"]: s for s in template_slots}
    merged_queries = []
    for rq in refined.get("retrieval_queries", []):
        if not isinstance(rq, dict):
            continue
        st = rq.get("slot_type", "")
        base = template_by_slot.get(st, {})
        merged_queries.append({
            "query_id": rq.get("query_id") or base.get("query_id", f"{case_id}_q"),
            "slot_type": st or base.get("slot_type", "unknown"),
            "query_text": rq.get("query_text") or base.get("query_text", ""),
            "focus_keywords": rq.get("focus_keywords") or base.get("focus_keywords", []),
            "xai_factor_ref": rq.get("xai_factor_ref") or base.get("xai_factor_ref", ""),
        })
    for st, base in template_by_slot.items():
        if st not in {q["slot_type"] for q in merged_queries}:
            merged_queries.append(base)

    if not merged_queries:
        merged_queries = base_plan.get("retrieval_queries", [])

    return {
        "case_id": case_id,
        "report_questions": list(report_questions),
        "internal_questions": list(report_questions),
        "retrieval_queries": merged_queries,
        "recommendation_objectives": refined.get("recommendation_objectives")
        or base_plan.get("recommendation_objectives", []),
    }
