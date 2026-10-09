"""Step 1 — query plan validation and slot repair."""

from __future__ import annotations

from controlled_agentic_xai_rag.safety.clinical_boundary import has_unsafe_query_wording
from controlled_agentic_xai_rag.schemas import validate
from controlled_agentic_xai_rag.tools.query_templates import build_template_slots, slots_to_query_plan


MANDATORY_SLOT_TYPES = {"risk_communication", "safety_non_diagnostic"}


def validate_query_plan(
    query_plan: dict,
    card: dict,
    formatted_case: dict,
) -> dict:
    """Validate and repair query plan; return audit record."""
    case_id = query_plan.get("case_id", card.get("case_id", ""))
    errors: list[str] = []
    repairs: list[str] = []

    ok, schema_errors = validate(query_plan, "query_plan.json")
    if not ok:
        errors.extend(schema_errors)

    queries = list(query_plan.get("retrieval_queries") or [])
    slot_types = {q.get("slot_type") for q in queries if isinstance(q, dict)}

    for mandatory in MANDATORY_SLOT_TYPES:
        if mandatory not in slot_types:
            errors.append(f"Missing mandatory slot: {mandatory}")

    xai_features = {f.get("feature") for f in card.get("top_positive_factors", [])}
    covered = {q.get("xai_factor_ref") for q in queries if q.get("xai_factor_ref")}
    missing_factors = xai_features - covered
    if missing_factors:
        errors.append(f"Missing XAI factor queries: {missing_factors}")

    unsafe_queries = []
    for q in queries:
        text = q.get("query_text", "")
        if has_unsafe_query_wording(text):
            unsafe_queries.append(q.get("query_id", ""))
            errors.append(f"Unsafe query wording: {q.get('query_id')}")
    if unsafe_queries:
        repairs.append("Removed unsafe queries for regeneration")

    repaired_plan = dict(query_plan)
    if errors:
        template_slots = build_template_slots(case_id, card, formatted_case)
        existing_by_slot = {q.get("slot_type"): q for q in queries}
        merged = []
        for slot in template_slots:
            st = slot["slot_type"]
            if st in existing_by_slot and existing_by_slot[st].get("query_id") not in unsafe_queries:
                if not has_unsafe_query_wording(existing_by_slot[st].get("query_text", "")):
                    merged.append(existing_by_slot[st])
                    continue
            merged.append(slot)
            repairs.append(f"Regenerated slot: {st}")
        repaired_plan = slots_to_query_plan(
            case_id,
            merged,
            formatted_case,
            query_plan.get("recommendation_objectives"),
        )

    ok_final, _ = validate(repaired_plan, "query_plan.json")
    final_queries = repaired_plan.get("retrieval_queries") or []
    final_slots = {q.get("slot_type") for q in final_queries if isinstance(q, dict)}
    final_covered = {q.get("xai_factor_ref") for q in final_queries if q.get("xai_factor_ref")}
    final_unsafe = [
        q.get("query_id", "")
        for q in final_queries
        if isinstance(q, dict) and has_unsafe_query_wording(q.get("query_text", ""))
    ]
    passed = (
        ok_final
        and MANDATORY_SLOT_TYPES <= final_slots
        and not (xai_features - final_covered)
        and not final_unsafe
    )

    return {
        "case_id": case_id,
        "passed": passed,
        "errors": errors,
        "repairs": repairs,
        "query_plan": repaired_plan,
    }
