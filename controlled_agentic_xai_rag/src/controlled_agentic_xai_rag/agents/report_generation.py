"""Agent 2 — non-clinical screening report generation."""

from __future__ import annotations

import json

from controlled_agentic_xai_rag.config_loader import cfg_get
from controlled_agentic_xai_rag.llm.ollama_client import structured_json
from controlled_agentic_xai_rag.retrieval.base import RetrievalHit

REPORT_SYSTEM = """You generate detailed, expert-facing non-clinical screening advisory reports for psychological professionals reviewing college student survey results.

Audience: counselors, student-support staff, and researchers — NOT the student directly.

Core principles:
1. SCREENING NOT DIAGNOSIS: The model probability is a screening signal from survey data, never a clinical diagnosis.
2. CONTRIBUTORS NOT CAUSES: XAI factors statistically contributed to the model score; describe them as "associated contributors" or "drivers in the model", never as proven causes of disorder.
3. EVIDENCE-GROUNDED: Every factual claim about student populations or risk factors must cite a retrieved chunk_id in the citations array.
4. ACTIONABLE FOR EXPERTS: Recommendations must be specific steps a professional can take (conversation focus, referrals, campus resources, monitoring) — not vague encouragement.
5. SAFETY: No treatment plans, medication advice, dosage, or crisis-management instructions. For elevated risk, include referral to qualified counselors/student support.

Writing standards (each narrative field must be substantive prose, NOT nested JSON objects):
- screening_summary: 3–5 sentences on overall status, model confidence, and what the expert should prioritize.
- contributor_analysis: For EACH top XAI contributor, write a short paragraph covering: (a) survey feature level, (b) contribution to model score, (c) what literature suggests about this factor in students, (d) what to explore in conversation. Rank by contribution strength.
- probability_interpretation: 2–4 sentences explaining the numeric probability, threshold, calibration, and how to communicate uncertainty.
- xai_explanation: Synthesize how factors combined to produce the model output; use contributor language, not causal claims.
- evidence_grounded_interpretation: 2–3 paragraphs linking retrieved literature to this specific profile pattern.
- actionable_recommendations: Numbered list (at least 4 items) of precise non-clinical steps: who to involve, what to discuss, which supports to offer, what to monitor, follow-up timing.
- non_clinical_recommendation: Executive summary (2–3 sentences) synthesizing actionable_recommendations for this profile — never a standalone generic referral sentence.
- protective_factors_note: If protective factors exist, explain how they lowered risk and how to reinforce them; if none, state that and suggest strength-building where appropriate.
- limitations: Screening tool limits, model limits, retrieval limits, mandatory disclaimer.

Output valid JSON only with string values for all narrative fields (no nested JSON strings)."""

NARRATIVE_FIELDS = (
    "screening_summary",
    "contributor_analysis",
    "probability_interpretation",
    "xai_explanation",
    "evidence_grounded_interpretation",
    "actionable_recommendations",
    "non_clinical_recommendation",
    "protective_factors_note",
    "limitations",
)

REPORT_JSON_FIELDS = (
    "case_id",
    *NARRATIVE_FIELDS,
    "citations",
    "safety_status",
)


def _chunk_limit() -> int:
    return int(cfg_get("retrieval", "report_chunk_chars", default=1200))


def _hits_context(hits: list[RetrievalHit]) -> str:
    limit = _chunk_limit()
    blocks = []
    for h in hits:
        blocks.append(
            f"[{h.chunk_id}] title={h.title} page={h.page} expertise={h.expertise_area}\n"
            f"{h.text[:limit]}"
        )
    return "\n\n".join(blocks) if blocks else "(No evidence retrieved — base recommendations on case data only.)"


def _contributor_brief(formatted_case: dict) -> str:
    lines = []
    levels = formatted_case.get("feature_levels") or {}
    contributors = formatted_case.get("top_xai_contributors") or []
    sorted_c = sorted(contributors, key=lambda x: float(x.get("contribution", 0) or 0), reverse=True)
    for i, f in enumerate(sorted_c, start=1):
        feat = f.get("feature", "")
        lines.append(
            f"{i}. {f.get('display_name', feat)} ({feat}): "
            f"contribution={f.get('contribution')}, "
            f"survey_level={levels.get(feat, 'unknown')}, "
            f"direction={f.get('direction', '')}"
        )
    return "\n".join(lines) if lines else "No positive contributors listed."


def _protective_brief(formatted_case: dict) -> str:
    factors = formatted_case.get("protective_factors") or []
    if not factors:
        return "No protective factors identified by the model for this profile."
    levels = formatted_case.get("feature_levels") or {}
    lines = []
    for f in factors:
        feat = f.get("feature", "")
        lines.append(
            f"- {f.get('display_name', feat)} ({feat}): "
            f"contribution={f.get('contribution')}, level={levels.get(feat, 'unknown')}"
        )
    return "\n".join(lines)


def _to_prose(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return _to_prose(json.loads(text))
            except json.JSONDecodeError:
                return text
        return text
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                title = item.get("title") or item.get("factor") or item.get("name") or item.get("feature", "")
                body = item.get("text") or item.get("description") or item.get("recommendation") or item.get("detail", "")
                if title and body:
                    parts.append(f"**{title}**: {body}")
                elif title:
                    parts.append(f"- {title}: {json.dumps({k: v for k, v in item.items() if k != 'title'})}")
                else:
                    parts.append(_to_prose(item))
            else:
                parts.append(f"- {_to_prose(item)}")
        return "\n".join(parts)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            label = k.replace("_", " ").strip().title()
            parts.append(f"**{label}**: {_to_prose(v)}")
        return "\n\n".join(parts)
    return str(value)


def _is_boilerplate_recommendation(text: str, referral: str = "") -> bool:
    """True when text is empty or only the canned config referral / JSON echo."""
    if not text or not str(text).strip():
        return True
    normalized = " ".join(str(text).lower().split())
    if len(normalized) < 25:
        return True
    if referral:
        ref_norm = " ".join(referral.lower().split())
        if normalized == ref_norm or (ref_norm in normalized and len(normalized) <= len(ref_norm) + 40):
            return True
    boilerplate_markers = (
        "consider speaking with a qualified counselor",
        "elevated_risk_referral",
        "mandatory_disclaimer",
    )
    if any(m in normalized for m in boilerplate_markers) and len(normalized) < 220:
        return True
    return False


def _executive_summary_from_actions(actions: str, max_chars: int = 500) -> str:
    """Derive a short expert summary from numbered actionable steps."""
    lines = [ln.strip() for ln in actions.splitlines() if ln.strip()]
    if not lines:
        return actions[:max_chars].strip()
    picked = []
    for ln in lines:
        if ln[0].isdigit() or ln.startswith("-") or ln.startswith("•"):
            picked.append(ln)
        if len("\n".join(picked)) >= max_chars:
            break
    summary = "\n".join(picked[:4]) if picked else actions[:max_chars]
    return summary.strip()


def _query_plan_section(query_plan: dict | None) -> str:
    if not query_plan:
        return ""
    questions = query_plan.get("report_questions") or query_plan.get("internal_questions") or []
    objectives = query_plan.get("recommendation_objectives") or []
    retrieval = query_plan.get("retrieval_queries") or []
    lines = ["=== QUESTIONS TO ANSWER IN THE REPORT (from query planning agent) ==="]
    for i, q in enumerate(questions, start=1):
        lines.append(f"{i}. {q}")
    if objectives:
        lines.append("\n=== REPORT OBJECTIVES ===")
        for o in objectives:
            lines.append(f"- {o}")
    if retrieval:
        lines.append("\n=== RETRIEVAL QUERIES USED (for evidence alignment) ===")
        for rq in retrieval:
            if isinstance(rq, dict):
                lines.append(f"- [{rq.get('query_id', '')}] {rq.get('query_text', '')[:300]}")
    lines.append(
        "\nYou MUST address every report question above in the narrative fields. "
        "Use evidence_grounded_interpretation and contributor_analysis to answer XAI-related questions; "
        "use actionable_recommendations for support/pathway questions."
    )
    return "\n".join(lines) + "\n\n"


def _build_user_prompt(
    formatted_case: dict,
    card: dict,
    hits: list[RetrievalHit],
    disclaimer: str,
    elevated: bool,
    query_plan: dict | None = None,
    revision_comments: list[str] | None = None,
) -> str:
    elevated_note = (
        "This case is elevated: actionable_recommendations MUST list specific support/referral "
        "steps for this profile (counseling, academic, financial, social) in your own words — "
        "do NOT paste a generic one-line referral."
        if elevated
        else "Recommendations should focus on wellbeing monitoring and strengths-based support."
    )
    prompt = (
        "=== EXPERT ADVISORY REPORT REQUEST ===\n\n"
        f"Case ID: {formatted_case['case_id']}\n"
        f"Elevated screening risk: {elevated}\n"
        f"Prediction probability: {formatted_case['prediction_probability']}\n"
        f"Risk threshold: {formatted_case.get('risk_threshold', 0.5)}\n"
        f"Risk category: {formatted_case['risk_category']}\n"
        f"Model: {formatted_case.get('model_name', 'unknown')}\n"
        f"Calibration: {formatted_case.get('calibration_status', 'unknown')}\n\n"
        "=== MAIN MODEL CONTRIBUTORS (ranked — explain each in contributor_analysis) ===\n"
        f"{_contributor_brief(formatted_case)}\n\n"
        "=== PROTECTIVE FACTORS ===\n"
        f"{_protective_brief(formatted_case)}\n\n"
        f"Formatted case (full):\n{json.dumps(formatted_case, indent=2)}\n\n"
        f"XAI card (full):\n{json.dumps(card, indent=2)}\n\n"
        f"Retrieved evidence ({len(hits)} chunks):\n{_hits_context(hits)}\n\n"
        f"{_query_plan_section(query_plan)}"
        f"Mandatory disclaimer (include in limitations only): {disclaimer}\n"
        f"{elevated_note}\n\n"
        "=== REQUIRED JSON FIELDS ===\n"
        + ", ".join(REPORT_JSON_FIELDS)
        + "\n\n"
        "Citations: provide at least one citation per major contributor discussed; "
        "each entry must be {chunk_id, claim_text} where claim_text is a specific supported statement.\n"
        "actionable_recommendations: numbered steps (1. 2. 3. ...) tailored to THIS profile's contributors and levels.\n"
        "non_clinical_recommendation: 2–3 sentence executive summary of actionable_recommendations — never a generic referral line alone."
    )
    if revision_comments:
        prompt += "\n\n=== REVISION REQUIRED (address every item) ===\n"
        prompt += "\n".join(f"- {c}" for c in revision_comments)
    return prompt


def _normalize_report(report: dict, formatted_case: dict, disclaimer: str, elevated: bool) -> dict:
    referral = cfg_get("safety", "elevated_risk_referral", default="")
    report["case_id"] = formatted_case["case_id"]
    for key in NARRATIVE_FIELDS:
        if key in report:
            report[key] = _to_prose(report.get(key))
        else:
            report.setdefault(key, "")

    if not report.get("contributor_analysis") and report.get("xai_explanation"):
        report["contributor_analysis"] = report["xai_explanation"]

    actions = report.get("actionable_recommendations", "")
    summary = report.get("non_clinical_recommendation", "")

    if _is_boilerplate_recommendation(actions, referral):
        actions = ""
        report["actionable_recommendations"] = ""

    if _is_boilerplate_recommendation(summary, referral):
        summary = ""

    if not actions and not _is_boilerplate_recommendation(report.get("non_clinical_recommendation", ""), referral):
        actions = report.get("non_clinical_recommendation", "")

    if actions and _is_boilerplate_recommendation(summary, referral):
        report["non_clinical_recommendation"] = _executive_summary_from_actions(actions)
    elif not summary and actions:
        report["non_clinical_recommendation"] = _executive_summary_from_actions(actions)
    else:
        report["non_clinical_recommendation"] = summary

    if not report.get("actionable_recommendations") and actions:
        report["actionable_recommendations"] = actions

    if not report.get("protective_factors_note"):
        prot = formatted_case.get("protective_factors") or []
        report["protective_factors_note"] = (
            _protective_brief(formatted_case) if prot else "No protective factors were identified by the model."
        )

    full_text = json.dumps(report)
    if disclaimer and disclaimer not in full_text:
        report["limitations"] = f"{report.get('limitations', '')} {disclaimer}".strip()

    report.setdefault("safety_status", "draft_unvalidated")
    report.setdefault("citations", [])
    return report


def run_report_generation_agent(
    formatted_case: dict,
    card: dict,
    hits: list[RetrievalHit],
    query_plan: dict | None = None,
    revision_comments: list[str] | None = None,
) -> dict:
    disclaimer = cfg_get("safety", "mandatory_disclaimer", default="")
    elevated = formatted_case["risk_category"] == "elevated_screening_risk"

    user = _build_user_prompt(
        formatted_case, card, hits, disclaimer, elevated, query_plan, revision_comments
    )
    report = structured_json(REPORT_SYSTEM, user)
    return _normalize_report(report, formatted_case, disclaimer, elevated)
