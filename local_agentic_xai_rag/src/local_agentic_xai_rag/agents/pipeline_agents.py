"""Two-agent pipeline steps (llama3 via Ollama)."""

from __future__ import annotations

import json

from local_agentic_xai_rag.config_loader import cfg_get
from local_agentic_xai_rag.llm.ollama_client import structured_json
from local_agentic_xai_rag.prompts import (
    ADVISORY_REPORT_SYSTEM,
    EVIDENCE_SYNTHESIS_SYSTEM,
    QUERY_PLANNING_SYSTEM,
    REVISION_SYSTEM,
)
from local_agentic_xai_rag.retrieval.base import RetrievalHit


def _probability_level(probability: float, threshold: float, band: float) -> str:
    if probability >= threshold + band:
        return "high_above_threshold"
    if probability <= threshold - band:
        return "below_threshold"
    return "near_threshold"


def run_query_planning_agent(
    case_id: str,
    case_summary: str,
    card: dict,
    profiles: dict,
) -> dict:
    threshold = float(cfg_get("pipeline", "risk_threshold", default=0.5))
    band = float(cfg_get("pipeline", "near_threshold_band", default=0.1))
    probability = float(card.get("prediction_probability", 0.5))
    prof = profiles.get(case_id, {})
    user = f"""De-identified case summary: {case_summary}

Prediction probability: {probability}
Risk threshold: {threshold}
Probability level hint: {_probability_level(probability, threshold, band)}
Calibration status: {card.get('calibration_status', 'unknown')}
Model name: {card.get('model_name', 'unknown')}

Feature levels (de-identified):
{json.dumps(prof.get('feature_levels', {}), indent=2)}

XAI card:
{json.dumps(card, indent=2)}

Generate query plan JSON for case_id={case_id}. Include retrieval queries for risk communication,
each top positive XAI factor, each protective factor, safety/non-diagnostic communication,
and model limitations if needed."""
    result = structured_json(QUERY_PLANNING_SYSTEM, user, temperature=0.1)
    result["case_id"] = case_id
    return result


def run_evidence_agent(
    case_id: str,
    case_summary: str,
    xai_card: dict,
    hits: list[RetrievalHit],
) -> dict:
    chunks_block = "\n\n".join(
        f"[{h.chunk_id}] ({h.title}, p.{h.page}): {h.text[:600]}" for h in hits
    )
    user = f"""Case summary: {case_summary}

XAI card:
{json.dumps(xai_card, indent=2)}

Retrieved chunks:
{chunks_block}

Synthesize supported claims linked to chunk IDs. case_id: {case_id}

Return JSON with keys: case_id, supported_claims, unsupported_or_uncertain_points, clinical_boundary"""
    result = structured_json(EVIDENCE_SYNTHESIS_SYSTEM, user, temperature=0.1)
    result["case_id"] = case_id
    result["_retrieved_chunk_ids"] = [h.chunk_id for h in hits]
    return result


def run_report_agent(
    case_id: str,
    case_summary: str,
    xai_card: dict,
    evidence: dict,
    revision_comments: list[str] | None = None,
) -> dict:
    system = ADVISORY_REPORT_SYSTEM if not revision_comments else REVISION_SYSTEM
    user = f"""Case summary: {case_summary}

XAI card:
{json.dumps(xai_card, indent=2)}

Verified evidence:
{json.dumps(evidence, indent=2)}

Generate advisory report JSON with: case_id, screening_summary, model_explanation, evidence_grounding, recommended_next_step, limitations, citations, safety_status"""
    if revision_comments:
        user += "\n\nFix these issues:\n" + "\n".join(revision_comments)
    result = structured_json(system, user, temperature=0.1)
    result["case_id"] = case_id
    # Coerce text fields to strings
    for key in ("screening_summary", "model_explanation", "evidence_grounding", "recommended_next_step", "limitations", "safety_status"):
        if key in result and not isinstance(result[key], str):
            result[key] = json.dumps(result[key]) if isinstance(result[key], (dict, list)) else str(result[key])
    # Normalize citations from evidence if missing
    if not result.get("citations") and evidence.get("supported_claims"):
        result["citations"] = [
            {
                "chunk_id": (c.get("supporting_chunk_ids") or [""])[0],
                "claim_supported": c.get("claim", ""),
            }
            for c in evidence["supported_claims"]
            if isinstance(c, dict) and c.get("supporting_chunk_ids")
        ]
    disclaimer = "This is a model-based screening signal and not a clinical diagnosis."
    if disclaimer.lower() not in json.dumps(result).lower():
        lim = result.get("limitations", "")
        result["limitations"] = f"{lim} {disclaimer}".strip()
    return result
