"""Normalize llama3 JSON outputs to pipeline schema."""

from __future__ import annotations

import json
import re
from typing import Any


def normalize_query_plan(raw: dict, case_id: str, card: dict) -> dict:
    from local_agentic_xai_rag.config_loader import cfg_get

    threshold = float(cfg_get("pipeline", "risk_threshold", default=0.5))
    band = float(cfg_get("pipeline", "near_threshold_band", default=0.1))
    probability = float(card.get("prediction_probability", 0.5))

    def _level() -> str:
        if probability >= threshold + band:
            return "high_above_threshold"
        if probability <= threshold - band:
            return "below_threshold"
        return "near_threshold"

    internal = []
    for q in raw.get("internal_questions") or []:
        if isinstance(q, dict):
            internal.append({"question": q.get("question", ""), "purpose": q.get("purpose", "")})
        elif isinstance(q, str):
            internal.append({"question": q, "purpose": "evidence gathering"})

    retrieval = []
    for rq in raw.get("retrieval_queries") or []:
        if not isinstance(rq, dict):
            continue
        keywords = rq.get("focus_keywords") or rq.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        retrieval.append(
            {
                "query_type": rq.get("query_type", "general"),
                "query_text": rq.get("query_text", rq.get("query", "")),
                "focus_keywords": keywords,
                "related_xai_factor": rq.get("related_xai_factor", rq.get("xai_factor", "")),
            }
        )

    objectives = raw.get("recommendation_objectives") or []
    if isinstance(objectives, str):
        objectives = [objectives]

    summary = raw.get("safe_case_summary") or raw.get("case_summary") or ""
    if isinstance(summary, dict):
        summary = json.dumps(summary)

    return {
        "case_id": case_id,
        "safe_case_summary": str(summary),
        "probability_level": raw.get("probability_level") or _level(),
        "prediction_probability": probability,
        "risk_threshold": threshold,
        "calibration_status": card.get("calibration_status", "unknown"),
        "internal_questions": internal,
        "retrieval_queries": [r for r in retrieval if r["query_text"]],
        "recommendation_objectives": [str(o) for o in objectives],
    }


def _first_chunk_id(raw: Any, valid_ids: set[str]) -> str:
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, str):
        return ""
    # Direct match
    if raw in valid_ids:
        return raw
    # Extract DOC/ART id prefix from verbose strings
    m = re.search(r"((?:DOC|ART)\d+_[A-Za-z0-9_]+)", raw)
    if m:
        prefix = m.group(1)
        for vid in valid_ids:
            if vid.startswith(prefix[:20]) or prefix in vid:
                return vid
    return ""


def normalize_evidence(raw: dict, case_id: str, valid_chunk_ids: list[str]) -> dict:
    valid = set(valid_chunk_ids)
    claims = []
    for c in raw.get("supported_claims") or []:
        if not isinstance(c, dict):
            continue
        ids_raw = c.get("supporting_chunk_ids") or c.get("chunk_IDs") or c.get("chunk_ids") or []
        ids = []
        for item in (ids_raw if isinstance(ids_raw, list) else [ids_raw]):
            cid = _first_chunk_id(item, valid)
            if cid:
                ids.append(cid)
        if not ids and valid_chunk_ids:
            ids = [valid_chunk_ids[0]]
        claims.append(
            {
                "claim": c.get("claim", ""),
                "supporting_chunk_ids": ids,
                "evidence_strength": c.get("evidence_strength", "moderate"),
                "relevance_to_xai": c.get("relevance_to_xai", ""),
            }
        )
    uncertain = []
    for u in raw.get("unsupported_or_uncertain_points") or []:
        if isinstance(u, dict):
            uncertain.append({"point": u.get("point", str(u)), "reason": u.get("reason", "insufficient evidence")})
    boundary = raw.get("clinical_boundary", "")
    if isinstance(boundary, dict):
        boundary = boundary.get("note", json.dumps(boundary))
    return {
        "case_id": case_id,
        "supported_claims": claims,
        "unsupported_or_uncertain_points": uncertain,
        "clinical_boundary": str(boundary),
        "_retrieved_chunk_ids": valid_chunk_ids,
    }


def _flatten_field(v: Any) -> str:
    if isinstance(v, str):
        # If it's JSON string, try to extract readable text
        if v.startswith("{") or v.startswith("["):
            try:
                obj = json.loads(v)
                if isinstance(obj, dict):
                    if "note" in obj:
                        return str(obj["note"])
                    if "type" in obj and "probability" in obj:
                        return f"Elevated screening risk (model probability {obj.get('probability')})."
                    if "status" in obj:
                        return str(obj["status"])
                    return json.dumps(obj)
                return json.dumps(obj)
            except json.JSONDecodeError:
                pass
        return v
    if isinstance(v, dict):
        if "note" in v:
            return str(v["note"])
        return json.dumps(v)
    return str(v or "")


def normalize_report(raw: dict, case_id: str, evidence: dict) -> dict:
    report = {
        "case_id": case_id,
        "screening_summary": _flatten_field(raw.get("screening_summary", "")),
        "model_explanation": _flatten_field(raw.get("model_explanation", "")),
        "evidence_grounding": _flatten_field(raw.get("evidence_grounding", "")),
        "recommended_next_step": _flatten_field(raw.get("recommended_next_step", "")),
        "limitations": _flatten_field(raw.get("limitations", "")),
        "safety_status": _flatten_field(raw.get("safety_status", "pending review")),
        "citations": [],
    }
    disclaimer = "This is a model-based screening signal and not a clinical diagnosis."
    if disclaimer.lower() not in report["limitations"].lower():
        report["limitations"] = f"{report['limitations']} {disclaimer}".strip()

    refs = cfg_get_referral()
    if refs and refs.lower() not in report["recommended_next_step"].lower():
        report["recommended_next_step"] = f"{report['recommended_next_step']} {refs}".strip()

    citations = raw.get("citations") or []
    if isinstance(citations, list) and citations and isinstance(citations[0], dict) and citations[0].get("claim_supported"):
        report["citations"] = citations
    else:
        for c in evidence.get("supported_claims") or []:
            ids = c.get("supporting_chunk_ids") or []
            if ids:
                report["citations"].append({"chunk_id": ids[0], "claim_supported": c.get("claim", "")})
    return report


def cfg_get_referral() -> str:
    from local_agentic_xai_rag.config_loader import cfg_get

    return cfg_get("safety", "elevated_risk_referral", default="")
