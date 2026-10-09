"""Safety rules and claim verification (llama3 for claim check only)."""

from __future__ import annotations

import json

import pandas as pd

from local_agentic_xai_rag.config_loader import cfg_get
from local_agentic_xai_rag.llm.ollama_client import structured_json
from local_agentic_xai_rag.prompts import FAITHFULNESS_SYSTEM
from local_agentic_xai_rag.retrieval.base import load_chunks
from local_agentic_xai_rag.safety.clinical_boundary import (
    clinical_boundary_flags,
    has_disclaimer,
    has_referral,
)


def narrative_text(narrative: dict) -> str:
    def _s(v) -> str:
        if isinstance(v, dict):
            return json.dumps(v)
        if isinstance(v, list):
            return json.dumps(v)
        return str(v or "")

    parts = [
        _s(narrative.get("screening_summary")),
        _s(narrative.get("model_explanation")),
        _s(narrative.get("evidence_grounding")),
        _s(narrative.get("recommended_next_step")),
        _s(narrative.get("limitations")),
    ]
    return " ".join(parts)


def validate_safety(narrative: dict, elevated: bool = True) -> dict:
    text = narrative_text(narrative)
    flags = clinical_boundary_flags(text)
    comments = []
    checks = {}
    for key, hits in flags.items():
        checks[key] = "fail" if hits else "pass"
        if hits:
            comments.append(f"{key}: matched {hits}")
    checks["missing_disclaimer"] = "fail" if not has_disclaimer(text) else "pass"
    if checks["missing_disclaimer"] == "fail":
        comments.append("Missing non-diagnostic disclaimer")
    checks["missing_referral"] = "fail" if not has_referral(text, elevated) else "pass"
    if checks["missing_referral"] == "fail":
        comments.append("Missing referral for elevated risk")
    fail = [k for k, v in checks.items() if v == "fail"]
    return {
        "case_id": narrative.get("case_id", ""),
        **checks,
        "overall_safety": "pass" if not fail else "fail",
        "revision_required": bool(fail),
        "comments": comments,
    }


def verify_claim(case_id: str, claim: str, chunk_id: str, chunk_text: str) -> dict:
    user = f"Claim: {claim}\n\nCited chunk [{chunk_id}]:\n{chunk_text[:1500]}"
    result = structured_json(FAITHFULNESS_SYSTEM, user, temperature=0.0)
    return {
        "case_id": case_id,
        "claim": claim,
        "cited_chunk_id": chunk_id,
        "support_status": result.get("support_status", "unsupported"),
        "explanation": result.get("explanation", ""),
    }


def verify_all_citations(narrative: dict) -> pd.DataFrame:
    chunks = load_chunks().set_index("chunk_id")["text"].to_dict()
    rows = []
    cid = narrative.get("case_id", "")
    citations = narrative.get("citations", [])
    if not isinstance(citations, list):
        return pd.DataFrame(rows)
    for cite in citations:
        if isinstance(cite, str):
            continue
        if not isinstance(cite, dict):
            continue
        chunk_id = cite.get("chunk_id", "")
        claim = cite.get("claim_supported", "")
        if not chunk_id or not claim:
            continue
        text = chunks.get(chunk_id, "")
        rows.append(verify_claim(cid, claim, chunk_id, text))
    return pd.DataFrame(rows)
