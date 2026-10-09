"""Deterministic pipeline tools (CrewAI-compatible callables)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from local_agentic_xai_rag.config_loader import cfg_get
from local_agentic_xai_rag.query_templates import resolve_case_queries
from local_agentic_xai_rag.retrieval.base import RetrievalHit
from local_agentic_xai_rag.retrieval.hybrid import get_retriever
from local_agentic_xai_rag.safety.clinical_boundary import (
    clinical_boundary_flags,
    has_disclaimer,
    has_referral,
)
from local_agentic_xai_rag.safety.validator import narrative_text, verify_claim
from local_agentic_xai_rag.retrieval.base import load_chunks


def format_case(case_id: str, card: dict, profiles: dict) -> dict:
    prof = profiles.get(case_id, {})
    levels = prof.get("feature_levels", {})
    return {
        "case_id": case_id,
        "prediction_probability": card.get("prediction_probability"),
        "prediction_label": card.get("prediction_label"),
        "model_name": card.get("model_name", "CatBoost"),
        "feature_levels": dict(list(levels.items())[:12]),
        "summary": (
            f"De-identified student case ({case_id}). "
            f"Feature levels: " + "; ".join(f"{k}: {v}" for k, v in list(levels.items())[:8])
        ),
    }


def hybrid_retrieve(
    case_id: str,
    card: dict,
    bm25_only: bool = False,
    query_plan: dict | None = None,
) -> tuple[list[dict], list[RetrievalHit]]:
    retriever = get_retriever(bm25_only=bm25_only)
    queries = resolve_case_queries(case_id, card, query_plan)
    seen: set[str] = set()
    hits: list[RetrievalHit] = []
    for q in queries:
        for h in retriever.retrieve(q["query_text"], k=5):
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                hits.append(h)
    top_k = cfg_get("retrieval", "top_k", default=8)
    return queries, hits[:top_k]


def safety_rules_check(report: dict, elevated: bool = True) -> dict:
    text = narrative_text(report)
    flags = clinical_boundary_flags(text)
    comments = []
    checks = {k: ("fail" if v else "pass") for k, v in flags.items()}
    for k, st in checks.items():
        if st == "fail":
            comments.append(f"{k}: failed")
    checks["missing_disclaimer"] = "fail" if not has_disclaimer(text) else "pass"
    if checks["missing_disclaimer"] == "fail":
        comments.append("Missing non-diagnostic disclaimer")
    checks["missing_referral"] = "fail" if not has_referral(text, elevated) else "pass"
    if checks["missing_referral"] == "fail":
        comments.append("Missing referral for elevated risk")
    fail = [k for k, v in checks.items() if v == "fail"]
    return {
        "case_id": report.get("case_id", ""),
        **checks,
        "overall_safety": "pass" if not fail else "fail",
        "revision_required": bool(fail),
        "comments": comments,
    }


def citation_check(report: dict) -> pd.DataFrame:
    chunks = load_chunks().set_index("chunk_id")["text"].to_dict()
    rows = []
    cid = report.get("case_id", "")
    for cite in report.get("citations") or []:
        if not isinstance(cite, dict):
            continue
        chunk_id = cite.get("chunk_id", "")
        claim = cite.get("claim_supported", "")
        if chunk_id and claim:
            rows.append(verify_claim(cid, claim, chunk_id, chunks.get(chunk_id, "")))
    return pd.DataFrame(rows)


def filter_unverified_citations(report: dict, claims_df: pd.DataFrame) -> dict:
    """Remove unsupported/contradicted citations from final report (G3 gate)."""
    if claims_df.empty:
        return report
    bad = set(
        claims_df.loc[
            claims_df["support_status"].isin(["unsupported", "contradicted"]),
            ["claim", "cited_chunk_id"],
        ].apply(lambda r: (r["claim"], r["cited_chunk_id"]), axis=1)
    )
    if not bad:
        return report
    out = dict(report)
    kept = [
        c
        for c in (out.get("citations") or [])
        if isinstance(c, dict)
        and (c.get("claim_supported", ""), c.get("chunk_id", "")) not in bad
    ]
    out["citations"] = kept
    return out


def claims_need_revision(claims_df: pd.DataFrame) -> tuple[bool, list[str]]:
    if claims_df.empty:
        return False, []
    bad = claims_df[claims_df["support_status"].isin(["unsupported", "contradicted"])]
    if bad.empty:
        return False, []
    comments = [
        f"Remove or revise unsupported claim: {r.claim[:120]} (chunk {r.cited_chunk_id})"
        for r in bad.itertuples()
    ]
    return True, comments


def compute_retrieval_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return (
        rows.groupby("retriever")[["precision@5", "recall@5", "recall@10", "mrr", "ndcg@10"]]
        .mean()
        .round(4)
        .reset_index()
    )
