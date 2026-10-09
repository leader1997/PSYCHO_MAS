"""Citation and claim faithfulness checker — one LLM call per citation, sequential."""

from __future__ import annotations

import pandas as pd

from controlled_agentic_xai_rag.llm.ollama_client import structured_json
from controlled_agentic_xai_rag.retrieval.base import RetrievalHit, load_chunks

FAITHFULNESS_SYSTEM = """You verify whether an expert-advisory claim is supported by a cited evidence chunk from the mental-health literature.

Rules:
- fully_supported: the chunk directly states or strongly implies the claim
- partially_supported: the chunk is related but does not fully establish the claim (e.g., general population finding applied to a specific factor)
- unsupported: the chunk does not mention the claim topic or cannot justify it
- contradicted: the chunk conflicts with the claim

Flag over-interpretation (causal language, clinical diagnosis) as unsupported even if topic matches.

Output JSON: {"support_status": "fully_supported"|"partially_supported"|"unsupported"|"contradicted", "explanation": "one or two sentences"}"""


def verify_claim(case_id: str, claim: str, chunk_id: str, chunk_text: str) -> dict:
    user = (
        f"Case: {case_id}\n"
        f"Expert advisory claim to verify:\n{claim}\n\n"
        f"Cited chunk ID: {chunk_id}\n"
        f"Evidence text:\n{chunk_text[:2000]}\n\n"
        "Does this chunk support the claim for use in a non-clinical screening advisory report?"
    )
    result = structured_json(FAITHFULNESS_SYSTEM, user)
    status = result.get("support_status", "unsupported")
    if status not in ("fully_supported", "partially_supported", "unsupported", "contradicted"):
        status = "unsupported"
    return {
        "case_id": case_id,
        "claim": claim,
        "cited_chunk_id": chunk_id,
        "support_status": status,
        "explanation": result.get("explanation", ""),
    }


def check_citations(report: dict, hits: list[RetrievalHit]) -> pd.DataFrame:
    chunk_map = {h.chunk_id: h.text for h in hits}
    corpus = load_chunks().set_index("chunk_id")["text"].to_dict()
    rows = []
    cid = report.get("case_id", "")
    for cite in report.get("citations") or []:
        if not isinstance(cite, dict):
            continue
        chunk_id = cite.get("chunk_id", "")
        claim = cite.get("claim_text", cite.get("claim_supported", ""))
        if chunk_id and claim:
            text = chunk_map.get(chunk_id) or corpus.get(chunk_id, "")
            rows.append(verify_claim(cid, claim, chunk_id, text))
    return pd.DataFrame(rows)


def claims_need_revision(claims_df: pd.DataFrame) -> tuple[bool, list[str]]:
    if claims_df.empty:
        return False, []
    bad = claims_df[claims_df["support_status"].isin(["unsupported", "contradicted"])]
    if bad.empty:
        return False, []
    comments = [
        f"Revise or remove unsupported claim: {r.claim[:120]} (chunk {r.cited_chunk_id})"
        for r in bad.itertuples()
    ]
    return True, comments


def filter_unverified_citations(report: dict, claims_df: pd.DataFrame) -> dict:
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
        c for c in (out.get("citations") or [])
        if isinstance(c, dict)
        and (c.get("claim_text", c.get("claim_supported", "")), c.get("chunk_id", "")) not in bad
    ]
    out["citations"] = kept
    return out
