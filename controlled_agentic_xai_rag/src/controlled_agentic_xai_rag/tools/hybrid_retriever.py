"""Step 2 — hybrid retrieval with audit logging."""

from __future__ import annotations

from controlled_agentic_xai_rag.config_loader import cfg_get
from controlled_agentic_xai_rag.retrieval.base import RetrievalHit
from controlled_agentic_xai_rag.retrieval.hybrid_rrf import get_retriever


def hybrid_retrieve(
    query_plan: dict,
    bm25_only: bool = False,
) -> tuple[list[dict], list[RetrievalHit], dict]:
    retriever = get_retriever(bm25_only=bm25_only)
    retrieval_info = {
        "requested_mode": "bm25_only" if bm25_only else "hybrid",
        "actual_mode": retriever.name,
        "vector_store": "faiss",
        "warning": "",
    }
    if hasattr(retriever, "dense_unavailable_reason") and retriever.dense_unavailable_reason:
        retrieval_info["actual_mode"] = "bm25_only"
        retrieval_info["warning"] = retriever.dense_unavailable_reason
    per_query_k = cfg_get("retrieval", "per_query_k", default=5)
    top_k = cfg_get("retrieval", "top_k", default=8)
    queries = query_plan.get("retrieval_queries") or []
    seen: set[str] = set()
    hits: list[RetrievalHit] = []
    audit_rows: list[dict] = []

    for q in queries:
        if not isinstance(q, dict):
            continue
        query_id = q.get("query_id", "")
        query_text = q.get("query_text", "").strip()
        if not query_text:
            continue
        for h in retriever.retrieve(query_text, k=per_query_k):
            audit_rows.append({
                "query_id": query_id,
                "slot_type": q.get("slot_type", ""),
                "query_text": query_text,
                "chunk_id": h.chunk_id,
                "rank": h.rank,
                "bm25_score": h.bm25_score,
                "dense_score": h.dense_score,
                "rrf_score": h.rrf_score,
                "doc_id": h.doc_id,
                "title": h.title,
                "page": h.page,
                "expertise_area": h.expertise_area,
                "section_heading": h.section_heading,
            })
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                h.query_id = query_id
                hits.append(h)

    return audit_rows, hits[:top_k], retrieval_info
