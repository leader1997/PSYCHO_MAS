"""Hybrid BM25 + llama3 dense (RRF)."""

from __future__ import annotations

from local_agentic_xai_rag.config_loader import cfg_get
from local_agentic_xai_rag.retrieval.base import BaseRetriever, RetrievalHit, load_chunks
from local_agentic_xai_rag.retrieval.bm25 import BM25Retriever
from local_agentic_xai_rag.retrieval.dense_llama3 import Llama3DenseRetriever


class HybridRetriever(BaseRetriever):
    name = "hybrid_llama3"

    def __init__(self):
        chunks = load_chunks()
        self.bm25 = BM25Retriever(chunks)
        self.dense = Llama3DenseRetriever(chunks)
        self.rrf_k = cfg_get("retrieval", "rrf_k", default=60)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        pool = max(k * 3, 20)
        scores: dict[str, float] = {}
        meta: dict[str, RetrievalHit] = {}
        for hits in (self.dense.retrieve(query, k=pool), self.bm25.retrieve(query, k=pool)):
            for h in hits:
                scores[h.chunk_id] = scores.get(h.chunk_id, 0) + 1 / (self.rrf_k + h.rank)
                meta[h.chunk_id] = h
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        out = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            base = meta[cid]
            out.append(
                RetrievalHit(
                    chunk_id=cid,
                    rank=rank,
                    score=score,
                    title=base.title,
                    page=base.page,
                    text=base.text,
                    method=self.name,
                    doc_id=base.doc_id,
                    phase1_id=base.phase1_id,
                )
            )
        return out


def get_retriever(bm25_only: bool = False) -> BaseRetriever:
    if bm25_only or cfg_get("retrieval", "bm25_only", default=False):
        return BM25Retriever()
    return HybridRetriever()
