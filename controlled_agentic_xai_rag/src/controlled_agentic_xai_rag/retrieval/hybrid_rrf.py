"""Hybrid BM25 + dense vector store (FAISS or ChromaDB, RRF)."""

from __future__ import annotations

from controlled_agentic_xai_rag.config_loader import cfg_get
from controlled_agentic_xai_rag.llm.ollama_client import EmbeddingUnavailableError
from controlled_agentic_xai_rag.retrieval.base import BaseRetriever, RetrievalHit, load_chunks
from controlled_agentic_xai_rag.retrieval.bm25 import BM25Retriever
from controlled_agentic_xai_rag.retrieval.dense_factory import get_dense_retriever, hybrid_retriever_name


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self):
        self.name = hybrid_retriever_name()
        chunks = load_chunks()
        self.bm25 = BM25Retriever(chunks)
        self.dense: BaseRetriever | None = None
        self.dense_unavailable_reason = ""
        try:
            self.dense = get_dense_retriever(chunks)
        except EmbeddingUnavailableError as exc:
            self.dense_unavailable_reason = str(exc)
            print(
                f"Dense retrieval unavailable; using BM25 only. {self.dense_unavailable_reason}",
                flush=True,
            )
        self.rrf_k = cfg_get("retrieval", "rrf_k", default=60)

    @property
    def using_dense(self) -> bool:
        return self.dense is not None

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        if not self.dense:
            return self.bm25.retrieve(query, k=k)

        pool = max(k * 3, 20)
        try:
            dense_hits = self.dense.retrieve(query, k=pool)
        except EmbeddingUnavailableError:
            return self.bm25.retrieve(query, k=k)

        scores: dict[str, float] = {}
        bm25_map: dict[str, float] = {}
        dense_map: dict[str, float] = {}
        meta: dict[str, RetrievalHit] = {}
        for hits, is_dense in ((dense_hits, True), (self.bm25.retrieve(query, k=pool), False)):
            for h in hits:
                scores[h.chunk_id] = scores.get(h.chunk_id, 0) + 1 / (self.rrf_k + h.rank)
                if is_dense:
                    dense_map[h.chunk_id] = h.dense_score or h.score
                else:
                    bm25_map[h.chunk_id] = h.bm25_score or h.score
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
                    rrf_score=score,
                    bm25_score=bm25_map.get(cid, 0.0),
                    dense_score=dense_map.get(cid, 0.0),
                    title=base.title,
                    page=base.page,
                    text=base.text,
                    method=self.name,
                    doc_id=base.doc_id,
                    expertise_area=base.expertise_area,
                    section_heading=base.section_heading,
                )
            )
        return out


def get_retriever(bm25_only: bool = False) -> BaseRetriever:
    if bm25_only or cfg_get("retrieval", "bm25_only", default=False):
        return BM25Retriever()
    return HybridRetriever()
