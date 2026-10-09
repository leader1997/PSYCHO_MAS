"""Cross-encoder reranker on hybrid candidates (R4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get  # noqa: E402
from retrieval.base import BaseRetriever, RetrievalHit, load_chunks  # noqa: E402
from retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


class HybridRerankRetriever(BaseRetriever):
    name = "hybrid_rerank"
    label = "Hybrid + cross-encoder reranker"

    def __init__(self, chunks: pd.DataFrame | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.hybrid = HybridRetriever(self.chunks)
        self.model_name = cfg_get(
            "retrieval", "cross_encoder_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        pool_k = cfg_get("retrieval", "hybrid_top_k", default=30)
        candidates = self.hybrid.retrieve(query, k=pool_k)
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self._get_model().predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))[:k]
        out = []
        for rank, (hit, score) in enumerate(ranked, start=1):
            out.append(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    rank=rank,
                    score=float(score),
                    title=hit.title,
                    page=hit.page,
                    text=hit.text,
                    method=self.name,
                    doc_id=hit.doc_id,
                    phase1_id=hit.phase1_id,
                )
            )
        return out


def get_retriever(name: str, chunks: pd.DataFrame | None = None) -> BaseRetriever:
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.dense_retriever import DenseRetriever

    mapping = {
        "dense": DenseRetriever,
        "bm25": BM25Retriever,
        "hybrid": HybridRetriever,
        "hybrid_rerank": HybridRerankRetriever,
    }
    if name not in mapping:
        raise ValueError(f"Unknown retriever: {name}")
    return mapping[name](chunks)
