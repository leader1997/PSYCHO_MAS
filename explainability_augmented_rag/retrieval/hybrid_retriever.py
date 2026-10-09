"""Hybrid RRF retrieval (R3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get  # noqa: E402
from retrieval.base import BaseRetriever, RetrievalHit, load_chunks  # noqa: E402
from retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from retrieval.dense_retriever import DenseRetriever  # noqa: E402


class HybridRetriever(BaseRetriever):
    name = "hybrid"
    label = "Reciprocal rank fusion (dense + BM25)"

    def __init__(self, chunks: pd.DataFrame | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.dense = DenseRetriever(self.chunks)
        self.bm25 = BM25Retriever(self.chunks)
        self.rrf_k = cfg_get("retrieval", "rrf_k", default=60)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        pool_k = max(k * 3, cfg_get("retrieval", "hybrid_top_k", default=30))
        d_hits = self.dense.retrieve(query, k=pool_k)
        b_hits = self.bm25.retrieve(query, k=pool_k)
        scores: dict[str, float] = {}
        meta: dict[str, RetrievalHit] = {}
        for hits in (d_hits, b_hits):
            for hit in hits:
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0) + 1 / (
                    self.rrf_k + hit.rank
                )
                meta[hit.chunk_id] = hit
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
