"""In-memory dense, BM25, and hybrid (RRF) retrieval on contextualized texts."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from ollama_contextual_eval.config import cfg
from ollama_contextual_eval.ollama_client import OllamaClient


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class ContextualIndex:
    def __init__(
        self,
        chunks: pd.DataFrame,
        indexed_texts: list[str],
        embeddings: np.ndarray,
        embed_model: str,
    ):
        if len(indexed_texts) != len(chunks):
            raise ValueError("indexed_texts length must match chunks")
        self.chunks = chunks.reset_index(drop=True)
        self.indexed_texts = indexed_texts
        self.embeddings = embeddings
        self.embed_model = embed_model
        self.chunk_ids = self.chunks["chunk_id"].astype(str).tolist()
        self.bm25 = BM25Okapi([_tokenize(t) for t in indexed_texts])
        norms = np.linalg.norm(embeddings, axis=1)
        norms[norms == 0] = 1.0
        self._emb_norms = norms

    @classmethod
    def build(
        cls,
        chunks: pd.DataFrame,
        indexed_texts: list[str],
        client: OllamaClient | None = None,
        embeddings: np.ndarray | None = None,
        batch_size: int = 16,
    ) -> ContextualIndex:
        client = client or OllamaClient()
        if embeddings is None:
            embeddings = client.embed_batch(indexed_texts, batch_size=batch_size)
        return cls(chunks, indexed_texts, embeddings, client.embed_model)

    def _dense_scores(self, query: str, client: OllamaClient) -> np.ndarray:
        q = client.embed_one(query)
        qn = np.linalg.norm(q) or 1.0
        return (self.embeddings @ q) / (self._emb_norms * qn)

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float64)

    def _rrf_fuse(self, dense_order: list[int], bm25_order: list[int], k: int, rrf_k: int) -> list[int]:
        scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_order[:k], start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(bm25_order[:k], start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        return sorted(scores, key=lambda i: scores[i], reverse=True)[:k]

    def retrieve(self, query: str, k: int, mode: str, client: OllamaClient | None = None) -> list[str]:
        client = client or OllamaClient()
        rrf_k = int(cfg("eval", "rrf_k", default=60))
        pool = max(k * 4, 40)

        if mode == "dense":
            scores = self._dense_scores(query, client)
            order = np.argsort(-scores)[:k]
            return [self.chunk_ids[i] for i in order]

        bm25_scores = self._bm25_scores(query)
        bm25_order = list(np.argsort(-bm25_scores)[:pool])

        if mode == "bm25":
            return [self.chunk_ids[i] for i in bm25_order[:k]]

        if mode == "hybrid":
            dense_scores = self._dense_scores(query, client)
            dense_order = list(np.argsort(-dense_scores)[:pool])
            fused = self._rrf_fuse(dense_order, bm25_order, pool, rrf_k)
            return [self.chunk_ids[i] for i in fused[:k]]

        raise ValueError(f"Unknown mode: {mode}")
