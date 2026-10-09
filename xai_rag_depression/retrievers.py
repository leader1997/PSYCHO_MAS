"""Three retrieval backends for RAGAS retriever ablation."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "outputs" / "kb"

RETRIEVER_SPECS = [
    ("dense_minilm", "Dense embeddings (all-MiniLM-L6-v2, cosine)"),
    ("bm25", "Sparse lexical (BM25Okapi)"),
    ("hybrid_rrf", "Hybrid (reciprocal rank fusion: dense + BM25)"),
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BaseRetriever(ABC):
    name: str
    label: str

    def __init__(self, docs: list[dict]):
        self.docs = docs

    @abstractmethod
    def retrieve(self, query: str, k: int = 8) -> list[dict]:
        ...


class DenseMiniLMRetriever(BaseRetriever):
    def __init__(self, docs: list[dict], embeddings: np.ndarray):
        super().__init__(docs)
        self.name = "dense_minilm"
        self.label = RETRIEVER_SPECS[0][1]
        self.embeddings = embeddings
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, k: int = 8) -> list[dict]:
        q = self._embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        sims = self.embeddings @ q
        idx = np.argsort(sims)[::-1][:k]
        return [self.docs[i] for i in idx]


class BM25Retriever(BaseRetriever):
    def __init__(self, docs: list[dict]):
        super().__init__(docs)
        self.name = "bm25"
        self.label = RETRIEVER_SPECS[1][1]
        from rank_bm25 import BM25Okapi

        corpus = [_tokenize(f"{d.get('title', '')} {d['text']}") for d in docs]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, k: int = 8) -> list[dict]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:k]
        return [self.docs[i] for i in idx]


class HybridRRFRetriever(BaseRetriever):
    """Reciprocal rank fusion of dense MiniLM + BM25."""

    def __init__(self, docs: list[dict], embeddings: np.ndarray, rrf_k: int = 60):
        super().__init__(docs)
        self.name = "hybrid_rrf"
        self.label = RETRIEVER_SPECS[2][1]
        self._dense = DenseMiniLMRetriever(docs, embeddings)
        self._bm25 = BM25Retriever(docs)
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 8) -> list[dict]:
        n = min(len(self.docs), max(k * 4, 30))
        dense_hits = self._dense.retrieve(query, k=n)
        bm25_hits = self._bm25.retrieve(query, k=n)

        def ranks(hits: list[dict]) -> dict[str, int]:
            return {d["id"]: i for i, d in enumerate(hits)}

        dense_r = ranks(dense_hits)
        bm25_r = ranks(bm25_hits)
        doc_ids = list({d["id"] for d in dense_hits} | {d["id"] for d in bm25_hits})
        id_to_doc = {d["id"]: d for d in self.docs}

        fused: list[tuple[float, str]] = []
        for doc_id in doc_ids:
            score = 0.0
            if doc_id in dense_r:
                score += 1.0 / (self._rrf_k + dense_r[doc_id] + 1)
            if doc_id in bm25_r:
                score += 1.0 / (self._rrf_k + bm25_r[doc_id] + 1)
            fused.append((score, doc_id))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [id_to_doc[doc_id] for _, doc_id in fused[:k]]


def load_documents() -> list[dict]:
    with open(KB_DIR / "documents.json", encoding="utf-8") as f:
        return json.load(f)


def get_retriever(name: str) -> BaseRetriever:
    docs = load_documents()
    embeddings = np.load(KB_DIR / "embeddings.npy")
    if name == "dense_minilm":
        return DenseMiniLMRetriever(docs, embeddings)
    if name == "bm25":
        return BM25Retriever(docs)
    if name == "hybrid_rrf":
        return HybridRRFRetriever(docs, embeddings)
    raise ValueError(f"Unknown retriever: {name}. Choose: {[s[0] for s in RETRIEVER_SPECS]}")


def list_retrievers() -> list[tuple[str, str]]:
    return list(RETRIEVER_SPECS)
