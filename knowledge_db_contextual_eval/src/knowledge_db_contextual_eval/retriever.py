"""BM25 index and hybrid retrieval (0.7 dense + 0.3 BM25, normalized)."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from knowledge_db_contextual_eval.config import cfg

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(text) for text in documents]
        self.doc_lens = np.asarray([len(tokens) for tokens in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 1.0
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n_docs = len(self.doc_tokens)
        self.idf = {term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.doc_tokens), dtype=np.float32)
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[idx] / self.avgdl)
                scores[idx] += idf * (tf * (self.k1 + 1) / denom)
        return scores


def normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-8:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector) or 1.0
    matrix_norm = np.linalg.norm(matrix, axis=1)
    matrix_norm[matrix_norm == 0] = 1.0
    return matrix @ query_vector / (matrix_norm * query_norm)


def hybrid_scores(
    dense: np.ndarray,
    lexical: np.ndarray,
    dense_weight: float | None = None,
) -> np.ndarray:
    dense_weight = dense_weight if dense_weight is not None else float(cfg("eval", "dense_weight", default=0.70))
    return dense_weight * normalize(dense) + (1 - dense_weight) * normalize(lexical)
