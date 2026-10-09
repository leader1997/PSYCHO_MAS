"""BM25 sparse retrieval (R2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.base import BaseRetriever, RetrievalHit, load_chunks  # noqa: E402


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    name = "bm25"
    label = "BM25Okapi sparse lexical"

    def __init__(self, chunks: pd.DataFrame | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.corpus = [_tokenize(t) for t in self.chunks["text"].tolist()]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        scores = self.bm25.get_scores(_tokenize(query))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits = []
        for rank, i in enumerate(idx, start=1):
            row = self.chunks.iloc[i]
            hits.append(
                RetrievalHit(
                    chunk_id=row["chunk_id"],
                    rank=rank,
                    score=float(scores[i]),
                    title=str(row.get("title", "")),
                    page=int(row.get("page", 0) or 0),
                    text=str(row["text"]),
                    method=self.name,
                    doc_id=str(row.get("doc_id", "")),
                    phase1_id=str(row.get("phase1_id", "")),
                )
            )
        return hits
