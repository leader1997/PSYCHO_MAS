"""BM25 retriever."""

from __future__ import annotations

import re

import pandas as pd
from rank_bm25 import BM25Okapi

from controlled_agentic_xai_rag.retrieval.base import BaseRetriever, RetrievalHit, load_chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    name = "bm25"

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
                    bm25_score=float(scores[i]),
                    title=str(row.get("title", "")),
                    page=int(row.get("page", 0) or 0),
                    text=str(row["text"]),
                    method=self.name,
                    doc_id=str(row.get("doc_id", "")),
                    expertise_area=str(row.get("expertise_area", "")),
                    section_heading=str(row.get("section_heading", "")),
                )
            )
        return hits
