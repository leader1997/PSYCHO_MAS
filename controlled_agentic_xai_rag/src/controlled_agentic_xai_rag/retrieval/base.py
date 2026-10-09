"""Shared retrieval types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd

from controlled_agentic_xai_rag.config_loader import path


@dataclass
class RetrievalHit:
    chunk_id: str
    rank: int
    score: float
    title: str
    page: int
    text: str
    method: str
    doc_id: str = ""
    expertise_area: str = ""
    section_heading: str = ""
    query_id: str = ""
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def load_chunks() -> pd.DataFrame:
    csv_path = path("processed_chunks_csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Corpus missing: {csv_path}")
    return pd.read_csv(csv_path)


class BaseRetriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        ...
