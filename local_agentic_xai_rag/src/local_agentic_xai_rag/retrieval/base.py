"""Shared retrieval types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd

from local_agentic_xai_rag.config_loader import path


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
    phase1_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_chunks() -> pd.DataFrame:
    csv_path = path("processed_chunks_csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase 2 corpus missing: {csv_path}")
    return pd.read_csv(csv_path)


class BaseRetriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        ...
