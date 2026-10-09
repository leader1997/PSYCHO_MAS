"""Shared retrieval types and chunk loading."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


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
    csv_path = ROOT / "data" / "processed_chunks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run retrieval/build_corpus.py first: {csv_path}")
    return pd.read_csv(csv_path)


def load_chunks_list() -> list[dict]:
    df = load_chunks()
    return df.to_dict(orient="records")


class BaseRetriever(ABC):
    name: str
    label: str

    @abstractmethod
    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        ...
