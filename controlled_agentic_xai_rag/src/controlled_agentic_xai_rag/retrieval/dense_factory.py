"""Dense vector retriever factory — FAISS only."""

from __future__ import annotations

import pandas as pd

from controlled_agentic_xai_rag.retrieval.base import BaseRetriever


def get_dense_retriever(chunks: pd.DataFrame) -> BaseRetriever:
    from controlled_agentic_xai_rag.retrieval.dense_llama3 import Llama3DenseRetriever

    return Llama3DenseRetriever(chunks)


def hybrid_retriever_name() -> str:
    return "hybrid_faiss"
