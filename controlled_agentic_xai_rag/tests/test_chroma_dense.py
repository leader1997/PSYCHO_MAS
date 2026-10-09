"""Chroma dense retriever smoke test (mocked embeddings)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_chunks():
    return pd.DataFrame(
        [
            {
                "chunk_id": "DOC_A_p1_c1",
                "doc_id": "DOC_A",
                "title": "Academic stress in students",
                "page": 1,
                "expertise_area": "screening",
                "section_heading": "body",
                "text": "Academic impairment and depression screening in college students.",
            },
            {
                "chunk_id": "DOC_B_p2_c1",
                "doc_id": "DOC_B",
                "title": "Campus belonging",
                "page": 2,
                "expertise_area": "belonging",
                "section_heading": "body",
                "text": "Social belonging buffers loneliness among undergraduates.",
            },
            {
                "chunk_id": "DOC_C_p3_c1",
                "doc_id": "DOC_C",
                "title": "Financial stress",
                "page": 3,
                "expertise_area": "finance",
                "section_heading": "body",
                "text": "Food insecurity and financial worry correlate with distress.",
            },
        ]
    )


def _mock_embed(texts, batch_size=None, model=None):
    dim = 8
    out = []
    for text in texts:
        seed = sum(ord(c) for c in text) % 997
        rng = np.random.default_rng(seed)
        vec = rng.random(dim, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        out.append(vec.tolist())
    return out


@patch("controlled_agentic_xai_rag.retrieval.dense_chroma.embed", side_effect=_mock_embed)
@patch("controlled_agentic_xai_rag.retrieval.dense_chroma.get_embed_model", return_value="mock-embed")
def test_chroma_dense_retrieve(mock_model, mock_embed, tiny_chunks, tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    from controlled_agentic_xai_rag.config_loader import load_config

    load_config.cache_clear()
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "mock-embed")

    import controlled_agentic_xai_rag.retrieval.dense_chroma as chroma_mod

    monkeypatch.setattr(
        chroma_mod,
        "path",
        lambda key: tmp_path if key == "chroma_index" else tmp_path / key,
    )

    retriever = chroma_mod.ChromaDenseRetriever(tiny_chunks)
    hits = retriever.retrieve("academic impairment depression college", k=2)

    assert len(hits) == 2
    assert hits[0].chunk_id
    assert hits[0].dense_score > 0
    assert hits[0].method == "chroma_dense"


def test_hybrid_faiss_name():
    from controlled_agentic_xai_rag.retrieval.dense_factory import hybrid_retriever_name

    assert hybrid_retriever_name() == "hybrid_faiss"
