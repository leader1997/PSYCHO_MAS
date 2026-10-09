"""Dense retrieval with Ollama llama3 embeddings."""

from __future__ import annotations

import json

import faiss
import numpy as np
import pandas as pd
import requests

from controlled_agentic_xai_rag.config_loader import cfg_get, path
from controlled_agentic_xai_rag.llm.ollama_client import (
    EmbeddingUnavailableError,
    embed,
    format_for_embedding,
    get_embed_model,
)
from controlled_agentic_xai_rag.retrieval.base import BaseRetriever, RetrievalHit, load_chunks


class Llama3DenseRetriever(BaseRetriever):
    name = "llama3_dense"

    def __init__(self, chunks: pd.DataFrame | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.index_dir = path("faiss_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embed_path = self.index_dir / "embeddings.npy"
        self.meta_path = self.index_dir / "chunk_ids.json"
        self.model_path = self.index_dir / "model.txt"
        self._index = None
        from controlled_agentic_xai_rag.retrieval.index_sync import ensure_vector_indexes_current

        ensure_vector_indexes_current()
        self._build()

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms

    def _build(self) -> None:
        ids = self.chunks["chunk_id"].tolist()
        texts = self.chunks["text"].tolist()
        embed_model = get_embed_model()
        if (
            self.embed_path.exists()
            and self.meta_path.exists()
            and self.model_path.exists()
            and self.model_path.read_text(encoding="utf-8").strip() == embed_model
        ):
            stored = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if stored == ids:
                embs = np.load(self.embed_path)
                self._index = faiss.IndexFlatIP(embs.shape[1])
                self._index.add(embs)
                return
        print(f"Building FAISS index with {embed_model!r} ({len(texts)} chunks)...", flush=True)
        batch = cfg_get("retrieval", "embed_batch_size", default=32)
        vecs = []
        try:
            for i in range(0, len(texts), batch):
                batch_texts = [
                    format_for_embedding(t, model=embed_model, role="document")
                    for t in texts[i : i + batch]
                ]
                vecs.extend(embed(batch_texts, batch_size=batch, model=embed_model))
                if (i // batch) % 10 == 0:
                    print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
        except (EmbeddingUnavailableError, requests.RequestException) as exc:
            raise EmbeddingUnavailableError(str(exc)) from exc
        embs = self._normalize(np.array(vecs, dtype=np.float32))
        np.save(self.embed_path, embs)
        self.meta_path.write_text(json.dumps(ids), encoding="utf-8")
        self.model_path.write_text(embed_model, encoding="utf-8")
        self._index = faiss.IndexFlatIP(embs.shape[1])
        self._index.add(embs)
        from controlled_agentic_xai_rag.retrieval.index_sync import mark_index_built

        mark_index_built(self.index_dir)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        embed_model = get_embed_model()
        q_text = format_for_embedding(query, model=embed_model, role="query")
        q = self._normalize(np.array([embed([q_text], model=embed_model)[0]], dtype=np.float32))
        scores, idx = self._index.search(q, min(k, len(self.chunks)))
        hits = []
        for rank, i in enumerate(idx[0], start=1):
            if i < 0:
                continue
            row = self.chunks.iloc[int(i)]
            hits.append(
                RetrievalHit(
                    chunk_id=row["chunk_id"],
                    rank=rank,
                    score=float(scores[0][rank - 1]),
                    dense_score=float(scores[0][rank - 1]),
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
