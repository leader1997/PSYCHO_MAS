"""Dense retrieval with ChromaDB + Ollama embeddings."""

from __future__ import annotations

import json

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

COLLECTION_NAME = "corpus_chunks"
CHROMA_ADD_BATCH = 100


class ChromaDenseRetriever(BaseRetriever):
    name = "chroma_dense"

    def __init__(self, chunks: pd.DataFrame | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.index_dir = path("chroma_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.index_dir / "model.txt"
        self.ids_path = self.index_dir / "chunk_ids.json"
        self._collection = None
        from controlled_agentic_xai_rag.retrieval.index_sync import ensure_vector_indexes_current

        ensure_vector_indexes_current()
        self._build()

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms

    def _client(self):
        try:
            import chromadb
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "ChromaDB not installed. Run: pip install chromadb"
            ) from exc
        return chromadb.PersistentClient(path=str(self.index_dir))

    def _cache_valid(self, ids: list[str], embed_model: str) -> bool:
        if not (self.meta_path.exists() and self.ids_path.exists()):
            return False
        if self.meta_path.read_text(encoding="utf-8").strip() != embed_model:
            return False
        try:
            stored = json.loads(self.ids_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return stored == ids

    def _row_metadata(self, row: pd.Series) -> dict:
        return {
            "doc_id": str(row.get("doc_id", "")),
            "title": str(row.get("title", ""))[:500],
            "page": int(row.get("page", 0) or 0),
            "expertise_area": str(row.get("expertise_area", ""))[:500],
            "section_heading": str(row.get("section_heading", ""))[:200],
        }

    def _build(self) -> None:
        ids = self.chunks["chunk_id"].astype(str).tolist()
        texts = self.chunks["text"].astype(str).tolist()
        embed_model = get_embed_model()

        client = self._client()
        if self._cache_valid(ids, embed_model):
            self._collection = client.get_collection(COLLECTION_NAME)
            if self._collection.count() == len(ids):
                return

        print(f"Building Chroma index with {embed_model!r} ({len(texts)} chunks)...", flush=True)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = int(cfg_get("retrieval", "embed_batch_size", default=32))
        embed_batch = max(batch_size, 1)
        for start in range(0, len(texts), embed_batch):
            end = min(start + embed_batch, len(texts))
            batch_ids = ids[start:end]
            batch_texts = [
                format_for_embedding(t, model=embed_model, role="document")
                for t in texts[start:end]
            ]
            try:
                vecs = embed(batch_texts, batch_size=embed_batch, model=embed_model)
            except (EmbeddingUnavailableError, requests.RequestException) as exc:
                raise EmbeddingUnavailableError(str(exc)) from exc

            embs = self._normalize(np.array(vecs, dtype=np.float32))
            metadatas = [self._row_metadata(self.chunks.iloc[start + i]) for i in range(end - start)]

            for add_start in range(0, len(batch_ids), CHROMA_ADD_BATCH):
                add_end = min(add_start + CHROMA_ADD_BATCH, len(batch_ids))
                sl = slice(add_start, add_end)
                collection.add(
                    ids=batch_ids[sl],
                    embeddings=embs[sl].tolist(),
                    documents=batch_texts[sl],
                    metadatas=metadatas[sl],
                )

            if (start // embed_batch) % 10 == 0:
                print(f"  embedded {end}/{len(texts)}", flush=True)

        self.meta_path.write_text(embed_model, encoding="utf-8")
        self.ids_path.write_text(json.dumps(ids), encoding="utf-8")
        self._collection = collection
        from controlled_agentic_xai_rag.retrieval.index_sync import mark_index_built

        mark_index_built(self.index_dir)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        embed_model = get_embed_model()
        q_text = format_for_embedding(query, model=embed_model, role="query")
        q_vec = self._normalize(np.array([embed([q_text], model=embed_model)[0]], dtype=np.float32))

        n = min(k, self._collection.count())
        if n <= 0:
            return []

        result = self._collection.query(
            query_embeddings=q_vec.tolist(),
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[RetrievalHit] = []
        result_ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        for rank, cid in enumerate(result_ids, start=1):
            meta = metadatas[rank - 1] if rank - 1 < len(metadatas) else {}
            dist = float(distances[rank - 1]) if rank - 1 < len(distances) else 0.0
            dense_score = max(0.0, 1.0 - dist)
            text = documents[rank - 1] if rank - 1 < len(documents) else ""
            hits.append(
                RetrievalHit(
                    chunk_id=cid,
                    rank=rank,
                    score=dense_score,
                    dense_score=dense_score,
                    title=str(meta.get("title", "")),
                    page=int(meta.get("page", 0) or 0),
                    text=str(text),
                    method=self.name,
                    doc_id=str(meta.get("doc_id", "")),
                    expertise_area=str(meta.get("expertise_area", "")),
                    section_heading=str(meta.get("section_heading", "")),
                )
            )
        return hits
