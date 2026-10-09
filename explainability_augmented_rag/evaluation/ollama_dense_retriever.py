"""FAISS dense retrieval with Ollama embeddings only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.ollama_client_local import EMBED_MODEL, embed
from generation.config_loader import path
from retrieval.base import BaseRetriever, RetrievalHit, load_chunks


class OllamaDenseRetriever(BaseRetriever):
    name = "ollama_dense"
    label = f"Ollama dense ({EMBED_MODEL}) + FAISS"

    def __init__(self, chunks: pd.DataFrame | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.index_dir = path("faiss_index") / "ollama_nomic"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embed_path = self.index_dir / "embeddings.npy"
        self.meta_path = self.index_dir / "chunk_ids.json"
        self.model_path = self.index_dir / "model.txt"
        self._index = None
        self._build()

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms

    def _build(self) -> None:
        ids = self.chunks["chunk_id"].tolist()
        texts = self.chunks["text"].tolist()
        if (
            self.embed_path.exists()
            and self.meta_path.exists()
            and self.model_path.read_text(encoding="utf-8").strip() == EMBED_MODEL
        ):
            stored = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if stored == ids:
                embs = np.load(self.embed_path)
                self._index = faiss.IndexFlatIP(embs.shape[1])
                self._index.add(embs)
                return
        print(f"Building Ollama FAISS index ({EMBED_MODEL}, {len(texts)} chunks)...", flush=True)
        batch = 64
        vecs = []
        for i in range(0, len(texts), batch):
            vecs.extend(embed(texts[i : i + batch], batch_size=batch))
            if (i // batch) % 5 == 0:
                print(f"  embedded {min(i+batch, len(texts))}/{len(texts)}", flush=True)
        embs = self._normalize(np.array(vecs, dtype=np.float32))
        np.save(self.embed_path, embs)
        self.meta_path.write_text(json.dumps(ids), encoding="utf-8")
        self.model_path.write_text(EMBED_MODEL, encoding="utf-8")
        self._index = faiss.IndexFlatIP(embs.shape[1])
        self._index.add(embs)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        q = self._normalize(np.array(embed([query]), dtype=np.float32))
        scores, idx = self._index.search(q, min(k, len(self.chunks)))
        hits = []
        for rank, (i, score) in enumerate(zip(idx[0], scores[0]), start=1):
            if i < 0:
                continue
            row = self.chunks.iloc[int(i)]
            hits.append(
                RetrievalHit(
                    chunk_id=row["chunk_id"],
                    rank=rank,
                    score=float(score),
                    title=str(row.get("title", "")),
                    page=int(row.get("page", 0) or 0),
                    text=str(row["text"]),
                    method=self.name,
                    doc_id=str(row.get("doc_id", "")),
                    phase1_id=str(row.get("phase1_id", "")),
                )
            )
        return hits
