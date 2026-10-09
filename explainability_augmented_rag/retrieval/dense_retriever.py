"""OpenAI dense retrieval with FAISS index (R1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get, load_dotenv, path  # noqa: E402
from retrieval.base import BaseRetriever, RetrievalHit, load_chunks  # noqa: E402


class DenseRetriever(BaseRetriever):
    name = "dense"
    label = "Dense OpenAI embeddings + FAISS"

    def __init__(self, chunks: pd.DataFrame | None = None):
        load_dotenv()
        self.chunks = chunks if chunks is not None else load_chunks()
        self.index_dir = path("faiss_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embed_path = self.index_dir / "dense_embeddings.npy"
        self.meta_path = self.index_dir / "chunk_ids.json"
        self._index: faiss.IndexFlatIP | None = None
        self._embeddings: np.ndarray | None = None
        self._use_local: bool = False
        self._build_index()

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        import os

        if self._use_local or not os.environ.get("OPENAI_API_KEY"):
            if not self._use_local:
                print("  Using local MiniLM embeddings (no OPENAI_API_KEY)", flush=True)
                self._use_local = True
            return self._embed_local(texts)
        try:
            from openai import OpenAI

            client = OpenAI()
            model = cfg_get("retrieval", "embedding_model", default="text-embedding-3-small")
            batch = 100
            vecs = []
            for i in range(0, len(texts), batch):
                batch_texts = texts[i : i + batch]
                resp = client.embeddings.create(model=model, input=batch_texts)
                vecs.extend([d.embedding for d in resp.data])
            arr = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return arr / norms
        except Exception as e:
            print(f"  OpenAI embeddings failed ({e}); using local MiniLM", flush=True)
            self._use_local = True
            return self._embed_local(texts)

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        arr = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(arr, dtype=np.float32)

    def _build_index(self) -> None:
        ids = self.chunks["chunk_id"].tolist()
        texts = self.chunks["text"].tolist()
        if self.embed_path.exists() and self.meta_path.exists():
            stored_ids = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if stored_ids == ids:
                self._embeddings = np.load(self.embed_path)
                dim = self._embeddings.shape[1]
                self._index = faiss.IndexFlatIP(dim)
                self._index.add(self._embeddings)
                return
        print("Building FAISS dense index (OpenAI embeddings)...", flush=True)
        self._embeddings = self._embed_openai(texts)
        np.save(self.embed_path, self._embeddings)
        self.meta_path.write_text(json.dumps(ids), encoding="utf-8")
        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)

    def retrieve(self, query: str, k: int = 8) -> list[RetrievalHit]:
        assert self._index is not None and self._embeddings is not None
        q = self._embed_openai([query])[0]
        scores, idx = self._index.search(q.reshape(1, -1), min(k, len(self.chunks)))
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
