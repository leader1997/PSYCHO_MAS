"""Thin Ollama client for embeddings and contextual text generation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import requests

from ollama_contextual_eval.config import cfg


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        embed_model: str | None = None,
        context_model: str | None = None,
        delay: float | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or cfg("ollama", "base_url", default="http://localhost:11434")).rstrip("/")
        self.embed_model = embed_model or cfg("ollama", "embed_model", default="nomic-embed-text")
        self.context_model = context_model or cfg("ollama", "context_model", default="qwen2.5:latest")
        self.delay = delay if delay is not None else float(cfg("ollama", "inter_call_delay_seconds", default=0.25))
        self.timeout = timeout or int(cfg("ollama", "request_timeout_seconds", default=600))

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        r = requests.post(f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def embed_one(self, text: str) -> np.ndarray:
        payload = {"model": self.embed_model, "input": text}
        try:
            data = self._post("/api/embed", payload)
            emb = data["embeddings"][0]
        except (KeyError, requests.HTTPError):
            data = self._post("/api/embeddings", {"model": self.embed_model, "prompt": text})
            emb = data["embedding"]
        return np.asarray(emb, dtype=np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": self.embed_model, "input": batch}
            try:
                data = self._post("/api/embed", payload)
                embs = data["embeddings"]
            except (KeyError, requests.HTTPError):
                embs = [self.embed_one(t) for t in batch]
            else:
                embs = [np.asarray(e, dtype=np.float32) for e in embs]
            vectors.extend(embs)
        return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)

    def generate_context(self, document_text: str, chunk_text: str) -> str:
        """Anthropic cookbook-style situating context (Ollama chat)."""
        prompt = f"""<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        data = self._post(
            "/api/chat",
            {
                "model": self.context_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        content = data.get("message", {}).get("content", "") or data.get("response", "")
        return " ".join(str(content).split())[: int(cfg("contextual", "max_context_tokens_approx", default=120) * 6)]
