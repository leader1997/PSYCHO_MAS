"""Ollama HTTP client (embed + context generation)."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import requests

from knowledge_db_contextual_eval.config import cfg


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        embed_model: str | None = None,
        context_model: str | None = None,
        timeout: int | None = None,
        embed_retries: int | None = None,
        max_embed_chars: int | None = None,
    ):
        self.base_url = (base_url or cfg("ollama", "base_url", default="http://localhost:11434")).rstrip("/")
        self.embed_model = embed_model or cfg("ollama", "embed_model", default="nomic-embed-text")
        self.context_model = context_model or cfg("ollama", "context_model", default="qwen2.5:latest")
        self.timeout = timeout or int(cfg("ollama", "request_timeout_seconds", default=300))
        self.embed_retries = embed_retries or int(cfg("ollama", "embed_retries", default=3))
        self.max_embed_chars = max_embed_chars or int(cfg("corpus", "max_embed_chars", default=6000))

    def _post(self, endpoint: str, payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.embed_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=timeout or self.timeout,
                )
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                last_error = exc
                print(f"Ollama call failed attempt {attempt}/{self.embed_retries}: {exc}")
                time.sleep(2 * attempt)
        raise last_error  # type: ignore[misc]

    def embed_text(self, text: str) -> np.ndarray:
        for max_chars in [self.max_embed_chars, 3000, 1500, 750]:
            try:
                data = self._post(
                    "/api/embeddings",
                    {"model": self.embed_model, "prompt": text[:max_chars]},
                )
                return np.asarray(data["embedding"], dtype=np.float32)
            except Exception as exc:
                print(f"Embedding failed with max_chars={max_chars}: {exc}")
        raise RuntimeError("Embedding failed after all truncation fallbacks")

    def generate_text(self, prompt: str, timeout: int | None = None) -> str:
        data = self._post(
            "/api/generate",
            {
                "model": self.context_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=timeout or self.timeout,
        )
        return " ".join(data.get("response", "").split())
