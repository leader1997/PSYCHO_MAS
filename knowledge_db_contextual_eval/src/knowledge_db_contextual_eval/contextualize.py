"""Plain, metadata, and Ollama-generated contextual chunk texts."""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowledge_db_contextual_eval.config import cfg, path
from knowledge_db_contextual_eval.ollama_client import OllamaClient


def metadata_context(chunk: dict) -> str:
    return (
        f"Document title: {chunk['title']}. "
        f"Category: {chunk['category_code']}. "
        f"Document topic: {chunk['main_topic']}. "
        f"Page: {chunk['page']}."
    )


def context_prompt(chunk: dict) -> str:
    return f"""You are preparing a chunk for retrieval in a scientific RAG system.

Document title: {chunk['title']}
Document category: {chunk['category_code']}
Document topic: {chunk['main_topic']}
Page: {chunk['page']}

Chunk:
{chunk['text'][:3500]}

Write one concise sentence, maximum 45 words, that explains what this chunk is about and how it fits the document. Do not add claims not supported by the title, topic, or chunk."""


class ContextBuilder:
    def __init__(self, client: OllamaClient | None = None, cache_dir: Path | None = None):
        self.client = client or OllamaClient()
        self.cache_dir = cache_dir or self._default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe = self.client.context_model.replace(":", "_")
        self.context_cache_path = self.cache_dir / f"ollama_contexts_{safe}.json"
        self._ollama_cache = self._load_ollama_cache()

    @staticmethod
    def _default_cache_dir() -> Path:
        legacy = path("legacy_cache_dir")
        if legacy.exists():
            return legacy
        local = path("cache_dir")
        local.mkdir(parents=True, exist_ok=True)
        return local

    def _load_ollama_cache(self) -> dict[str, str]:
        if self.context_cache_path.exists():
            return json.loads(self.context_cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_ollama_cache(self) -> None:
        self.context_cache_path.write_text(
            json.dumps(self._ollama_cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def ollama_context(self, chunk: dict) -> str:
        key = chunk["chunk_id"]
        if key not in self._ollama_cache:
            text = self.client.generate_text(context_prompt(chunk), timeout=360)
            self._ollama_cache[key] = text[:700]
            self._save_ollama_cache()
        return self._ollama_cache[key]

    def text_for_condition(self, chunk: dict, condition: str) -> str:
        if condition == "plain":
            return chunk["text"]
        if condition == "metadata_context":
            return metadata_context(chunk) + "\n\n" + chunk["text"]
        if condition == "ollama_context":
            return self.ollama_context(chunk) + "\n\n" + chunk["text"]
        raise ValueError(condition)


def cache_stem(condition: str, embed_model: str, n_chunks: int) -> str:
    p = int(cfg("corpus", "max_pages_per_doc", default=8))
    c = int(cfg("corpus", "max_chunks_per_doc", default=8))
    w = int(cfg("corpus", "words_per_chunk", default=260))
    signature = f"{condition}_{embed_model}_{n_chunks}_{p}_{c}_{w}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", signature)
