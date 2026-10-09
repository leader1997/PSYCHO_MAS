"""Build plain, metadata, and Ollama-generated contextual chunk texts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ollama_contextual_eval.config import cfg, path
from ollama_contextual_eval.ollama_client import OllamaClient


def metadata_prefix(row: pd.Series) -> str:
    parts = [
        f"Document: {row.get('title', '')}",
        f"Expertise: {row.get('expertise_area', '')}",
        f"Page: {row.get('page', '')}",
    ]
    if row.get("section_heading"):
        parts.append(f"Section: {row['section_heading']}")
    return ". ".join(str(p) for p in parts if p and not str(p).endswith(": ")) + "."


def build_document_text(chunks_df: pd.DataFrame, doc_id: str, max_chars: int) -> str:
    doc_chunks = chunks_df[chunks_df["doc_id"] == doc_id].sort_values(["page", "chunk_id"])
    parts = [str(t) for t in doc_chunks["text"].tolist() if str(t).strip()]
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def contextual_index_text(context: str, chunk_text: str) -> str:
    context = (context or "").strip()
    chunk_text = (chunk_text or "").strip()
    if not context:
        return chunk_text
    return f"{context}\n\n{chunk_text}"


def plain_texts(chunks: pd.DataFrame) -> list[str]:
    return [str(t) for t in chunks["text"].tolist()]


def metadata_texts(chunks: pd.DataFrame) -> list[str]:
    return [contextual_index_text(metadata_prefix(row), str(row["text"])) for _, row in chunks.iterrows()]


def _ollama_cache_path(model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return path("cache_dir") / f"ollama_context_{safe}.jsonl"


def load_ollama_context_cache(model: str) -> dict[str, str]:
    cache_path = _ollama_cache_path(model)
    out: dict[str, str] = {}
    if not cache_path.exists():
        return out
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["chunk_id"])] = str(rec["context"])
    return out


def save_ollama_context_cache(model: str, mapping: dict[str, str]) -> Path:
    cache_path = _ollama_cache_path(model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for chunk_id, context in mapping.items():
            f.write(json.dumps({"chunk_id": chunk_id, "context": context}, ensure_ascii=False) + "\n")
    return cache_path


def ollama_context_texts(
    chunks: pd.DataFrame,
    client: OllamaClient | None = None,
    *,
    force_regenerate: bool = False,
) -> tuple[list[str], dict[str, str]]:
    client = client or OllamaClient()
    max_doc = int(cfg("contextual", "max_document_chars", default=12000))
    cache = {} if force_regenerate else load_ollama_context_cache(client.context_model)

    doc_text_cache: dict[str, str] = {}
    texts: list[str] = []
    examples: dict[str, str] = {}

    for idx, row in chunks.iterrows():
        chunk_id = str(row["chunk_id"])
        chunk_text = str(row["text"])
        doc_id = str(row["doc_id"])

        if chunk_id not in cache:
            if doc_id not in doc_text_cache:
                doc_text_cache[doc_id] = build_document_text(chunks, doc_id, max_doc)
            cache[chunk_id] = client.generate_context(doc_text_cache[doc_id], chunk_text)

        context = cache[chunk_id]
        texts.append(contextual_index_text(context, chunk_text))
        if len(examples) < 5:
            examples[chunk_id] = context

    save_ollama_context_cache(client.context_model, cache)
    return texts, examples


def build_representation_texts(
    representation: str,
    chunks: pd.DataFrame,
    client: OllamaClient | None = None,
    *,
    force_regenerate_context: bool = False,
) -> tuple[list[str], dict]:
    meta: dict = {"representation": representation}
    if representation == "plain":
        return plain_texts(chunks), meta
    if representation == "metadata":
        return metadata_texts(chunks), meta
    if representation == "ollama_context":
        texts, examples = ollama_context_texts(
            chunks, client=client, force_regenerate=force_regenerate_context
        )
        preview_by_id = dict(zip(chunks["chunk_id"].astype(str), chunks["text"].astype(str)))
        meta["context_examples"] = [
            {"chunk_id": cid, "context": ctx, "preview": preview_by_id.get(cid, "")[:200]}
            for cid, ctx in examples.items()
        ]
        return texts, meta
    raise ValueError(f"Unknown representation: {representation}")
