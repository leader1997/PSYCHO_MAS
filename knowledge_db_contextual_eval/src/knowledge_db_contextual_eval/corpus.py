"""Load PDFs from knowledge_db metadata and extract page-aware chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

from knowledge_db_contextual_eval.config import cfg, path


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_words(
    text: str,
    words_per_chunk: int | None = None,
    overlap: int | None = None,
    min_chunk_chars: int | None = None,
) -> list[str]:
    words_per_chunk = words_per_chunk or int(cfg("corpus", "words_per_chunk", default=260))
    overlap = overlap or int(cfg("corpus", "chunk_overlap", default=60))
    min_chunk_chars = min_chunk_chars or int(cfg("corpus", "min_chunk_chars", default=250))

    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + words_per_chunk])
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)
        if start + words_per_chunk >= len(words):
            break
    return chunks


def load_documents(limit_docs: int | None = None) -> list[dict]:
    kb_root = path("knowledge_db_root")
    meta_path = path("metadata_jsonl")
    docs: list[dict] = []
    with meta_path.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            if doc.get("download_status") != "downloaded":
                continue
            pdf_path = kb_root / doc.get("local_pdf_path", "")
            if not pdf_path.exists():
                continue
            doc["abs_pdf_path"] = str(pdf_path)
            docs.append(doc)
    docs = sorted(docs, key=lambda item: item["doc_id"])
    return docs[:limit_docs] if limit_docs else docs


def extract_doc_chunks(doc: dict) -> list[dict]:
    max_pages = int(cfg("corpus", "max_pages_per_doc", default=8))
    max_chunks = int(cfg("corpus", "max_chunks_per_doc", default=8))
    chunks: list[dict] = []
    try:
        reader = PdfReader(doc["abs_pdf_path"])
    except Exception as exc:
        print("PDF read failed", doc["doc_id"], exc)
        return chunks

    for page_idx, page in enumerate(reader.pages[:max_pages], start=1):
        try:
            page_text = clean_text(page.extract_text())
        except Exception:
            page_text = ""
        for chunk_idx, chunk_text in enumerate(chunk_words(page_text), start=1):
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_p{page_idx}_c{chunk_idx}",
                    "doc_id": doc["doc_id"],
                    "category_code": doc.get("category_code", ""),
                    "title": doc.get("title", ""),
                    "main_topic": doc.get("main_topic", ""),
                    "source_url": doc.get("source_url") or doc.get("pdf_url", ""),
                    "page": page_idx,
                    "text": chunk_text,
                }
            )
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def _chunk_cache_name() -> str:
    p = int(cfg("corpus", "max_pages_per_doc", default=8))
    c = int(cfg("corpus", "max_chunks_per_doc", default=8))
    w = int(cfg("corpus", "words_per_chunk", default=260))
    return f"chunks_p{p}_c{c}_w{w}.jsonl"


def _resolve_cache_dir(prefer_legacy: bool = True) -> Path:
    local = path("cache_dir")
    legacy = path("legacy_cache_dir")
    if prefer_legacy and legacy.exists():
        return legacy
    local.mkdir(parents=True, exist_ok=True)
    return local


def load_or_build_chunks(
    docs: list[dict],
    *,
    prefer_legacy_cache: bool = True,
    force_rebuild: bool = False,
) -> list[dict]:
    cache_dir = _resolve_cache_dir(prefer_legacy_cache)
    chunk_cache = cache_dir / _chunk_cache_name()

    if not force_rebuild and chunk_cache.exists():
        return [
            json.loads(line)
            for line in chunk_cache.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    chunks: list[dict] = []
    for doc in docs:
        chunks.extend(extract_doc_chunks(doc))

    local_cache = path("cache_dir")
    local_cache.mkdir(parents=True, exist_ok=True)
    out = local_cache / _chunk_cache_name()
    out.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in chunks), encoding="utf-8")
    return chunks
