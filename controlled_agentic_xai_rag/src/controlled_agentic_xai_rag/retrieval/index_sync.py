"""Keep FAISS vector index in sync with processed_chunks.csv after corpus rebuilds."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from controlled_agentic_xai_rag.config_loader import path


def _fingerprint_path() -> Path:
    return path("processed_chunks_csv").parent / "corpus_fingerprint.txt"


def compute_corpus_fingerprint(chunks_csv: Path | None = None) -> str:
    """Stable hash from chunk ids + corpus file size (detects new PDFs / rebuilds)."""
    csv_path = chunks_csv or path("processed_chunks_csv")
    if not csv_path.exists():
        return ""
    df = pd.read_csv(csv_path, usecols=["chunk_id"])
    ids = df["chunk_id"].astype(str).tolist()
    payload = f"{len(ids)}|{csv_path.stat().st_size}|" + "|".join(ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_stored_fingerprint() -> str:
    fp = _fingerprint_path()
    return fp.read_text(encoding="utf-8").strip() if fp.exists() else ""


def _index_fingerprint_file(index_dir: Path) -> Path:
    return index_dir / "corpus_fingerprint.txt"


def _faiss_artifacts_exist() -> bool:
    d = path("faiss_index")
    return (d / "embeddings.npy").exists() or (d / "chunk_ids.json").exists()


def invalidate_vector_indexes() -> None:
    """Remove FAISS cache so the next hybrid run rebuilds from corpus."""
    index_dir = path("faiss_index")
    if not index_dir.exists():
        return
    for name in (
        "embeddings.npy",
        "chunk_ids.json",
        "model.txt",
        "corpus_fingerprint.txt",
    ):
        p = index_dir / name
        if p.exists():
            p.unlink()


def record_corpus_built(chunks_csv: Path | None = None) -> str:
    """Write fingerprint after build_corpus; invalidate stale vector caches."""
    fp = compute_corpus_fingerprint(chunks_csv)
    if not fp:
        return ""
    prev = read_stored_fingerprint()
    _fingerprint_path().write_text(fp, encoding="utf-8")
    if prev and prev != fp:
        invalidate_vector_indexes()
    elif _faiss_artifacts_exist():
        invalidate_vector_indexes()
    return fp


def ensure_vector_indexes_current(chunks_csv: Path | None = None) -> bool:
    """
    If corpus changed since indexes were built, invalidate FAISS cache.
    Returns True when indexes were invalidated (rebuild will happen on next hybrid run).
    """
    current = compute_corpus_fingerprint(chunks_csv)
    if not current:
        return False

    stored = read_stored_fingerprint()
    if not stored:
        _fingerprint_path().write_text(current, encoding="utf-8")
        if _faiss_artifacts_exist():
            invalidate_vector_indexes()
            return True
        return False

    if stored != current:
        _fingerprint_path().write_text(current, encoding="utf-8")
        invalidate_vector_indexes()
        return True

    faiss_dir = path("faiss_index")
    idx_fp = _index_fingerprint_file(faiss_dir)
    if (faiss_dir / "embeddings.npy").exists():
        if not idx_fp.exists() or idx_fp.read_text(encoding="utf-8").strip() != current:
            invalidate_vector_indexes()
            return True

    return False


def mark_index_built(index_dir: Path, fingerprint: str | None = None) -> None:
    """Call after a successful FAISS build."""
    fp = fingerprint or read_stored_fingerprint() or compute_corpus_fingerprint()
    if not fp:
        return
    index_dir.mkdir(parents=True, exist_ok=True)
    _index_fingerprint_file(index_dir).write_text(fp, encoding="utf-8")
    if not _fingerprint_path().exists():
        _fingerprint_path().write_text(fp, encoding="utf-8")


def indexes_need_rebuild(chunks_csv: Path | None = None) -> bool:
    """Read-only check: True if hybrid dense index is missing or out of date."""
    current = compute_corpus_fingerprint(chunks_csv)
    if not current:
        return True
    stored = read_stored_fingerprint()
    if not stored or stored != current:
        return True
    faiss_dir = path("faiss_index")
    if not (faiss_dir / "embeddings.npy").exists():
        return True
    idx_fp = _index_fingerprint_file(faiss_dir)
    return not idx_fp.exists() or idx_fp.read_text(encoding="utf-8").strip() != current


def corpus_index_status() -> dict:
    """Summary for UI / CLI (no side effects)."""
    chunks_csv = path("processed_chunks_csv")
    chunk_count = 0
    if chunks_csv.exists():
        chunk_count = len(pd.read_csv(chunks_csv, usecols=["chunk_id"]))
    current = compute_corpus_fingerprint()
    stored = read_stored_fingerprint()
    faiss_dir = path("faiss_index")
    return {
        "chunk_count": chunk_count,
        "corpus_fingerprint": current[:12] if current else "",
        "corpus_recorded": bool(stored),
        "faiss_built": (faiss_dir / "embeddings.npy").exists(),
        "indexes_need_rebuild": indexes_need_rebuild(),
    }
