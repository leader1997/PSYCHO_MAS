"""Rebuild corpus from knowledge_db PDFs and sync vector-index fingerprints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync corpus chunks with knowledge_db PDFs")
    parser.add_argument(
        "--anchor-only",
        action="store_true",
        help="Only write corpus fingerprint + invalidate stale indexes (skip PDF re-chunking)",
    )
    parser.add_argument(
        "--warm-index",
        action="store_true",
        help="After sync, build dense vector index (requires Ollama embeddings)",
    )
    args = parser.parse_args()

    from controlled_agentic_xai_rag.retrieval.index_sync import (
        corpus_index_status,
        record_corpus_built,
    )

    if not args.anchor_only:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_corpus.py")], check=True, cwd=ROOT)
    else:
        fp = record_corpus_built(ROOT / "data" / "processed_chunks.csv")
        print(f"Anchored corpus fingerprint ({fp[:12]}…) and cleared stale vector caches if needed.")

    status = corpus_index_status()
    print(
        f"Corpus: {status['chunk_count']} chunks · "
        f"FAISS built: {status['faiss_built']} · "
        f"Needs vector rebuild: {status['indexes_need_rebuild']}"
    )

    if args.warm_index:
        from controlled_agentic_xai_rag.retrieval.dense_factory import get_dense_retriever
        from controlled_agentic_xai_rag.retrieval.base import load_chunks

        print("Building dense vector index (this may take several minutes)…", flush=True)
        get_dense_retriever(load_chunks())
        print("Vector index ready.")


if __name__ == "__main__":
    main()
