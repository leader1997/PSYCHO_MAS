"""Compare FAISS vs ChromaDB dense retrieval on the same query."""

from __future__ import annotations

import argparse
import sys

from controlled_agentic_xai_rag.config_loader import ensure_dirs
from controlled_agentic_xai_rag.llm.ollama_client import embeddings_available, get_embed_model
from controlled_agentic_xai_rag.retrieval.base import load_chunks
from controlled_agentic_xai_rag.retrieval.dense_factory import use_vector_store


def _run_store(store: str, query: str, k: int) -> list[tuple[str, float, str]]:
    chunks = load_chunks()
    with use_vector_store(store):
        if store == "chroma":
            from controlled_agentic_xai_rag.retrieval.dense_chroma import ChromaDenseRetriever

            retriever = ChromaDenseRetriever(chunks)
        else:
            from controlled_agentic_xai_rag.retrieval.dense_llama3 import Llama3DenseRetriever

            retriever = Llama3DenseRetriever(chunks)
        hits = retriever.retrieve(query, k=k)
    return [(h.chunk_id, h.dense_score, h.title[:60]) for h in hits]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare FAISS vs Chroma dense retrieval")
    parser.add_argument(
        "--query",
        default="college student academic impairment depression anxiety screening support",
        help="Test query text",
    )
    parser.add_argument("-k", type=int, default=5, help="Top-k hits per store")
    args = parser.parse_args(argv)

    ensure_dirs()
    if not embeddings_available(force_check=True):
        print(
            f"Embeddings unavailable for {get_embed_model()!r}. "
            "Run: ollama pull nomic-embed-text",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Embed model: {get_embed_model()}")
    print(f"Query: {args.query}\n")

    for store in ("faiss", "chroma"):
        print(f"=== {store.upper()} ===")
        try:
            rows = _run_store(store, args.query, args.k)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        for rank, (cid, score, title) in enumerate(rows, start=1):
            print(f"  {rank}. {cid}  score={score:.4f}  {title}")
        print()

    print("Overlap@k (chunk_ids in both top-k lists):")
    try:
        faiss_ids = {r[0] for r in _run_store("faiss", args.query, args.k)}
        chroma_ids = {r[0] for r in _run_store("chroma", args.query, args.k)}
        overlap = faiss_ids & chroma_ids
        print(f"  {len(overlap)}/{args.k} shared: {sorted(overlap)}")
    except Exception as exc:
        print(f"  Could not compute overlap: {exc}")


if __name__ == "__main__":
    main()
