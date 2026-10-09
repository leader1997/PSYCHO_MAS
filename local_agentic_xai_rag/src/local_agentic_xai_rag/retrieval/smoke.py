"""Retrieval smoke test."""

from __future__ import annotations

from local_agentic_xai_rag.config_loader import validate_phase2_paths
from local_agentic_xai_rag.llm.ollama_client import ensure_llama3
from local_agentic_xai_rag.retrieval.hybrid import get_retriever


def main() -> int:
    missing = validate_phase2_paths()
    if missing:
        raise FileNotFoundError("Missing Phase 2 paths:\n" + "\n".join(missing))
    ensure_llama3()
    r = get_retriever(bm25_only=True)
    hits = r.retrieve("loneliness and depression among college students", k=5)
    print(f"Retriever={r.name} hits={len(hits)}")
    for h in hits[:3]:
        print(f"  {h.chunk_id} score={h.score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
