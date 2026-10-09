"""Contextual RAG wrapper: Ollama-context hybrid K=5 (knowledge_db_contextual_eval).

Freezes retrieved contexts once per question and reuses them for every model,
so the answer comparison is fair (retrieval held constant).
"""

from __future__ import annotations

import json
from pathlib import Path

from report_agent_eval.config import cfg, path


def _safe(part: str) -> str:
    return part.replace(":", "_").replace("/", "_")


class ContextualRetriever:
    """Lazily builds the Ollama-context hybrid index over the knowledge_db corpus."""

    def __init__(
        self,
        top_k: int | None = None,
        condition: str | None = None,
        mode: str | None = None,
    ):
        self.top_k = int(top_k if top_k is not None else cfg("retrieval", "top_k", default=5))
        self.condition = condition or str(cfg("retrieval", "condition", default="ollama_context"))
        self.mode = mode or str(cfg("retrieval", "mode", default="hybrid"))
        self._runner = None

    def _ensure_runner(self):
        if self._runner is not None:
            return
        # Imported here so importing this module never requires the sibling package.
        from knowledge_db_contextual_eval.contextualize import ContextBuilder
        from knowledge_db_contextual_eval.corpus import load_documents, load_or_build_chunks
        from knowledge_db_contextual_eval.evaluate import EvalRunner
        from knowledge_db_contextual_eval.ollama_client import OllamaClient

        docs = load_documents()
        chunks = load_or_build_chunks(docs, prefer_legacy_cache=True)
        client = OllamaClient()
        builder = ContextBuilder(client)
        runner = EvalRunner(
            docs,
            chunks,
            queries=[],
            conditions=[self.condition],
            client=client,
            context_builder=builder,
            top_k=self.top_k,
        )
        runner.build_indexes(prefer_legacy_cache=True)
        self._runner = runner
        print(
            f"Retriever ready: condition={self.condition} mode={self.mode} "
            f"K={self.top_k} chunks={len(chunks)} docs={len(docs)}",
            flush=True,
        )

    def retrieve(self, query: str) -> list[dict]:
        self._ensure_runner()
        results = self._runner.retrieve(query, self.condition, self.mode)
        hits: list[dict] = []
        for rank, (_idx, score, chunk) in enumerate(results, start=1):
            hits.append(
                {
                    "rank": rank,
                    "score": round(float(score), 6),
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk.get("title", ""),
                    "page": chunk.get("page", ""),
                    "category_code": chunk.get("category_code", ""),
                    "text": chunk["text"],
                }
            )
        return hits


def context_path(case_id: str, qid: str) -> Path:
    return path("contexts_dir") / _safe(case_id) / f"{_safe(qid)}.json"


def load_frozen_contexts(case_id: str, qid: str) -> dict | None:
    fp = context_path(case_id, qid)
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    return None


def save_frozen_contexts(case_id: str, qid: str, payload: dict) -> Path:
    fp = context_path(case_id, qid)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def context_texts(frozen: dict) -> list[str]:
    """Plain chunk texts used as RAGAS `contexts` and shown to answering LLMs."""
    return [h["text"] for h in frozen.get("hits", []) if h.get("text")]
