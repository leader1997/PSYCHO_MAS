"""Run full contextual retrieval ablation (knowledge_db/logs methodology)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from knowledge_db_contextual_eval.config import cfg, path
from knowledge_db_contextual_eval.contextualize import ContextBuilder, cache_stem
from knowledge_db_contextual_eval.corpus import load_documents, load_or_build_chunks
from knowledge_db_contextual_eval.ollama_client import OllamaClient
from knowledge_db_contextual_eval.queries import load_all_queries, relevant_doc_ids
from knowledge_db_contextual_eval.retriever import BM25Index, cosine_scores, hybrid_scores


def _cache_dir(prefer_legacy: bool = True) -> Path:
    legacy = path("legacy_cache_dir")
    if prefer_legacy and legacy.exists():
        return legacy
    local = path("cache_dir")
    local.mkdir(parents=True, exist_ok=True)
    return local


def build_embeddings(
    condition: str,
    chunks: list[dict],
    context_builder: ContextBuilder,
    client: OllamaClient,
    *,
    prefer_legacy_cache: bool = True,
    force_reembed: bool = False,
) -> np.ndarray:
    cache_dir = _cache_dir(prefer_legacy_cache)
    stem = cache_dir / cache_stem(condition, client.embed_model, len(chunks))
    matrix_path = stem.with_suffix(".npy")
    manifest_path = stem.with_suffix(".json")
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if not force_reembed and matrix_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("chunk_ids") == chunk_ids:
            print(f"Loading cached embeddings: {matrix_path}")
            return np.load(matrix_path)

    vectors: list[np.ndarray] = []
    start = time.time()
    for index, chunk in enumerate(chunks, start=1):
        vectors.append(client.embed_text(context_builder.text_for_condition(chunk, condition)))
        if index % 10 == 0 or index == len(chunks):
            print(f"{condition}: embedded {index}/{len(chunks)} after {time.time() - start:.1f}s", flush=True)

    matrix = np.vstack(vectors)
    out_dir = path("cache_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / cache_stem(condition, client.embed_model, len(chunks))
    np.save(out_stem.with_suffix(".npy"), matrix)
    out_stem.with_suffix(".json").write_text(json.dumps({"chunk_ids": chunk_ids}, indent=2), encoding="utf-8")
    return matrix


def dcg(relevances: list[int | float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


class EvalRunner:
    def __init__(
        self,
        docs: list[dict],
        chunks: list[dict],
        queries: list[dict],
        conditions: list[str],
        client: OllamaClient | None = None,
        context_builder: ContextBuilder | None = None,
        top_k: int | None = None,
    ):
        self.docs = docs
        self.chunks = chunks
        self.queries = queries
        self.conditions = conditions
        self.client = client or OllamaClient()
        self.context_builder = context_builder or ContextBuilder(self.client)
        self.top_k = top_k if top_k is not None else int(cfg("eval", "top_k", default=5))
        self.dense_weight = float(cfg("eval", "dense_weight", default=0.70))
        self.embedding_matrices: dict[str, np.ndarray] = {}
        self.bm25_indexes: dict[str, BM25Index] = {}
        self._query_emb_cache: dict[str, np.ndarray] = {}

    def build_indexes(self, *, prefer_legacy_cache: bool = True, force_reembed: bool = False) -> None:
        for condition in self.conditions:
            print(f"\n=== Index: {condition} ===", flush=True)
            self.embedding_matrices[condition] = build_embeddings(
                condition,
                self.chunks,
                self.context_builder,
                self.client,
                prefer_legacy_cache=prefer_legacy_cache,
                force_reembed=force_reembed,
            )
            texts = [self.context_builder.text_for_condition(c, condition) for c in self.chunks]
            self.bm25_indexes[condition] = BM25Index(texts)

    def _query_embedding(self, query: str) -> np.ndarray:
        key = hashlib.sha1((self.client.embed_model + "\n" + query).encode("utf-8")).hexdigest()
        if key not in self._query_emb_cache:
            self._query_emb_cache[key] = self.client.embed_text(query)
        return self._query_emb_cache[key]

    def retrieve(self, query: str, condition: str, mode: str) -> list[tuple[int, float, dict]]:
        dense = cosine_scores(self._query_embedding(query), self.embedding_matrices[condition])
        if mode == "dense":
            scores = dense
        elif mode == "hybrid":
            lexical = self.bm25_indexes[condition].scores(query)
            scores = hybrid_scores(dense, lexical, self.dense_weight)
        else:
            raise ValueError(mode)
        order = np.argsort(-scores)[: self.top_k]
        return [(int(idx), float(scores[idx]), self.chunks[int(idx)]) for idx in order]

    def evaluate_one(self, query_spec: dict, condition: str, mode: str) -> dict:
        expected_docs = relevant_doc_ids(query_spec, self.docs)
        results = self.retrieve(query_spec["query"], condition, mode)
        retrieved_docs = [item[2]["doc_id"] for item in results]
        retrieved_categories = [item[2]["category_code"] for item in results]
        binary_relevance = [1 if doc_id in expected_docs else 0 for doc_id in retrieved_docs]
        first_hit = next((idx + 1 for idx, rel in enumerate(binary_relevance) if rel), None)
        ideal_rels = sorted(binary_relevance, reverse=True)
        idcg = dcg(ideal_rels) or 1.0
        return {
            "query_id": query_spec["query_id"],
            "source": query_spec["source"],
            "condition": condition,
            "mode": mode,
            "expected_category": query_spec["expected_category"],
            "recall_at_k": 1.0 if first_hit else 0.0,
            "mrr_at_k": 1.0 / first_hit if first_hit else 0.0,
            "ndcg_at_k": dcg(binary_relevance) / idcg,
            "category_accuracy_at_1": (
                1.0
                if retrieved_categories and retrieved_categories[0] == query_spec["expected_category"]
                else 0.0
            ),
            "first_hit_rank": first_hit or "",
            "top_doc_ids": ",".join(retrieved_docs),
            "top_categories": ",".join(retrieved_categories),
            "query": query_spec["query"],
        }

    def run(self, modes: list[str] | None = None) -> tuple[list[dict], list[dict]]:
        modes = modes or list(cfg("eval", "modes", default=["dense", "hybrid"]))
        rows: list[dict] = []
        for condition in self.conditions:
            for mode in modes:
                print(f"--- Evaluating {condition} x {mode} ---", flush=True)
                for query_spec in self.queries:
                    rows.append(self.evaluate_one(query_spec, condition, mode))
        return rows, summarize(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["mode"], row["source"])].append(row)
    summary = []
    for (condition, mode, source), items in sorted(grouped.items()):
        summary.append(
            {
                "condition": condition,
                "mode": mode,
                "source": source,
                "n_queries": len(items),
                "recall_at_k": round(float(np.mean([item["recall_at_k"] for item in items])), 4),
                "mrr_at_k": round(float(np.mean([item["mrr_at_k"] for item in items])), 4),
                "ndcg_at_k": round(float(np.mean([item["ndcg_at_k"] for item in items])), 4),
                "category_accuracy_at_1": round(
                    float(np.mean([item["category_accuracy_at_1"] for item in items])), 4
                ),
            }
        )
    return summary


def run_evaluation(
    *,
    conditions: list[str] | None = None,
    modes: list[str] | None = None,
    limit_docs: int | None = None,
    skip_ollama_context: bool = False,
    prefer_legacy_cache: bool = True,
    force_reembed: bool = False,
    force_rebuild_chunks: bool = False,
    top_k: int | None = None,
) -> dict:
    conditions = conditions or list(cfg("eval", "conditions", default=["plain", "metadata_context", "ollama_context"]))
    if skip_ollama_context:
        conditions = [c for c in conditions if c != "ollama_context"]

    docs = load_documents(limit_docs=limit_docs)
    chunks = load_or_build_chunks(docs, prefer_legacy_cache=prefer_legacy_cache, force_rebuild=force_rebuild_chunks)
    queries = load_all_queries(chunks, docs)

    print(f"Documents: {len(docs)}, Chunks: {len(chunks)}, Queries: {len(queries)}")

    client = OllamaClient()
    context_builder = ContextBuilder(client)
    top_k = top_k if top_k is not None else int(cfg("eval", "top_k", default=5))
    runner = EvalRunner(docs, chunks, queries, conditions, client, context_builder, top_k=top_k)
    runner.build_indexes(prefer_legacy_cache=prefer_legacy_cache, force_reembed=force_reembed)
    rows, summary = runner.run(modes=modes)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = path("outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    result_json = out_dir / f"paper_contextual_retrieval_eval_{stamp}.json"
    detail_csv = out_dir / f"paper_contextual_retrieval_eval_{stamp}_details.csv"
    summary_csv = out_dir / f"paper_contextual_retrieval_eval_{stamp}_summary.csv"

    payload = {
        "settings": {
            "embed_model": client.embed_model,
            "context_model": client.context_model,
            "max_pages_per_doc": int(cfg("corpus", "max_pages_per_doc", default=8)),
            "max_chunks_per_doc": int(cfg("corpus", "max_chunks_per_doc", default=8)),
            "words_per_chunk": int(cfg("corpus", "words_per_chunk", default=260)),
            "chunk_overlap": int(cfg("corpus", "chunk_overlap", default=60)),
            "top_k": top_k,
            "dense_weight": float(cfg("eval", "dense_weight", default=0.70)),
            "conditions": conditions,
        },
        "corpus": {
            "documents": len(docs),
            "chunks": len(chunks),
            "categories": dict(Counter(chunk["category_code"] for chunk in chunks)),
        },
        "summary": summary,
        "details": rows,
        "run_id": stamp,
    }
    result_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with detail_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # Stable symlinks-like copies
    (out_dir / "summary_latest.csv").write_bytes(summary_csv.read_bytes())
    (out_dir / "details_latest.csv").write_bytes(detail_csv.read_bytes())
    (out_dir / "result_latest.json").write_bytes(result_json.read_bytes())

    payload["artifacts"] = {
        "result_json": str(result_json),
        "summary_csv": str(summary_csv),
        "details_csv": str(detail_csv),
    }
    return payload
