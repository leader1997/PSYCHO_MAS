"""Run contextual retrieval ablation on the 100-question benchmark."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ollama_contextual_eval.config import ROOT, cfg, path
from ollama_contextual_eval.contextualize import build_representation_texts
from ollama_contextual_eval.ollama_client import OllamaClient
from ollama_contextual_eval.retriever import ContextualIndex

# Reuse gold resolution + metrics from the main scientific eval package
_MAS_SRC = (ROOT / ".." / "controlled_agentic_xai_rag" / "src").resolve()
if str(_MAS_SRC) not in sys.path:
    sys.path.insert(0, str(_MAS_SRC))

from controlled_agentic_xai_rag.evaluation.benchmark_gold import resolve_gold_chunks  # noqa: E402
from controlled_agentic_xai_rag.evaluation.metrics import (  # noqa: E402
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _embed_cache_file(representation: str, embed_model: str, n_chunks: int) -> Path:
    safe = embed_model.replace(":", "_")
    return path("cache_dir") / f"embeddings_{representation}_{safe}_{n_chunks}.npy"


def _load_or_build_index(
    representation: str,
    chunks: pd.DataFrame,
    client: OllamaClient,
    *,
    force_reembed: bool = False,
    force_regenerate_context: bool = False,
) -> tuple[ContextualIndex, dict]:
    indexed_texts, meta = build_representation_texts(
        representation,
        chunks,
        client=client,
        force_regenerate_context=force_regenerate_context,
    )
    cache_file = _embed_cache_file(representation, client.embed_model, len(chunks))
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not force_reembed:
        embeddings = np.load(cache_file)
        if embeddings.shape[0] != len(chunks):
            embeddings = None
    else:
        embeddings = None

    index = ContextualIndex.build(chunks, indexed_texts, client=client, embeddings=embeddings)
    if embeddings is None or not cache_file.exists() or force_reembed:
        np.save(cache_file, index.embeddings)
        meta["embeddings_cache"] = str(cache_file)

    meta["indexed_text_sample"] = indexed_texts[0][:300] if indexed_texts else ""
    return index, meta


def run_ablation(
    *,
    representations: list[str] | None = None,
    retrieval_modes: list[str] | None = None,
    question_limit: int | None = None,
    chunk_limit: int | None = None,
    skip_ollama_context: bool = False,
    force_reembed: bool = False,
    force_regenerate_context: bool = False,
    client: OllamaClient | None = None,
) -> dict:
    client = client or OllamaClient()
    k = int(cfg("eval", "k", default=10))
    representations = representations or list(cfg("eval", "representations", default=["plain", "metadata", "ollama_context"]))
    retrieval_modes = retrieval_modes or list(cfg("eval", "retrieval_modes", default=["dense", "hybrid"]))
    question_limit = question_limit if question_limit is not None else cfg("eval", "question_limit")
    chunk_limit = chunk_limit if chunk_limit is not None else cfg("eval", "chunk_limit")

    if skip_ollama_context and "ollama_context" in representations:
        representations = [r for r in representations if r != "ollama_context"]

    chunks = pd.read_csv(path("chunks_csv"))
    meta_path = path("corpus_metadata_csv")
    metadata = (
        pd.read_csv(meta_path)
        if meta_path.exists()
        else pd.DataFrame(columns=["doc_id", "title", "file_name", "expertise_area"])
    )
    if chunk_limit:
        chunks = chunks.head(int(chunk_limit)).copy()

    benchmark = pd.read_csv(path("benchmark_csv"))
    if question_limit:
        benchmark = benchmark.head(int(question_limit))

    gold_map: dict[str, set[str]] = {}
    for _, row in benchmark.iterrows():
        qid = str(row.get("question_id", row.name))
        gold_map[qid] = resolve_gold_chunks(row, chunks, metadata)

    resolved = sum(1 for g in gold_map.values() if g)
    print(f"Gold resolution: {resolved}/{len(benchmark)} questions with >=1 gold chunk")

    per_question_rows: list[dict] = []
    summary_rows: list[dict] = []
    build_meta: dict[str, dict] = {}

    for rep in representations:
        print(f"\n=== Building index: {rep} ===", flush=True)
        index, meta = _load_or_build_index(
            rep,
            chunks,
            client,
            force_reembed=force_reembed,
            force_regenerate_context=force_regenerate_context,
        )
        build_meta[rep] = meta

        for mode in retrieval_modes:
            print(f"--- Evaluating {rep} × {mode} ---", flush=True)
            hits_list: list[float] = []
            recalls: list[float] = []
            precisions: list[float] = []
            mrrs: list[float] = []
            ndcgs: list[float] = []

            for _, row in benchmark.iterrows():
                qid = str(row.get("question_id", row.name))
                question = str(row.get("question", row.get("query_text", ""))).strip()
                gold = gold_map.get(qid, set())
                if not question or not gold:
                    continue
                retrieved = index.retrieve(question, k=k, mode=mode, client=client)

                hit = int(any(c in gold for c in retrieved))
                hits_list.append(hit)
                recalls.append(recall_at_k(gold, retrieved, k))
                precisions.append(precision_at_k(gold, retrieved, k))
                mrrs.append(mrr(gold, retrieved))
                ndcgs.append(ndcg_at_k(gold, retrieved, k))

                per_question_rows.append(
                    {
                        "question_id": qid,
                        "question": question,
                        "representation": rep,
                        "retrieval_mode": mode,
                        "k": k,
                        "gold_count": len(gold),
                        "hit_at_k": hit,
                        "recall_at_k": recalls[-1],
                        "precision_at_k": precisions[-1],
                        "mrr": mrrs[-1],
                        "ndcg_at_k": ndcgs[-1],
                        "retrieved_chunk_ids": "|".join(retrieved),
                    }
                )

            n = len(hits_list)
            summary_rows.append(
                {
                    "representation": rep,
                    "retrieval_mode": mode,
                    "k": k,
                    "n_questions": n,
                    "n_benchmark_rows": len(benchmark),
                    "gold_resolved": resolved,
                    "hit_at_k": sum(hits_list) / n if n else 0.0,
                    "recall_at_k": sum(recalls) / n if n else 0.0,
                    "precision_at_k": sum(precisions) / n if n else 0.0,
                    "mrr": sum(mrrs) / n if n else 0.0,
                    "ndcg_at_k": sum(ndcgs) / n if n else 0.0,
                }
            )

    out_dir = path("outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    per_q_path = out_dir / f"per_question_{stamp}.csv"
    summary_path = out_dir / f"summary_{stamp}.csv"
    report_path = out_dir / f"report_{stamp}.json"

    pd.DataFrame(per_question_rows).to_csv(per_q_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    report = {
        "timestamp_utc": stamp,
        "methodology": "Anthropic contextual embeddings cookbook (Ollama-only)",
        "reference": "https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide",
        "corpus_chunks": len(chunks),
        "benchmark_questions": len(benchmark),
        "gold_resolved": resolved,
        "settings": {
            "embed_model": client.embed_model,
            "context_model": client.context_model,
            "representations": representations,
            "retrieval_modes": retrieval_modes,
            "k": k,
        },
        "summary": summary_rows,
        "build_meta": build_meta,
        "artifacts": {
            "per_question_csv": str(per_q_path),
            "summary_csv": str(summary_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Also write stable symlinks-like copies for convenience
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_latest.csv", index=False)
    pd.DataFrame(per_question_rows).to_csv(out_dir / "per_question_latest.csv", index=False)

    return report
