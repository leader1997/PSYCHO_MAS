"""RAGAS benchmark with retriever ablation (dense / BM25 / hybrid)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import llm_client
import rag_benchmark_eval as rbe
import ragas_eval as reval
from retrievers import RETRIEVER_SPECS, get_retriever

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RETRIEVER_EVAL_DIR = OUTPUT_DIR / "ragas_retriever_eval"
COMPARISON_SUMMARY_CSV = RETRIEVER_EVAL_DIR / "ragas_retriever_comparison.csv"


def _paths_for_retriever(retriever_id: str) -> dict[str, Path]:
    d = RETRIEVER_EVAL_DIR / retriever_id
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "full_results": d / "rag_eval_full_results.csv",
        "ragas_results": d / "rag_eval_ragas_results.csv",
        "ragas_summary": d / "rag_eval_ragas_summary.csv",
        "ragas_by_category": d / "rag_eval_ragas_by_category.csv",
        "retrieval_metrics": d / "rag_eval_retrieval_metrics.csv",
    }


def _context_parts_for_ragas(ctx: str, hits: list[dict]) -> list[str]:
    """Split context into RAGAS context list (blocks + retrieved chunks)."""
    parts = [p.strip() for p in ctx.split("\n\n") if p.strip()]
    if not parts:
        return [rbe._doc_to_context_text(d) for d in hits]
    return parts


def compute_retrieval_metrics_for_retriever(
    retriever_id: str,
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    benchmark = benchmark if benchmark is not None else rbe.build_benchmark()
    retriever = get_retriever(retriever_id)
    paths = _paths_for_retriever(retriever_id)

    ks = [1, 3, 5, 10]
    rows = []
    for _, row in benchmark.iterrows():
        gold = set(str(row["gold_relevant_chunk_ids"]).split("|"))
        query = row["question"]
        hits = retriever.retrieve(query, k=max(ks))
        retrieved_ids = [d["id"] for d in hits]
        rec = {
            "question_id": row["question_id"],
            "category": row["category"],
            "retriever": retriever_id,
        }
        for k in ks:
            top = retrieved_ids[:k]
            hits_n = sum(1 for r in top if r in gold)
            rec[f"precision@{k}"] = round(hits_n / k if k else 0.0, 4)
            rec[f"recall@{k}"] = round(hits_n / len(gold) if gold else 0.0, 4)
        rows.append(rec)

    results = pd.DataFrame(rows)
    results.to_csv(paths["retrieval_metrics"], index=False)
    return results


def run_benchmark_for_retriever(
    retriever_id: str,
    limit: int | None = None,
    skip_existing: bool = True,
    run_ragas: bool = True,
) -> pd.DataFrame:
    """Generate answers with one retriever + full RAG context, then RAGAS score."""
    if not llm_client.is_available():
        raise RuntimeError("Ollama unavailable for generation.")

    retriever = get_retriever(retriever_id)
    paths = _paths_for_retriever(retriever_id)
    benchmark = rbe.build_benchmark()
    if limit is None:
        limit = int(os.environ.get("XAI_BENCHMARK_LIMIT", str(len(benchmark))))
    benchmark = benchmark.head(limit)

    done: set[str] = set()
    existing = pd.DataFrame()
    if paths["full_results"].exists() and skip_existing:
        existing = pd.read_csv(paths["full_results"])
        done = set(existing["question_id"].astype(str))

    retrieve_fn = retriever.retrieve
    _, _, doc_by_id, _, _ = rbe._load_kb()
    cases, agg_card = rbe._load_xai_artifacts()
    dataset_doc = doc_by_id["dataset_overview"]["text"]

    rows = []
    pending = benchmark[~benchmark["question_id"].astype(str).isin(done)]
    for i, (_, row) in enumerate(pending.iterrows(), 1):
        qid = row["question_id"]
        print(f"[{retriever_id}] [{i}/{len(pending)}] {qid}...", flush=True)
        query = f"{row['question']} {row['gold_relevant_topics']}"
        hits = retrieve_fn(query, k=8)
        ctx = rbe.build_benchmark_context(
            row,
            retrieve_fn,
            doc_by_id,
            cases,
            agg_card,
            dataset_doc,
            hits=hits,
        )
        contexts = _context_parts_for_ragas(ctx, hits)
        answer = llm_client.chat(
            rbe.BENCHMARK_XAI_SYSTEM,
            f"CONTEXT:\n{ctx}\n\nQUESTION: {row['question']}",
        )
        gold = set(str(row["gold_relevant_chunk_ids"]).split("|"))
        retrieved_ids = [d["id"] for d in retrieve_fn(row["question"], k=10)]
        p5, r5 = rbe.precision_recall_at_k(retrieved_ids, gold, 5)
        safety = rbe.evaluate_safety(row, answer)

        rows.append(
            {
                "question_id": qid,
                "category": row["category"],
                "answer_type": row["answer_type"],
                "difficulty": row["difficulty"],
                "question": row["question"],
                "answer": answer,
                "retrieved_contexts": json.dumps(contexts, ensure_ascii=False),
                "retriever": retriever_id,
                "retriever_label": retriever.label,
                "system": "full",
                "llm_backend": llm_client.backend_name(),
                "llm_model": llm_client.model_name(),
                "ragas_backend": reval.ragas_backend_name(),
                "ragas_model": reval.ragas_model_name(),
                "ragas_embedding_model": reval.ragas_embedding_model_name(),
                "precision@5": round(p5, 4),
                "recall@5": round(r5, 4),
                **safety,
            }
        )

    if rows:
        new_df = pd.DataFrame(rows)
        combined = (
            pd.concat([existing, new_df], ignore_index=True)
            if not existing.empty
            else new_df
        )
        combined.drop_duplicates(subset=["question_id"], keep="last", inplace=True)
        combined.to_csv(paths["full_results"], index=False)
    else:
        combined = pd.read_csv(paths["full_results"])

    if run_ragas and reval.is_ragas_available():
        print(f"\n=== RAGAS scoring [{retriever_id}] ===", flush=True)
        reval.run_ragas_benchmark(
            benchmark=benchmark,
            results=combined,
            skip_existing=skip_existing,
            ragas_results_csv=paths["ragas_results"],
            ragas_summary_csv=paths["ragas_summary"],
            ragas_by_category_csv=paths["ragas_by_category"],
            full_results_csv=paths["full_results"],
        )
        combined = pd.read_csv(paths["full_results"])

    return combined


def run_all_retrievers(
    limit: int | None = None,
    skip_existing: bool = True,
    run_ragas: bool = True,
    retriever_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Run benchmark + RAGAS for each retriever and write comparison table."""
    RETRIEVER_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    retriever_ids = retriever_ids or [s[0] for s in RETRIEVER_SPECS]
    benchmark = rbe.build_benchmark()
    if limit is None:
        limit = int(os.environ.get("XAI_BENCHMARK_LIMIT", str(len(benchmark))))

    comparison_rows = []
    for rid in retriever_ids:
        label = dict(RETRIEVER_SPECS).get(rid, rid)
        print(f"\n{'='*60}\nRetriever: {rid} — {label}\n{'='*60}", flush=True)
        compute_retrieval_metrics_for_retriever(rid, benchmark.head(limit))
        run_benchmark_for_retriever(
            rid, limit=limit, skip_existing=skip_existing, run_ragas=run_ragas
        )
        paths = _paths_for_retriever(rid)
        if paths["ragas_summary"].exists():
            summary = pd.read_csv(paths["ragas_summary"])
            row = {"retriever": rid, "retriever_label": label}
            for m in reval.RAGAS_METRICS:
                if m in summary.columns:
                    row[m] = summary[m].iloc[0]
            comparison_rows.append(row)

    if comparison_rows:
        comp = pd.DataFrame(comparison_rows)
        comp.to_csv(COMPARISON_SUMMARY_CSV, index=False)
        return comp
    return pd.DataFrame()


def summarize_comparison() -> pd.DataFrame:
    if COMPARISON_SUMMARY_CSV.exists():
        return pd.read_csv(COMPARISON_SUMMARY_CSV)
    rows = []
    for rid, label in RETRIEVER_SPECS:
        p = _paths_for_retriever(rid)["ragas_summary"]
        if not p.exists():
            continue
        summary = pd.read_csv(p)
        row = {"retriever": rid, "retriever_label": label}
        for m in reval.RAGAS_METRICS:
            if m in summary.columns:
                row[m] = summary[m].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)
