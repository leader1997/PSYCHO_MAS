"""Retriever evaluation metrics (WP4)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import path  # noqa: E402
from generation.query_builder import build_benchmark_query  # noqa: E402
from retrieval.reranker import get_retriever  # noqa: E402


def _parse_gold(value) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {x.strip() for x in str(value).split("|") if x.strip()}


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = retrieved[:k]
    return sum(1 for c in top if c in gold) / k


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    if not gold:
        return float("nan")
    for i, c in enumerate(retrieved, start=1):
        if c in gold:
            return 1 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    dcg = sum(
        (1 if retrieved[i] in gold else 0) / np.log2(i + 2) for i in range(min(k, len(retrieved)))
    )
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_retriever(retriever_name: str, benchmark: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    retriever = get_retriever(retriever_name)
    rows = []
    for _, row in benchmark.iterrows():
        query = build_benchmark_query(
            row["question"], row["category"], row.get("gold_relevant_topics", "")
        )
        hits = retriever.retrieve(query, k=k)
        retrieved_ids = [h.chunk_id for h in hits]
        gold = _parse_gold(row.get("expected_relevant_chunk_ids", ""))
        rows.append(
            {
                "question_id": row["question_id"],
                "retriever": retriever_name,
                "precision@5": precision_at_k(retrieved_ids, gold, 5),
                "recall@5": recall_at_k(retrieved_ids, gold, 5),
                "recall@10": recall_at_k(retrieved_ids, gold, 10),
                "mrr": mrr(retrieved_ids, gold),
                "ndcg@10": ndcg_at_k(retrieved_ids, gold, 10),
                "gold_size": len(gold),
            }
        )
    return pd.DataFrame(rows)


def run_ablation(limit: int | None = None) -> pd.DataFrame:
    bench_path = path("benchmark_csv")
    if not bench_path.exists():
        from scripts.remap_gold_chunks import remap_benchmark

        remap_benchmark()
    benchmark = pd.read_csv(bench_path)
    if limit:
        benchmark = benchmark.head(limit)

    all_parts = []
    retrieval_rows = []
    for name in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        print(f"Evaluating {name}...", flush=True)
        metrics = evaluate_retriever(name, benchmark)
        all_parts.append(metrics)
        for _, row in benchmark.iterrows():
            query = build_benchmark_query(
                row["question"], row["category"], row.get("gold_relevant_topics", "")
            )
            hits = get_retriever(name).retrieve(query, k=8)
            for h in hits:
                retrieval_rows.append(
                    {
                        "question_id": row["question_id"],
                        "query": query,
                        "retriever": name,
                        **h.to_dict(),
                    }
                )

    ablation = pd.concat(all_parts, ignore_index=True)
    summary = (
        ablation.groupby("retriever")[["precision@5", "recall@5", "recall@10", "mrr", "ndcg@10"]]
        .mean()
        .round(4)
        .reset_index()
    )
    results_dir = path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(results_dir / "retriever_ablation_results.csv", index=False)
    summary.to_csv(results_dir / "retriever_ablation_summary.csv", index=False)
    pd.DataFrame(retrieval_rows).to_csv(results_dir / "retrieval_results.csv", index=False)
    print("\n=== Retriever ablation summary ===")
    print(summary.to_string(index=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_ablation(limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
