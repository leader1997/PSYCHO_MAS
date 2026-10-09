"""Gold-chunk retrieval evaluation (BM25 vs hybrid llama3)."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from local_agentic_xai_rag.config_loader import ensure_dirs, path
from local_agentic_xai_rag.llm.ollama_client import ensure_llama3
from local_agentic_xai_rag.retrieval.hybrid import get_retriever


def _parse_gold(value) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {x.strip() for x in str(value).split("|") if x.strip()}


def build_benchmark_query(question: str, category: str, topics: str = "") -> str:
    parts = [question]
    if category:
        parts.append(f"Context: {category}.")
    if topics and str(topics).strip():
        parts.append(f"Topics: {topics}")
    return " ".join(parts)


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return sum(1 for c in retrieved[:k] if c in gold) / k


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(retrieved[:k]) & gold) / len(gold)


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
    dcg = sum((1 if retrieved[i] in gold else 0) / np.log2(i + 2) for i in range(min(k, len(retrieved))))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(name: str, benchmark: pd.DataFrame, bm25_only: bool) -> pd.DataFrame:
    retriever = get_retriever(bm25_only=bm25_only)
    rows = []
    for _, row in benchmark.iterrows():
        query = build_benchmark_query(
            row["question"], row.get("category", ""), row.get("gold_relevant_topics", "")
        )
        hits = retriever.retrieve(query, k=10)
        ids = [h.chunk_id for h in hits]
        gold = _parse_gold(row.get("expected_relevant_chunk_ids", ""))
        rows.append(
            {
                "question_id": row["question_id"],
                "category": row.get("category", ""),
                "retriever": name,
                "precision@5": precision_at_k(ids, gold, 5),
                "recall@5": recall_at_k(ids, gold, 5),
                "recall@10": recall_at_k(ids, gold, 10),
                "mrr": mrr(ids, gold),
                "ndcg@10": ndcg_at_k(ids, gold, 10),
            }
        )
    return pd.DataFrame(rows)


def run_retrieval_eval(limit: int | None = None, bm25_only_hybrid: bool = False) -> pd.DataFrame:
    ensure_dirs()
    ensure_llama3()
    bench = pd.read_csv(path("benchmark_gold_csv"))
    if limit:
        bench = bench.head(limit)

    parts = [
        evaluate("bm25", bench, bm25_only=True),
        evaluate("hybrid_llama3", bench, bm25_only=bm25_only_hybrid),
    ]
    all_df = pd.concat(parts, ignore_index=True)
    summary = (
        all_df.groupby("retriever")[["precision@5", "recall@5", "recall@10", "mrr", "ndcg@10"]]
        .mean()
        .round(4)
        .reset_index()
    )
    out = path("results_dir")
    all_df.to_csv(out / "retrieval_eval_results.csv", index=False)
    summary.to_csv(out / "retrieval_eval_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bm25-only-hybrid", action="store_true", help="Use BM25 for hybrid slot (fast)")
    args = parser.parse_args()
    run_retrieval_eval(limit=args.limit, bm25_only_hybrid=args.bm25_only_hybrid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
