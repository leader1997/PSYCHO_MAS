"""Retrieval evaluation on gold benchmark."""

from __future__ import annotations

import argparse

import pandas as pd

from controlled_agentic_xai_rag.config_loader import path
from controlled_agentic_xai_rag.evaluation.benchmark_gold import (
    load_corpus_chunks,
    load_corpus_metadata,
    resolve_gold_chunks,
)
from controlled_agentic_xai_rag.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from controlled_agentic_xai_rag.retrieval.hybrid_rrf import get_retriever


def eval_retrieval(limit: int = 100, bm25_only: bool = False, k: int = 10) -> pd.DataFrame:
    bench_path = path("benchmarks_dir") / "rag_benchmark_100_questions.csv"
    if not bench_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(bench_path).head(limit)
    chunks = load_corpus_chunks()
    meta = load_corpus_metadata()
    retriever = get_retriever(bm25_only=bm25_only)
    rows: list[dict] = []

    for _, row in df.iterrows():
        q = str(row.get("question", row.get("query_text", ""))).strip()
        relevant = resolve_gold_chunks(row, chunks, meta)
        if not q or not relevant:
            continue
        hits = retriever.retrieve(q, k=k)
        retrieved = [h.chunk_id for h in hits]
        rows.append({
            "question_id": row.get("question_id", ""),
            "category": row.get("category", ""),
            "difficulty": row.get("difficulty", ""),
            "answer_type": row.get("answer_type_master", row.get("answer_type", "")),
            "retriever": retriever.name,
            "n_gold": len(relevant),
            "precision@5": precision_at_k(relevant, retrieved, 5),
            "recall@5": recall_at_k(relevant, retrieved, 5),
            "recall@10": recall_at_k(relevant, retrieved, 10),
            "hit@5": float(any(c in relevant for c in retrieved[:5])),
            "hit@10": float(any(c in relevant for c in retrieved[:10])),
            "mrr": mrr(relevant, retrieved),
            "ndcg@10": ndcg_at_k(relevant, retrieved, 10),
        })
    return pd.DataFrame(rows)


def eval_retrieval_ablation(limit: int = 100, k: int = 10) -> pd.DataFrame:
    parts = [eval_retrieval(limit=limit, bm25_only=True, k=k)]
    parts.append(eval_retrieval(limit=limit, bm25_only=False, k=k))
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="Run BM25 vs hybrid comparison")
    args = parser.parse_args()

    if args.ablation:
        df = eval_retrieval_ablation(limit=args.limit)
        out_detail = path("results_dir") / "retrieval_per_question.csv"
        out_detail.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_detail, index=False)
    else:
        df = eval_retrieval(limit=args.limit, bm25_only=args.bm25_only)

    out = path("results_dir") / "retrieval_eval_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("No benchmark rows evaluated.")
        return

    metric_cols = ["precision@5", "recall@5", "recall@10", "hit@5", "hit@10", "mrr", "ndcg@10"]
    summary = df.groupby("retriever")[metric_cols].mean(numeric_only=True).round(4).reset_index()
    summary.to_csv(out, index=False)
    print(f"Evaluated {len(df)} questions.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
