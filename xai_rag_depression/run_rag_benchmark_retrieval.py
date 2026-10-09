#!/usr/bin/env python3
"""
Evaluate retrieval on rag_eval_benchmark.csv: Precision@k and Recall@k.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
KB_DIR = OUTPUT_DIR / "kb"
BENCHMARK = OUTPUT_DIR / "rag_eval_benchmark.csv"


def retrieve_top_k(query: str, docs, embeddings, embedder, k: int) -> list[str]:
    q = embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    sims = embeddings @ q
    idx = np.argsort(sims)[::-1][:k]
    return [docs[i]["id"] for i in idx]


def precision_recall_at_k(retrieved: list[str], gold: set[str], k: int) -> tuple[float, float]:
    top = retrieved[:k]
    hits = len([r for r in top if r in gold])
    precision = hits / k if k else 0.0
    recall = hits / len(gold) if gold else 0.0
    return precision, recall


def main() -> int:
    if not BENCHMARK.exists():
        print(f"Missing {BENCHMARK}. Run: python build_rag_eval_benchmark.py")
        return 1

    df = pd.read_csv(BENCHMARK)
    docs = json.loads((KB_DIR / "documents.json").read_text(encoding="utf-8"))
    embeddings = np.load(KB_DIR / "embeddings.npy")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    ks = [1, 3, 5, 10]
    rows = []
    for _, row in df.iterrows():
        gold = set(str(row["gold_relevant_chunk_ids"]).split("|"))
        retrieved = retrieve_top_k(row["question"], docs, embeddings, embedder, k=max(ks))
        rec = {"question_id": row["question_id"], "category": row["category"]}
        for k in ks:
            p, r = precision_recall_at_k(retrieved, gold, k)
            rec[f"precision@{k}"] = round(p, 4)
            rec[f"recall@{k}"] = round(r, 4)
        rows.append(rec)

    results = pd.DataFrame(rows)
    out = OUTPUT_DIR / "rag_eval_retrieval_metrics.csv"
    results.to_csv(out, index=False)

    summary = results.groupby("category")[[f"precision@{k}" for k in ks] + [f"recall@{k}" for k in ks]].mean()
    summary_path = OUTPUT_DIR / "rag_eval_retrieval_summary.csv"
    summary.round(4).to_csv(summary_path)

    print(f"Evaluated {len(results)} questions -> {out}")
    print("\nOverall mean metrics:")
    for k in ks:
        print(f"  P@{k}={results[f'precision@{k}'].mean():.3f}  R@{k}={results[f'recall@{k}'].mean():.3f}")
    print(f"\nPer-category summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
