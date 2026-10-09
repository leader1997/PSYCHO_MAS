#!/usr/bin/env python3
"""Run 100-question RAG benchmark: retrieval + generation + RAGAS (Batal et al. 2026)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import llm_client
import rag_benchmark_eval as rbe
import ragas_eval as reval

os.chdir(Path(__file__).resolve().parent)


def main() -> int:
    print("=== Build / load 100-question benchmark ===")
    benchmark = rbe.build_benchmark()
    print(f"Loaded {len(benchmark)} questions from {rbe.BENCHMARK_CSV}")

    print("\n=== Retrieval metrics (chunk gold P@k / R@k) ===")
    retrieval = rbe.compute_retrieval_metrics(benchmark)
    for k in (1, 3, 5, 10):
        print(
            f"  P@{k}={retrieval[f'precision@{k}'].mean():.3f}  "
            f"R@{k}={retrieval[f'recall@{k}'].mean():.3f}"
        )
    print(f"Saved -> {rbe.RETRIEVAL_CSV}")

    if not llm_client.is_available():
        print("\nLLM unavailable — skipping generation. Start Ollama or set OPENAI_API_KEY.")
        return 1
    if not reval.is_ragas_available():
        print("\nRAGAS unavailable — start Ollama (or set OPENAI_API_KEY for OpenAI RAGAS backend).")
        return 1

    limit = int(os.environ.get("XAI_BENCHMARK_LIMIT", str(len(benchmark))))
    print(f"\n=== Generation + RAGAS (limit={limit}) ===")
    print(
        f"Generation: {llm_client.backend_name()}/{llm_client.model_name()} | "
        f"RAGAS: {reval.ragas_backend_name()}/{reval.ragas_model_name()} "
        f"(embed: {reval.ragas_embedding_model_name()})"
    )
    results = rbe.run_full_benchmark(limit=limit)
    print(f"Saved -> {rbe.FULL_RESULTS_CSV}")
    print(f"RAGAS -> {reval.RAGAS_RESULTS_CSV}")

    summaries = reval.summarize_ragas(results)
    if summaries:
        print("\n=== RAGAS overall means (Batal et al. 2026 metrics) ===")
        print(summaries["overall"].to_string(index=False))
        print("\n=== RAGAS by category ===")
        print(summaries["by_category"].to_string())

    print("\nDONE run_rag_benchmark_full")
    return 0


if __name__ == "__main__":
    sys.exit(main())
