#!/usr/bin/env python3
"""Standalone runner for 07_ragas_benchmark_evaluation.ipynb."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import llm_client
import rag_benchmark_eval as rbe
import ragas_eval as reval

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("XAI_LLM_BACKEND", "ollama")
os.environ.setdefault("XAI_JUDGE_BACKEND", "ollama")
os.environ.setdefault("XAI_RAGAS_BACKEND", "ollama")

FIG_DIR = Path("outputs/ragas_eval/figures")


def save_ragas_figure(summaries: dict) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    overall = summaries["overall"]
    cols = [c for c in reval.RAGAS_METRICS if c in overall.columns]
    ax = overall[cols].T.plot.bar(figsize=(10, 5), legend=False, rot=45)
    ax.set_title("RAGAS metrics (Batal et al. 2026)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = FIG_DIR / "fig_ragas_overall.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path


def main() -> int:
    print("=== Notebook 7 — RAGAS benchmark evaluation ===")
    benchmark = rbe.build_benchmark()
    print(f"Benchmark: {len(benchmark)} questions")

    rbe.compute_retrieval_metrics(benchmark)
    print(f"Chunk retrieval metrics -> {rbe.RETRIEVAL_CSV}")

    if not llm_client.is_available():
        print("Generation LLM unavailable.")
        return 1
    if not reval.is_ragas_available():
        print("RAGAS unavailable — start Ollama (pull chat + embed models).")
        return 1

    limit = int(os.environ.get("XAI_BENCHMARK_LIMIT", str(len(benchmark))))
    print(f"Generation limit: {limit}")
    print(
        f"RAGAS: {reval.ragas_backend_name()}/{reval.ragas_model_name()} "
        f"(embed: {reval.ragas_embedding_model_name()})"
    )
    results = rbe.run_full_benchmark(limit=limit, run_ragas=True)

    summaries = reval.summarize_ragas(results)
    if summaries:
        print("\n=== RAGAS overall ===")
        print(summaries["overall"].to_string(index=False))
        fig = save_ragas_figure(summaries)
        print(f"Figure -> {fig}")

    print("\nDONE run_nb7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
