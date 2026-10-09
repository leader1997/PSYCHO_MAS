#!/usr/bin/env python3
"""Runner for 09_ragas_retriever_evaluation.ipynb."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault("XAI_LLM_BACKEND", "ollama")
os.environ.setdefault("XAI_JUDGE_BACKEND", "ollama")
os.environ.setdefault("XAI_RAGAS_BACKEND", "ollama")

import llm_client
import rag_retriever_eval as rre
import ragas_eval as reval


def main() -> int:
    if not llm_client.is_available():
        print("Ollama unavailable.")
        return 1
    if not reval.is_ragas_available():
        print("RAGAS unavailable — start Ollama (pull chat + embed models).")
        return 1

    limit = int(os.environ.get("XAI_BENCHMARK_LIMIT", "100"))
    print(f"=== Notebook 9 — RAGAS retriever ablation (limit={limit}) ===")
    print(
        f"RAGAS: {reval.ragas_backend_name()}/{reval.ragas_model_name()} "
        f"(embed: {reval.ragas_embedding_model_name()})"
    )

    comp = rre.run_all_retrievers(limit=limit, skip_existing=True, run_ragas=True)
    if not comp.empty:
        print("\n=== RAGAS comparison ===")
        print(comp.to_string(index=False))
        print(f"\nSaved -> {rre.COMPARISON_SUMMARY_CSV}")

    print("\nDONE run_nb9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
