#!/usr/bin/env python3
"""Collect Ollama answers for all 4 RAG systems on the 100-Q benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

import rag_benchmark_eval as rbe

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)


def main() -> int:
    os.environ.setdefault("XAI_LLM_BACKEND", "ollama")

    limit = int(os.environ.get("XAI_BENCHMARK_LIMIT", "100"))
    print(f"Collecting answers: {len(rbe.SYSTEMS)} systems × up to {limit} questions")
    print(f"Output: {rbe.SYSTEM_ANSWERS_CSV}")
    print(f"Wide:   {rbe.SYSTEM_ANSWERS_WIDE_CSV}")

    df = rbe.collect_all_system_answers(limit=limit, skip_existing=True)
    print(f"\nSaved {len(df)} rows (long format)")
    if rbe.SYSTEM_ANSWERS_WIDE_CSV.exists():
        wide = pd.read_csv(rbe.SYSTEM_ANSWERS_WIDE_CSV)
        print(f"Saved {len(wide)} questions (wide format)")
    print("DONE run_collect_system_answers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
