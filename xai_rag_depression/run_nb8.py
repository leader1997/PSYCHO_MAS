#!/usr/bin/env python3
"""Runner for notebook 08_three_system_evaluation.ipynb."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
os.environ.setdefault("XAI_LLM_BACKEND", "ollama")

import pandas as pd

import three_system_eval as tse
from case_profiles import PROFILES_JSON, build_case_profiles


def main() -> int:
    print("Building case profiles (25 features × 3 cases)...")
    build_case_profiles()
    print(f"Profiles -> {PROFILES_JSON}")

    print("\nRunning three-system evaluation (Ollama + optional GPT judge)...")
    results = tse.run_three_system_eval(skip_existing=True)

    summaries = tse.summarize(results)
    if summaries:
        print("\n=== Overall means ===")
        print(summaries["overall"].to_string())
        print("\n=== Per case ===")
        print(summaries["by_case"].to_string())

    if tse.ANSWERS_CSV.exists():
        n = len(pd.read_csv(tse.ANSWERS_CSV))
        print(f"\nAnswers: {n} rows -> {tse.ANSWERS_CSV}")

    print("\nDONE run_nb8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
