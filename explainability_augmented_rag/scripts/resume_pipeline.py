#!/usr/bin/env python3
"""Resume pipeline after partial run (OpenAI / ChatGPT backend)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.check_call([sys.executable, *cmd], cwd=ROOT, env=os.environ.copy())


def main() -> int:
    steps = [
        ["retrieval/retriever_eval.py", "--limit", "100"],
        ["generation/evidence_synthesis.py", "--limit", "3"],
        ["generation/narrative_generator.py", "--limit", "3"],
        ["safety/safety_validator.py"],
        ["safety/faithfulness_checker.py"],
        ["evaluation/build_benchmark.py"],
        ["-m", "evaluation.ragas_evaluation", "--limit", "100"],
        ["-m", "evaluation.ablation_evaluation"],
        ["evaluation/error_analysis.py"],
        ["evaluation/generate_paper_outputs.py"],
        ["evaluation/build_scientific_paper_mmd.py"],
    ]
    for cmd in steps:
        run(cmd)
    print("\nResume complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
