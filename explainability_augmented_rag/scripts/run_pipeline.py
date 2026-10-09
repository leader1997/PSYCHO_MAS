#!/usr/bin/env python3
"""Run full Phase 2 pipeline (use --smoke for quick test)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Avoid OpenMP crash when cross-encoder + numpy/torch load together (Windows)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.check_call([sys.executable, *cmd], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Limit eval to 5 questions / 3 cases")
    parser.add_argument("--skip-seed", action="store_true")
    args = parser.parse_args()
    limit = "5" if args.smoke else "100"
    case_limit = "3" if args.smoke else "3"

    if not args.skip_seed:
        run(["scripts/seed_from_phase1.py"])
    run(["retrieval/build_corpus.py"])
    run(["xai/build_xai_cards.py"])
    run(["generation/query_builder.py"])
    run(["retrieval/retriever_eval.py", "--limit", limit])
    run(["generation/evidence_synthesis.py", "--limit", case_limit])
    run(["generation/narrative_generator.py", "--limit", case_limit])
    run(["safety/safety_validator.py"])
    run(["safety/faithfulness_checker.py"])
    run(["evaluation/build_benchmark.py"])
    run(["-m", "evaluation.ragas_evaluation", "--limit", limit])
    run(["-m", "evaluation.ablation_evaluation"])
    run(["evaluation/error_analysis.py"])
    run(["evaluation/generate_paper_outputs.py"])
    run(["evaluation/create_expert_review_template.py"])
    run(["evaluation/build_scientific_paper_mmd.py"])
    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
