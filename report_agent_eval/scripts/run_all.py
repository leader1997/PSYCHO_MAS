"""End-to-end driver: retrieval -> answers -> RAGAS -> aggregate."""

import argparse
import runpy
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

_SCRIPTS = Path(__file__).resolve().parent


def _run(script: str, argv: list[str]) -> None:
    print(f"\n===== {script} {' '.join(argv)} =====", flush=True)
    saved = sys.argv
    sys.argv = [str(_SCRIPTS / script), *argv]
    try:
        runpy.run_path(str(_SCRIPTS / script), run_name="__main__")
    finally:
        sys.argv = saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full report-agent RAGAS evaluation")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of candidate models")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    common: list[str] = []
    if args.limit is not None:
        common += ["--limit", str(args.limit)]
    if args.force:
        common += ["--force"]

    model_args: list[str] = []
    if args.models:
        model_args = ["--models", *args.models]

    _run("run_retrieval.py", common)
    _run("run_answers.py", common + model_args)
    _run("run_ragas.py", common + model_args)
    _run("aggregate.py", [])


if __name__ == "__main__":
    main()
