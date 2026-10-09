#!/usr/bin/env python3
"""
Run strengthened analysis pipeline (roadmap priorities 1–6).
Does NOT re-run full nb1–nb4 unless --full-pipeline is passed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], env: dict | None = None) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.call(cmd, cwd=ROOT, env=e)


def main() -> int:
    p = argparse.ArgumentParser(description="Strengthened pipeline runner")
    p.add_argument("--freeze-only", action="store_true", help="Only freeze v1 baseline")
    p.add_argument("--full-pipeline", action="store_true", help="Run nb1–nb4 before analyses")
    p.add_argument("--full-quality", action="store_true", help="Set XAI_FAST=0 for nb1/nb2")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    args = p.parse_args()

    py = sys.executable
    env_fast = {"XAI_FAST": "0" if args.full_quality else os.environ.get("XAI_FAST", "1")}

    # Always freeze current state first
    run([py, "run_freeze_baseline.py"])

    if args.freeze_only:
        return 0

    if args.full_pipeline:
        for script in ["run_nb1.py", "run_nb2.py"]:
            rc = run([py, script], env=env_fast)
            if rc != 0:
                return rc
        rc = run([py, "run_download_literature.py"])
        if rc != 0:
            return rc
        rc = run([py, "run_nb3.py"])
        if rc != 0:
            return rc
        rc = run([py, "run_nb4.py"])
        if rc != 0:
            return rc
        run([py, "run_freeze_baseline.py"])  # v2 snapshot after full run

    if not args.skip_ablation:
        run([py, "run_feature_ablation.py"])

    if not args.skip_validation:
        run([py, "run_validation_suite.py"])
        run([py, "run_faithfulness_controls.py"])
        run([py, "run_weighting_comparison.py"])

    print("\nDONE strengthened pipeline. See outputs/strengthened/ and outputs/versions/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
