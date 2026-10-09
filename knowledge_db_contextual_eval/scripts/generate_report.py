#!/usr/bin/env python3
"""Generate markdown report and SVG charts from latest eval run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_db_contextual_eval.report import build_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None, help="Timestamp run id; default uses result_latest.json")
    args = p.parse_args()
    out = build_report(run_id=args.run_id)
    print(out)


if __name__ == "__main__":
    main()
