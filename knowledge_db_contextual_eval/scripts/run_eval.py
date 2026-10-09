#!/usr/bin/env python3
"""Run knowledge_db contextual retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_db_contextual_eval.evaluate import run_evaluation  # noqa: E402
from knowledge_db_contextual_eval.report import build_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Ollama contextual retrieval eval (knowledge_db corpus)")
    p.add_argument("--conditions", default="", help="plain,metadata_context,ollama_context")
    p.add_argument("--skip-ollama-context", action="store_true")
    p.add_argument("--limit-docs", type=int, default=None)
    p.add_argument("--no-legacy-cache", action="store_true", help="Do not reuse knowledge_db/logs cache")
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument("--force-rebuild-chunks", action="store_true")
    p.add_argument("--generate-report", action="store_true")
    p.add_argument("--top-k", type=int, default=None, help="Retrieval depth K (default from config.yaml)")
    args = p.parse_args()

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()] or None

    report = run_evaluation(
        conditions=conditions,
        limit_docs=args.limit_docs,
        skip_ollama_context=args.skip_ollama_context,
        prefer_legacy_cache=not args.no_legacy_cache,
        force_reembed=args.force_reembed,
        force_rebuild_chunks=args.force_rebuild_chunks,
        top_k=args.top_k,
    )
    print(json.dumps({"summary": report["summary"], "artifacts": report["artifacts"]}, indent=2))

    if args.generate_report:
        md = build_report(run_id=report.get("run_id"))
        print(f"Report: {md}")


if __name__ == "__main__":
    main()
