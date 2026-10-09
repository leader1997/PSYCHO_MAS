#!/usr/bin/env python3
"""CLI: Ollama-only contextual retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ollama_contextual_eval.evaluate import run_ablation  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Contextual retrieval ablation (plain / metadata / ollama_context × dense / hybrid)"
    )
    p.add_argument("--question-limit", type=int, default=None)
    p.add_argument("--chunk-limit", type=int, default=None, help="Smoke test: e.g. 500 chunks")
    p.add_argument("--skip-ollama-context", action="store_true", help="Skip LLM context generation")
    p.add_argument("--representations", default="", help="Comma-separated: plain,metadata,ollama_context")
    p.add_argument("--modes", default="", help="Comma-separated: dense,hybrid")
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument("--force-regenerate-context", action="store_true")
    p.add_argument("--embed-model", default=None)
    p.add_argument("--context-model", default=None)
    args = p.parse_args()

    reps = [x.strip() for x in args.representations.split(",") if x.strip()] or None
    modes = [x.strip() for x in args.modes.split(",") if x.strip()] or None

    from ollama_contextual_eval.ollama_client import OllamaClient

    client = OllamaClient(
        embed_model=args.embed_model or None,
        context_model=args.context_model or None,
    )

    report = run_ablation(
        representations=reps,
        retrieval_modes=modes,
        question_limit=args.question_limit,
        chunk_limit=args.chunk_limit,
        skip_ollama_context=args.skip_ollama_context,
        force_reembed=args.force_reembed,
        force_regenerate_context=args.force_regenerate_context,
        client=client,
    )
    print(json.dumps({"summary": report["summary"], "artifacts": report["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
