"""CLI entry point."""

from __future__ import annotations

import argparse
import sys

from controlled_agentic_xai_rag.config_loader import ensure_dirs, validate_inputs
from controlled_agentic_xai_rag.inputs.loader import load_xai_cards
from controlled_agentic_xai_rag.llm.ollama_client import ensure_model, list_installed_models
from controlled_agentic_xai_rag.pipeline import PipelineOptions, run_case


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Controlled Agentic XAI-RAG — HMS prediction → query planning → hybrid RAG → report"
    )
    parser.add_argument("--case-id", type=str, help="Preset case ID or label for live HMS run")
    parser.add_argument("--all-cases", action="store_true", help="Run all seeded preset case IDs")
    parser.add_argument("--model", type=str, default=None, help="Ollama chat model")
    parser.add_argument("--embed-model", type=str, default=None, help="Ollama embedding model (hybrid retrieval)")
    parser.add_argument(
        "--no-live-prediction",
        action="store_true",
        help="Use seeded XAI cards instead of live CatBoost+SHAP (dev/tests only)",
    )
    parser.add_argument(
        "--hms-test-index",
        type=int,
        default=None,
        help="HMS held-out test row index (0..N-1); default uses preset for --case-id",
    )
    parser.add_argument("--skip-ollama-check", action="store_true", help="Skip Ollama availability check")
    args = parser.parse_args(argv)

    ensure_dirs()
    missing = validate_inputs()
    if missing:
        print("Missing prerequisites:\n" + "\n".join(missing), file=sys.stderr)
        print("Run: python scripts/seed_inputs.py && python scripts/build_corpus.py", file=sys.stderr)
        sys.exit(1)

    if not args.skip_ollama_check:
        try:
            ensure_model(args.model)
        except RuntimeError as e:
            print(f"Ollama check failed: {e}", file=sys.stderr)
            if list_installed_models():
                print(f"Available: {', '.join(list_installed_models())}", file=sys.stderr)
            sys.exit(1)

    case_ids = []
    if args.all_cases:
        case_ids = [c["case_id"] for c in load_xai_cards()]
    elif args.case_id:
        case_ids = [args.case_id]
    else:
        parser.print_help()
        sys.exit(1)

    opts = PipelineOptions(
        ollama_model=args.model,
        embed_model=args.embed_model,
        live_prediction=not args.no_live_prediction,
        hms_test_index=args.hms_test_index,
    )
    for cid in case_ids:
        print(f"Running case: {cid}", flush=True)
        result = run_case(cid, opts)
        if result.get("prediction"):
            p = result["prediction"]
            print(
                f"  live prediction: p={p.get('prediction_probability')} "
                f"test_index={p.get('test_index')}",
                flush=True,
            )
        n_q = len((result.get("query_plan") or {}).get("report_questions") or [])
        print(
            f"  model: {result.get('ollama_model')}  "
            f"report_questions: {n_q}  "
            f"status: {result['final_report'].get('safety_status')}",
            flush=True,
        )
    print("Pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
