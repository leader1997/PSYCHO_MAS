"""End-to-end: build profiles (if needed) -> generate -> judge -> aggregate."""

import argparse
import json

import _bootstrap  # noqa: F401

from query_agent_eval.agent import plan_path, run_one as generate_one
from query_agent_eval.aggregate import run_aggregate
from query_agent_eval.config import cfg, path
from query_agent_eval.judge import run_one as judge_one
from query_agent_eval.profiles import build_profiles, load_profiles, save_profiles


def _ensure_profiles():
    try:
        return load_profiles()
    except FileNotFoundError:
        records = build_profiles()
        save_profiles(records)
        return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="Subset of generator models")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    ap.add_argument("--skip-judge", action="store_true", help="Only run generation + aggregate")
    ap.add_argument("--force", action="store_true", help="Ignore caches")
    args = ap.parse_args()

    models = args.models or list(cfg("generators", default=[]))
    records = _ensure_profiles()
    if args.limit:
        records = records[: args.limit]

    total = len(models) * len(records)
    step = 0
    for model in models:
        for rec in records:
            step += 1
            gen = generate_one(model, rec, force=args.force)
            gstat = "ok" if gen.get("parsed_ok") else f"FAIL({gen.get('error') or 'parse'})"
            print(f"[gen {step}/{total}] {model} {rec['case_id']} -> {gstat}", flush=True)
            if not args.skip_judge:
                jres = judge_one(model, rec, gen, force=args.force)
                jstat = "ok" if jres.get("parsed_ok") else f"skip({jres.get('error') or 'parse'})"
                print(f"[judge {step}/{total}] {model} {rec['case_id']} -> {jstat}", flush=True)

    out = run_aggregate(models)
    print("\n=== Aggregation ===")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
