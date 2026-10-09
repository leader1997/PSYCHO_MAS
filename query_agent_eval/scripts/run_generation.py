"""Run the Query Planning Agent across models x cases; cache plans to disk."""

import argparse

import _bootstrap  # noqa: F401

from query_agent_eval.agent import run_one
from query_agent_eval.config import cfg
from query_agent_eval.profiles import load_profiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="Subset of generator models")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    ap.add_argument("--force", action="store_true", help="Regenerate even if cached")
    args = ap.parse_args()

    models = args.models or list(cfg("generators", default=[]))
    records = load_profiles()
    if args.limit:
        records = records[: args.limit]

    total = len(models) * len(records)
    done = 0
    for model in models:
        for rec in records:
            res = run_one(model, rec, force=args.force)
            done += 1
            status = "ok" if res.get("parsed_ok") else f"FAIL({res.get('error') or 'parse'})"
            print(f"[{done}/{total}] {model} {rec['case_id']} -> {status}", flush=True)
    print(f"Generation complete: {done} plans across {len(models)} models.")


if __name__ == "__main__":
    main()
