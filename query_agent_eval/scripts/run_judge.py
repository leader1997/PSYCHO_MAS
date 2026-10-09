"""Score cached plans with the LLM judge; cache judgments to disk."""

import argparse
import json

import _bootstrap  # noqa: F401

from query_agent_eval.agent import plan_path
from query_agent_eval.config import cfg
from query_agent_eval.judge import run_one
from query_agent_eval.profiles import load_profiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="Subset of generator models")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    ap.add_argument("--force", action="store_true", help="Re-judge even if cached")
    args = ap.parse_args()

    models = args.models or list(cfg("generators", default=[]))
    records = load_profiles()
    if args.limit:
        records = records[: args.limit]

    total = len(models) * len(records)
    done = 0
    for model in models:
        for rec in records:
            pp = plan_path(model, rec["case_id"])
            plan_res = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else {}
            res = run_one(model, rec, plan_res, force=args.force)
            done += 1
            overall = res.get("scores", {})
            status = "ok" if res.get("parsed_ok") else f"skip({res.get('error') or 'parse'})"
            print(f"[{done}/{total}] judge {model} {rec['case_id']} -> {status}", flush=True)
    print(f"Judging complete: {done} judgments across {len(models)} models.")


if __name__ == "__main__":
    main()
