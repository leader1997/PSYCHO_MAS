"""Aggregate metrics + judge scores into CSVs and a markdown report."""

import argparse

import _bootstrap  # noqa: F401

from query_agent_eval.aggregate import run_aggregate
from query_agent_eval.config import cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="Subset of generator models")
    args = ap.parse_args()
    models = args.models or list(cfg("generators", default=[]))
    out = run_aggregate(models)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
