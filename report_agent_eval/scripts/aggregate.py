"""Aggregate RAGAS scores into by-model tables, charts, and the paper markdown."""

import _bootstrap  # noqa: F401

from report_agent_eval.aggregate import run_aggregate


def main() -> None:
    written = run_aggregate()
    print("Aggregation complete. Wrote:")
    for key, fp in written.items():
        print(f"  {key}: {fp}")


if __name__ == "__main__":
    main()
