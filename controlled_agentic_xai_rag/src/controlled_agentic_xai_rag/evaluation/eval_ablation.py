"""End-to-end ablation runner."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from controlled_agentic_xai_rag.config_loader import path
from controlled_agentic_xai_rag.inputs.loader import load_xai_cards
from controlled_agentic_xai_rag.pipeline import PipelineOptions, run_case

CONDITIONS = {
    "full_system": PipelineOptions(live_prediction=False),
}


def run_ablation(case_ids: list[str] | None = None) -> pd.DataFrame:
    case_ids = case_ids or [c["case_id"] for c in load_xai_cards()]
    per_case = []
    for cond, opts in CONDITIONS.items():
        for cid in case_ids:
            try:
                result = run_case(cid, opts)
                per_case.append({
                    "condition": cond,
                    "case_id": cid,
                    "safety_status": result["final_report"].get("safety_status"),
                    "n_hits": len(result["hits"]),
                    "n_citations": len(result["final_report"].get("citations", [])),
                    "validation_passed": False,
                })
            except Exception as e:
                per_case.append({
                    "condition": cond,
                    "case_id": cid,
                    "error": str(e),
                })
    df = pd.DataFrame(per_case)
    out_dir = path("results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation_summary.csv", index=False)
    with open(out_dir / "ablation_per_case.jsonl", "w", encoding="utf-8") as f:
        for row in per_case:
            f.write(json.dumps(row) + "\n")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    run_ablation(args.case_id)


if __name__ == "__main__":
    main()
