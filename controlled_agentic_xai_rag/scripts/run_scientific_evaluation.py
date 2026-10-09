"""Run scientific evaluation artifacts (retrieval, reports, limitations)."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from controlled_agentic_xai_rag.config_loader import ensure_dirs, path
from controlled_agentic_xai_rag.evaluation.benchmark_gold import (
    PRESET_CASE_IDS,
    ensure_claims_csv_header,
)
from controlled_agentic_xai_rag.evaluation.eval_report_batch import (
    eval_reports,
    evaluation_gaps,
    pass_rates_with_n,
)
from controlled_agentic_xai_rag.evaluation.eval_retrieval import eval_retrieval_ablation

METRIC_COLS = ["precision@5", "recall@5", "recall@10", "hit@5", "hit@10", "mrr", "ndcg@10"]


def run(limit: int = 100, skip_retrieval: bool = False) -> None:
    ensure_dirs()
    out_dir = path("paper_outputs") / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_detail = pd.DataFrame()
    if not skip_retrieval:
        retrieval_detail = eval_retrieval_ablation(limit=limit)
        retrieval_detail.to_csv(out_dir / "retrieval_per_question.csv", index=False)
        path("results_dir").mkdir(parents=True, exist_ok=True)
        retrieval_detail.to_csv(path("results_dir") / "retrieval_per_question.csv", index=False)
    else:
        for cached in (out_dir / "retrieval_per_question.csv", path("results_dir") / "retrieval_per_question.csv"):
            if cached.exists():
                retrieval_detail = pd.read_csv(cached)
                print(f"Loaded cached retrieval: {cached} ({len(retrieval_detail)} rows)")
                break

    if not retrieval_detail.empty:
        summary = retrieval_detail.groupby("retriever")[METRIC_COLS].mean().round(4).reset_index()
        summary.to_csv(out_dir / "table3_retrieval_summary.csv", index=False)
        summary.to_csv(path("results_dir") / "retrieval_eval_summary.csv", index=False)

    report_df = eval_reports(PRESET_CASE_IDS)
    report_df.to_csv(out_dir / "table5_report_quality.csv", index=False)
    pass_rates_with_n(report_df).to_csv(out_dir / "table5b_report_pass_rates.csv", index=False)

    claims_path = path("results_dir") / "claim_level_faithfulness.csv"
    ensure_claims_csv_header(claims_path)

    qp_dir = path("results_dir") / "query_plans"
    qp_n = len(list(qp_dir.glob("*.json"))) if qp_dir.exists() else 0
    gaps = evaluation_gaps(report_df, len(retrieval_detail), qp_n)
    pd.DataFrame(gaps).to_csv(out_dir / "table_evaluation_limitations.csv", index=False)

    master = []
    if not retrieval_detail.empty:
        for ret in retrieval_detail["retriever"].unique():
            sub = retrieval_detail[retrieval_detail["retriever"] == ret]
            master.append({"component": "Retrieval", "metric": f"hit@10 ({ret})", "value": round(sub["hit@10"].mean(), 4), "n": len(sub)})
    found = report_df[report_df.get("found", False) == True]  # noqa: E712
    if not found.empty:
        master.append({"component": "Report quality", "metric": "Safety pass rate", "value": round(found["safety_pass"].mean(), 4), "n": len(found)})
        master.append({"component": "Report quality", "metric": "XAI alignment pass rate", "value": round(found["xai_alignment_pass"].mean(), 4), "n": len(found)})
    pd.DataFrame(master).to_csv(out_dir / "table_master_summary.csv", index=False)

    print(json.dumps({"retrieval_n": len(retrieval_detail), "report_n": len(found), "query_plans": qp_n}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()
    run(limit=args.limit, skip_retrieval=args.skip_retrieval)


if __name__ == "__main__":
    main()
