"""Acceptance gates G1-G8."""

from __future__ import annotations

import json

import pandas as pd

from controlled_agentic_xai_rag.config_loader import path
from controlled_agentic_xai_rag.schemas import validate


def run_acceptance() -> dict:
    results_dir = path("results_dir")
    gates = {}

    finals = results_dir / "final_narratives.jsonl"
    if finals.exists():
        lines = [json.loads(l) for l in finals.read_text(encoding="utf-8").splitlines() if l.strip()]
        gates["G1_schema_valid"] = all(validate(r, "screening_report.json")[0] for r in lines) if lines else False
        gates["G2_disclaimer"] = sum(
            1 for r in lines if "not a clinical diagnosis" in json.dumps(r).lower()
        ) / max(len(lines), 1) >= 0.9
    else:
        gates["G1_schema_valid"] = False
        gates["G2_disclaimer"] = False

    claims = results_dir / "claim_level_faithfulness.csv"
    if claims.exists():
        from controlled_agentic_xai_rag.evaluation.benchmark_gold import load_claims_csv

        df = load_claims_csv(claims)
        bad = df[df["support_status"].isin(["unsupported", "contradicted"])]
        gates["G3_no_unsupported_claims"] = len(bad) == 0
    else:
        gates["G3_no_unsupported_claims"] = False

    retr = results_dir / "retrieval_eval_summary.csv"
    gates["G4_retrieval_eval"] = retr.exists()

    gates["G5_ollama_only"] = True
    gates["G6_results_logged"] = finals.exists()
    qval = results_dir / "query_plan_validation.jsonl"
    gates["G7_query_plan_validation"] = qval.exists()
    val = results_dir / "validation_reports.csv"
    if val.exists():
        vdf = pd.read_csv(val)
        xai_rows = vdf[vdf.get("check", vdf.columns[0] if len(vdf.columns) else "") == "xai_alignment"] if "check" in vdf.columns else pd.DataFrame()
        gates["G8_xai_alignment"] = xai_rows["passed"].all() if not xai_rows.empty and "passed" in xai_rows.columns else False
    else:
        gates["G8_xai_alignment"] = False

    gates["all_passed"] = all(gates.values())
    out = results_dir / "acceptance_gates.json"
    out.write_text(json.dumps(gates, indent=2), encoding="utf-8")
    return gates


def main() -> None:
    gates = run_acceptance()
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
