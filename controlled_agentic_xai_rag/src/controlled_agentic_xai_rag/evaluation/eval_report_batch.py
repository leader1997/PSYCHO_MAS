"""Offline batch evaluation of saved pipeline reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from controlled_agentic_xai_rag.config_loader import path
from controlled_agentic_xai_rag.evaluation.benchmark_gold import PRESET_CASE_IDS, latest_by_case
from controlled_agentic_xai_rag.schemas import validate
from controlled_agentic_xai_rag.validators.safety_rules import check_safety_rules
from controlled_agentic_xai_rag.validators.xai_alignment import check_xai_alignment


def load_case_card(case_id: str) -> dict:
    try:
        from controlled_agentic_xai_rag.inputs.loader import get_xai_card

        return get_xai_card(case_id)
    except Exception:
        fc_path = path("results_dir") / "formatted_cases" / f"{case_id}.json"
        if fc_path.exists():
            fc = json.loads(fc_path.read_text(encoding="utf-8"))
            return {
                "case_id": case_id,
                "prediction_probability": fc.get("prediction_probability", 0.5),
                "top_positive_factors": fc.get("top_xai_contributors", []),
                "top_protective_factors": fc.get("protective_factors", []),
            }
        return {"case_id": case_id, "prediction_probability": 0.5, "top_positive_factors": [], "top_protective_factors": []}


def question_coverage(report: dict, plan: dict) -> float:
    questions = plan.get("report_questions") or plan.get("internal_questions") or []
    if not questions:
        return float("nan")
    body = json.dumps(report).lower()

    def _qtext(q) -> str:
        if isinstance(q, dict):
            return str(q.get("question", q.get("text", "")))
        return str(q)

    answered = sum(
        1
        for q in questions
        if _qtext(q).lower()[:40] in body
        or (isinstance(q, dict) and str(q.get("slot_type", "")).lower() in body)
    )
    return answered / len(questions)


def load_final_narratives() -> dict[str, dict]:
    finals = path("results_dir") / "final_narratives.jsonl"
    if not finals.exists():
        return {}
    records = [json.loads(line) for line in finals.read_text(encoding="utf-8").splitlines() if line.strip()]
    return latest_by_case(records)


def load_query_plans() -> dict[str, dict]:
    plan_dir = path("results_dir") / "query_plans"
    plans: dict[str, dict] = {}
    if plan_dir.exists():
        for p in sorted(plan_dir.glob("*.json")):
            plan = json.loads(p.read_text(encoding="utf-8"))
            cid = plan.get("case_id", p.stem)
            plans[cid] = plan
    jsonl = path("results_dir") / "query_plans.jsonl"
    if jsonl.exists():
        for plan in [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]:
            plans[plan.get("case_id", "")] = plan
    return plans


def eval_reports(case_ids: list[str] | None = None) -> pd.DataFrame:
    case_ids = case_ids or PRESET_CASE_IDS
    narratives = load_final_narratives()
    plans = load_query_plans()
    rows: list[dict] = []

    for cid in case_ids:
        report = narratives.get(cid)
        if not report:
            rows.append({"case_id": cid, "found": False})
            continue
        card = load_case_card(cid)
        summary = report.get("screening_summary") or ""
        elevated = "elevated" in str(summary).lower() or card.get("prediction_probability", 0) >= 0.5
        safety = check_safety_rules(report, elevated=elevated)
        xai = check_xai_alignment(report, card)
        schema_ok, _ = validate(report, "screening_report.json")
        plan = plans.get(cid, {})
        rows.append({
            "case_id": cid,
            "found": True,
            "schema_valid": schema_ok,
            "safety_pass": safety["passed"],
            "xai_alignment_pass": xai["passed"],
            "question_coverage": question_coverage(report, plan) if plan else float("nan"),
            "has_query_plan": bool(plan),
            "n_citations": len(report.get("citations", [])),
            "pipeline_status": report.get("pipeline_status", report.get("validation_status", "unknown")),
        })
    return pd.DataFrame(rows)


def pass_rates_with_n(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    found = df[df.get("found", True) == True]  # noqa: E712
    n = len(found)
    rows = []
    for col, label in [
        ("safety_pass", "safety_pass"),
        ("xai_alignment_pass", "xai_alignment_pass"),
        ("schema_valid", "schema_valid"),
    ]:
        if col in found.columns:
            rate = found[col].mean() if n else float("nan")
            rows.append({"check": label, "pass_rate": round(rate, 4), "n_cases": n, "n_pass": int(found[col].sum())})
    if "question_coverage" in found.columns:
        qc = found["question_coverage"].dropna()
        rows.append({
            "check": "question_coverage_mean",
            "pass_rate": round(qc.mean(), 4) if len(qc) else float("nan"),
            "n_cases": len(qc),
            "n_pass": "",
        })
    return pd.DataFrame(rows)


def evaluation_gaps(df: pd.DataFrame, retrieval_n: int, qp_n: int) -> list[dict]:
    gaps: list[dict] = []
    if retrieval_n < 50:
        gaps.append({"gap": "retrieval_benchmark", "detail": f"Only {retrieval_n} benchmark questions resolved (target 100)."})
    found = df[df.get("found", False) == True] if not df.empty else pd.DataFrame()  # noqa: E712
    if len(found) < 3:
        gaps.append({"gap": "report_eval_n", "detail": f"Only {len(found)} preset cases evaluated (target 3)."})
    if not df.empty and df.get("has_query_plan", pd.Series(dtype=bool)).sum() < 3:
        gaps.append({"gap": "query_plans", "detail": "Fewer than 3 saved query plans; re-run pipeline with RUN_PIPELINE=True."})
    gaps.append({"gap": "expert_review", "detail": "No completed expert review scores (template only)."})
    gaps.append({"gap": "ragas", "detail": "RAGAS not run on generated reports in this evaluation pass."})
    return gaps


def main() -> None:
    df = eval_reports()
    out_dir = path("paper_outputs") / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "table5_report_quality.csv", index=False)
    pass_rates_with_n(df).to_csv(out_dir / "table5b_report_pass_rates.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
