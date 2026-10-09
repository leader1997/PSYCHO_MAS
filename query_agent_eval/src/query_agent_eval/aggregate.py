"""Aggregate Level-1 auto metrics + Level-2 judge scores into CSVs and a report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from query_agent_eval.agent import plan_path
from query_agent_eval.config import cfg, path
from query_agent_eval.judge import DIMENSIONS, judge_path
from query_agent_eval.metrics_auto import compute
from query_agent_eval.profiles import load_profiles

AUTO_METRICS = [
    "schema_valid",
    "n_questions",
    "in_question_band",
    "xai_factor_coverage",
    "protective_factor_coverage",
    "required_type_coverage",
    "unsafe_wording_rate",
    "redundancy_rate",
]


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_long_table(models: list[str]) -> pd.DataFrame:
    records = load_profiles()
    rows: list[dict] = []
    for rec in records:
        case_id = rec["case_id"]
        expected = rec.get("expected", {})
        for model in models:
            plan_res = _load_json(plan_path(model, case_id))
            judge_res = _load_json(judge_path(model, case_id))

            row: dict = {
                "model": model,
                "case_id": case_id,
                "stratum": rec.get("stratum"),
                "plan_present": plan_res is not None,
                "parsed_ok": bool(plan_res.get("parsed_ok")) if plan_res else False,
            }
            plan = plan_res.get("plan", {}) if plan_res else {}
            auto = compute(plan if isinstance(plan, dict) else {}, expected)
            for m in AUTO_METRICS:
                row[m] = auto.get(m)

            scores = judge_res.get("scores", {}) if judge_res else {}
            for d in DIMENSIONS:
                row[f"judge_{d}"] = scores.get(d)
            valid_scores = [scores.get(d) for d in DIMENSIONS if isinstance(scores.get(d), (int, float))]
            row["judge_overall"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None
            rows.append(row)
    return pd.DataFrame(rows)


# Level-1 rate/fraction metrics reported as percentages (0-100).
_L1_PCT_COLS = [
    "schema_validity",
    "pct_in_band",
    "xai_coverage",
    "protective_coverage",
    "required_type_coverage",
    "unsafe_wording_rate",
    "redundancy_rate",
    "parse_success",
]


def level1_by_model(long: pd.DataFrame) -> pd.DataFrame:
    agg = long.groupby("model").agg(
        n_cases=("case_id", "count"),
        schema_validity=("schema_valid", "mean"),
        mean_questions=("n_questions", "mean"),
        pct_in_band=("in_question_band", "mean"),
        xai_coverage=("xai_factor_coverage", "mean"),
        protective_coverage=("protective_factor_coverage", "mean"),
        required_type_coverage=("required_type_coverage", "mean"),
        unsafe_wording_rate=("unsafe_wording_rate", "mean"),
        redundancy_rate=("redundancy_rate", "mean"),
        parse_success=("parsed_ok", "mean"),
    )
    for c in _L1_PCT_COLS:
        agg[c] = (agg[c] * 100.0).round(1)
    agg["mean_questions"] = agg["mean_questions"].round(2)
    return agg.reset_index()


def level2_by_model(long: pd.DataFrame) -> pd.DataFrame:
    cols = {f"judge_{d}": (f"judge_{d}", "mean") for d in DIMENSIONS}
    cols["judge_overall"] = ("judge_overall", "mean")
    agg = long.groupby("model").agg(**cols)
    # Convert 1-5 judge scores to percentage: score / 5 * 100.
    agg = (agg / 5.0 * 100.0).round(1)
    return agg.reset_index()


def _fmt_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def build_report(long: pd.DataFrame, l1: pd.DataFrame, l2: pd.DataFrame) -> str:
    judge_model = str(cfg("judge", "model", default="gpt-oss:20b"))
    n_cases = long["case_id"].nunique()
    models = list(long["model"].unique())

    lines: list[str] = []
    lines.append("# Query Planning Agent Evaluation Report")
    lines.append("")
    lines.append(
        f"Evaluated **{len(models)} local generators** on **{n_cases} HMS-derived case profiles** "
        f"(stratified high/low/borderline), scored with Level-1 automatic metrics and a "
        f"Level-2 LLM judge (`{judge_model}`)."
    )
    lines.append("")
    lines.append(f"Generators: {', '.join(f'`{m}`' for m in models)}")
    lines.append("")

    lines.append("## Level 1 — Automatic metrics (per model)")
    lines.append("")
    lines.append(_fmt_table(l1))
    lines.append("")
    lines.append(
        "- All rate/coverage columns are percentages (0-100%); `mean_questions` is a count. "
        "`unsafe_wording_rate` and `redundancy_rate` are lower-is-better."
    )
    lines.append("")

    lines.append(f"## Level 2 — LLM judge ({judge_model}) — ratings as percentages (per model)")
    lines.append("")
    lines.append(_fmt_table(l2))
    lines.append("")
    lines.append("- Each judge dimension is scored 1-5 then rescaled to a percentage (score / 5 x 100).")
    lines.append("")

    # Leaderboard by judge_overall
    if "judge_overall" in l2.columns and l2["judge_overall"].notna().any():
        lb = l2.sort_values("judge_overall", ascending=False)[["model", "judge_overall"]]
        lb = lb.rename(columns={"judge_overall": "judge_overall_pct"})
        lines.append("### Leaderboard (mean judge overall, %)")
        lines.append("")
        lines.append(_fmt_table(lb))
        lines.append("")

    # Best / worst examples by judge_overall (long table still holds 1-5 scale)
    scored = long[long["judge_overall"].notna()]
    if not scored.empty:
        best = scored.loc[scored["judge_overall"].idxmax()]
        worst = scored.loc[scored["judge_overall"].idxmin()]
        best_pct = round(best["judge_overall"] / 5.0 * 100.0, 1)
        worst_pct = round(worst["judge_overall"] / 5.0 * 100.0, 1)
        lines.append("### Example extremes (by judge overall)")
        lines.append("")
        lines.append(
            f"- Best: `{best['model']}` on `{best['case_id']}` "
            f"(overall {best_pct}%, {int(best['n_questions'])} questions)."
        )
        lines.append(
            f"- Worst: `{worst['model']}` on `{worst['case_id']}` "
            f"(overall {worst_pct}%, {int(worst['n_questions'])} questions)."
        )
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        f"- Single-judge design (`{judge_model}`); judge-model bias is possible. The judge is "
        "excluded from the generator set to avoid self-judging."
    )
    lines.append(f"- Small sample (N={n_cases} cases per model); treat differences as indicative, not conclusive.")
    lines.append(
        "- Auto metrics use heuristic matching (feature-name overlap, token Jaccard) and are "
        "proxies, not ground truth."
    )
    lines.append("")
    return "\n".join(lines)


def run_aggregate(models: list[str] | None = None) -> dict:
    models = models or list(cfg("generators", default=[]))
    out_dir = path("outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    long = build_long_table(models)
    l1 = level1_by_model(long)
    l2 = level2_by_model(long)

    long_path = out_dir / "per_case_long.csv"
    l1_path = out_dir / "level1_by_model.csv"
    l2_path = out_dir / "level2_by_model.csv"
    report_path = out_dir / "report.md"

    long.to_csv(long_path, index=False)
    l1.to_csv(l1_path, index=False)
    l2.to_csv(l2_path, index=False)
    report_path.write_text(build_report(long, l1, l2), encoding="utf-8")

    return {
        "per_case_long": str(long_path),
        "level1_by_model": str(l1_path),
        "level2_by_model": str(l2_path),
        "report": str(report_path),
        "n_rows": len(long),
    }
