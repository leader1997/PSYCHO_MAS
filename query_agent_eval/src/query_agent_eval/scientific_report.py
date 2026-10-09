"""Generate paper-ready markdown report and SVG charts for query agent evaluation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from query_agent_eval.config import cfg, path

COLORS = {
    "gemma4": "#59A14F",
    "llama3": "#4E79A7",
    "qwen2.5": "#F28E2B",
    "phi4": "#E15759",
    "mistral": "#B07AA1",
}

JUDGE_DIM_LABELS = {
    "judge_relevance": "Relevance",
    "judge_xai_alignment": "XAI alignment",
    "judge_kdb_answerability": "KDB answerability",
    "judge_report_usefulness": "Report usefulness",
    "judge_probability_adaptation": "Probability adaptation",
    "judge_safety": "Safety",
    "judge_clarity": "Clarity",
    "judge_non_redundancy": "Non-redundancy",
}

L1_LABELS = {
    "schema_validity": "Schema validity",
    "pct_in_band": "In 5-9 band",
    "xai_coverage": "XAI coverage",
    "protective_coverage": "Protective coverage",
    "required_type_coverage": "Required-type coverage",
    "parse_success": "Parse success",
}


def _charts_dir() -> Path:
    d = path("outputs_dir") / "charts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def svg_leaderboard(l2: pd.DataFrame, out: Path) -> None:
    df = l2.sort_values("judge_overall", ascending=True)
    models = df["model"].tolist()
    values = df["judge_overall"].tolist()
    width, height = 720, 340
    margin_l, margin_r, margin_t, margin_b = 120, 40, 50, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_v = 100.0
    bar_h = plot_h / max(len(models), 1) * 0.65
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Judge overall score by model (%)</text>',
    ]
    for i, (model, val) in enumerate(zip(models, values)):
        y = margin_t + i * (plot_h / len(models)) + 8
        w = (val / max_v) * plot_w
        color = COLORS.get(model, "#888")
        svg.append(f'<text x="{margin_l-8}" y="{y+bar_h/2+4}" text-anchor="end" font-family="Arial" font-size="13">{model}</text>')
        svg.append(f'<rect x="{margin_l}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>')
        svg.append(f'<text x="{margin_l+w+6:.1f}" y="{y+bar_h/2+4}" font-family="Arial" font-size="12" fill="#333">{val:.1f}%</text>')
    svg.append("</svg>")
    out.write_text("\n".join(svg), encoding="utf-8")


def svg_grouped_metrics(df: pd.DataFrame, metrics: list[str], title: str, out: Path, y_max: float = 100) -> None:
    models = df["model"].tolist()
    width, height = 980, 420
    margin_l, margin_r, margin_t, margin_b = 72, 32, 64, 100
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    group_w = plot_w / len(metrics)
    bar_w = group_w / (len(models) + 1)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
    ]
    for tick in [0, 25, 50, 75, 100]:
        y = margin_t + plot_h - (tick / y_max) * plot_h
        svg.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        svg.append(f'<text x="{margin_l-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#555">{tick}</text>')
    for gi, metric in enumerate(metrics):
        x0 = margin_l + gi * group_w
        for mi, model in enumerate(models):
            val = float(df.loc[df["model"] == model, metric].iloc[0])
            bar_h = (val / y_max) * plot_h
            x = x0 + bar_w + mi * bar_w
            y = margin_t + plot_h - bar_h
            color = COLORS.get(model, "#888")
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.8:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        label = L1_LABELS.get(metric, JUDGE_DIM_LABELS.get(metric, metric))
        svg.append(f'<text x="{x0+group_w/2:.1f}" y="{height-58}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    legend_y = height - 24
    for idx, model in enumerate(models):
        x = margin_l + idx * 130
        svg.append(f'<rect x="{x}" y="{legend_y-11}" width="14" height="11" fill="{COLORS.get(model,"#888")}"/>')
        svg.append(f'<text x="{x+18}" y="{legend_y-1}" font-family="Arial" font-size="12">{model}</text>')
    svg.append("</svg>")
    out.write_text("\n".join(svg), encoding="utf-8")


def svg_stratum_heatmap(long: pd.DataFrame, out: Path) -> None:
    pivot = long.groupby(["model", "stratum"])["judge_overall"].mean().unstack()
    pivot = (pivot / 5.0 * 100.0).round(1)
    strata = ["high", "low", "borderline"]
    models = list(pivot.index)
    cell_w, cell_h = 88, 36
    width = 180 + len(strata) * cell_w
    height = 100 + len(models) * cell_h
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="180" y="24" font-family="Arial" font-size="16" font-weight="700">Judge overall (%) by risk stratum</text>',
    ]
    for j, s in enumerate(strata):
        x = 180 + j * cell_w + cell_w / 2
        svg.append(f'<text x="{x:.1f}" y="52" text-anchor="middle" font-family="Arial" font-size="12" font-weight="600">{s}</text>')
    for i, model in enumerate(models):
        y = 70 + i * cell_h
        svg.append(f'<text x="170" y="{y+cell_h/2+4}" text-anchor="end" font-family="Arial" font-size="12">{model}</text>')
        for j, s in enumerate(strata):
            val = float(pivot.loc[model, s]) if s in pivot.columns else 0.0
            x = 180 + j * cell_w
            intensity = max(0, min(1, val / 100))
            g = int(220 - 120 * intensity)
            fill = f"rgb({g},{int(80+100*intensity)},{g})"
            svg.append(f'<rect x="{x+4}" y="{y+4}" width="{cell_w-8}" height="{cell_h-8}" fill="{fill}" stroke="#ccc"/>')
            svg.append(f'<text x="{x+cell_w/2:.1f}" y="{y+cell_h/2+4}" text-anchor="middle" font-family="Arial" font-size="12">{val:.1f}</text>')
    svg.append("</svg>")
    out.write_text("\n".join(svg), encoding="utf-8")


def _fmt_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def build_scientific_report() -> Path:
    out_dir = path("outputs_dir")
    charts = _charts_dir()
    long = pd.read_csv(out_dir / "per_case_long.csv")
    l1 = pd.read_csv(out_dir / "level1_by_model.csv")
    l2 = pd.read_csv(out_dir / "level2_by_model.csv")

    svg_leaderboard(l2, charts / "judge_overall_leaderboard.svg")
    l1_metrics = ["schema_validity", "pct_in_band", "xai_coverage", "required_type_coverage", "parse_success"]
    svg_grouped_metrics(l1, l1_metrics, "Level-1 automatic metrics by model (%)", charts / "level1_metrics.svg")
    judge_dims = [c for c in l2.columns if c.startswith("judge_") and c != "judge_overall"]
    svg_grouped_metrics(l2, judge_dims, "Level-2 judge dimensions by model (%)", charts / "level2_judge_dimensions.svg")
    svg_stratum_heatmap(long, charts / "judge_by_stratum.svg")

    judge_model = str(cfg("judge", "model", default="gpt-oss:20b"))
    generators = list(cfg("generators", default=[]))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_cases = long["case_id"].nunique()
    total_questions = int(long["n_questions"].sum())

    # Stratified tables
    stratum_judge = (
        long.groupby(["model", "stratum"])["judge_overall"]
        .mean()
        .div(5.0)
        .mul(100.0)
        .round(1)
        .unstack()
        .reset_index()
    )
    stratum_judge.columns.name = None

    stratum_nq = long.groupby(["model", "stratum"])["n_questions"].mean().round(2).unstack().reset_index()

    best_row = long.loc[long["judge_overall"].idxmax()]
    worst_row = long.loc[long["judge_overall"].idxmin()]
    top = l2.sort_values("judge_overall", ascending=False).iloc[0]

    lines = []
    lines.append("# Query Planning Agent Evaluation — Scientific Results")
    lines.append("")
    lines.append(
        "Paper-ready results for the autonomous **Query Planning Agent** evaluation. "
        "Five local Ollama LLMs generate evidence-seeking question plans for "
        "50 HMS-derived student case profiles; outputs are scored with Level-1 automatic "
        f"metrics and a Level-2 LLM judge (`{judge_model}`)."
    )
    lines.append("")
    lines.append(f"- **Report generated:** `{ts}` (UTC)")
    lines.append(f"- **Cases:** {n_cases} (20 elevated / 20 low / 10 borderline)")
    lines.append(f"- **Generators:** {', '.join(f'`{m}`' for m in generators)}")
    lines.append(f"- **Judge:** `{judge_model}` (excluded from generator set)")
    lines.append(f"- **Total model-case runs:** {len(long)} ({n_cases} x {len(generators)})")
    lines.append(f"- **Total generated questions:** {total_questions} (mean {long['n_questions'].mean():.2f} per plan)")
    lines.append("")

    lines.append("## Experimental Setting")
    lines.append("")
    lines.append("- **Task:** Autonomous query planning for non-diagnostic student depression/anxiety screening.")
    lines.append("- **Input per case:** De-identified profile, ML probability, threshold, risk label, TreeSHAP XAI contributors, protective factors, calibration metadata.")
    lines.append("- **Output per case:** JSON question plan (5-9 questions preferred) with retrieval queries, expected evidence, and self-check flags.")
    lines.append("- **Execution:** Sequential generation then immediate judging; 1.0 s delay between Ollama API calls.")
    lines.append("- **Level-1 metrics:** Schema validity, question count/band, XAI coverage, protective coverage, required question-type coverage, unsafe wording rate, redundancy rate.")
    lines.append("- **Level-2 judge dimensions (1-5, reported as %):** relevance, XAI alignment, KDB answerability, report usefulness, probability adaptation, safety, clarity, non-redundancy.")
    lines.append("")

    lines.append("## Level 1 — Automatic Metrics (aggregate)")
    lines.append("")
    lines.append(_fmt_table(l1))
    lines.append("")
    lines.append("![Level-1 automatic metrics](charts/level1_metrics.svg)")
    lines.append("")

    lines.append("### Interpretation (Level 1)")
    lines.append("")
    best_xai = l1.loc[l1["xai_coverage"].idxmax()]
    best_type = l1.loc[l1["required_type_coverage"].idxmax()]
    lines.append(
        f"- **`{best_type['model']}`** achieved the highest required question-type coverage "
        f"({best_type['required_type_coverage']:.1f}%) and in-band question count "
        f"({best_type['pct_in_band']:.1f}%)."
    )
    lines.append(
        f"- **`{best_xai['model']}`** achieved the highest XAI factor coverage "
        f"({best_xai['xai_coverage']:.1f}%)."
    )
    lines.append(
        f"- **`llama3`** generated the fewest questions on average ({l1.loc[l1['model']=='llama3','mean_questions'].iloc[0]:.2f}) "
        f"and only {l1.loc[l1['model']=='llama3','pct_in_band'].iloc[0]:.1f}% within the preferred 5-9 band."
    )
    unsafe = long[long["unsafe_wording_rate"] > 0]
    lines.append(
        f"- Unsafe wording was rare overall ({len(unsafe)} model-case rows with any unsafe question); "
        f"`gemma4` had the highest unsafe-wording rate ({l1.loc[l1['model']=='gemma4','unsafe_wording_rate'].iloc[0]:.1f}%)."
    )
    lines.append("")

    lines.append("## Level 2 — LLM Judge Scores (aggregate, %)")
    lines.append("")
    lines.append(_fmt_table(l2))
    lines.append("")
    lines.append("![Judge overall leaderboard](charts/judge_overall_leaderboard.svg)")
    lines.append("")
    lines.append("![Level-2 judge dimensions](charts/level2_judge_dimensions.svg)")
    lines.append("")

    lines.append("### Leaderboard (judge overall)")
    lines.append("")
    lb = l2.sort_values("judge_overall", ascending=False)[["model", "judge_overall"]]
    lb = lb.rename(columns={"judge_overall": "overall_%"})
    lines.append(_fmt_table(lb))
    lines.append("")

    lines.append("## Results by Risk Stratum")
    lines.append("")
    lines.append("### Mean judge overall (%)")
    lines.append("")
    lines.append(_fmt_table(stratum_judge))
    lines.append("")
    lines.append("### Mean number of questions")
    lines.append("")
    lines.append(_fmt_table(stratum_nq))
    lines.append("")
    lines.append("![Judge overall by stratum](charts/judge_by_stratum.svg)")
    lines.append("")

    lines.append("## Main Finding")
    lines.append("")
    lines.append(
        f"The highest mean judge overall score was **`{top['model']}` at {top['judge_overall']:.1f}%**, "
        f"followed by **`mistral` ({l2.loc[l2['model']=='mistral','judge_overall'].iloc[0]:.1f}%)** "
        f"and **`phi4` ({l2.loc[l2['model']=='phi4','judge_overall'].iloc[0]:.1f}%)**. "
        f"**`llama3`** scored lowest ({l2.loc[l2['model']=='llama3','judge_overall'].iloc[0]:.1f}%), "
        "driven mainly by weaker XAI alignment and KDB answerability."
    )
    lines.append("")
    lines.append(
        "Automatic metrics and judge scores are not fully aligned: `mistral` leads on XAI coverage "
        "while `gemma4` leads on judge overall. This suggests that heuristic XAI string-matching "
        "and human-like judge assessment capture partially different quality aspects."
    )
    lines.append("")

    lines.append("## Example Extremes")
    lines.append("")
    lines.append(
        f"- **Best judged plan:** `{best_row['model']}` on `{best_row['case_id']}` "
        f"({best_row['judge_overall']/5*100:.1f}%, {int(best_row['n_questions'])} questions, stratum={best_row['stratum']})."
    )
    lines.append(
        f"- **Worst judged plan:** `{worst_row['model']}` on `{worst_row['case_id']}` "
        f"({worst_row['judge_overall']/5*100:.1f}%, {int(worst_row['n_questions'])} questions, stratum={worst_row['stratum']})."
    )
    lines.append("")

    lines.append("## Data Files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("| --- | --- |")
    lines.append("| `outputs/per_case_long.csv` | One row per (model, case) with all Level-1 and Level-2 fields |")
    lines.append("| `outputs/level1_by_model.csv` | Mean automatic metrics per generator |")
    lines.append("| `outputs/level2_by_model.csv` | Mean judge scores (%) per generator |")
    lines.append("| `outputs/plans/{model}/{case_id}.json` | Cached question plans |")
    lines.append("| `outputs/judge/{model}/{case_id}.json` | Cached judge scores |")
    lines.append("")

    lines.append("## Limitations and Caveats")
    lines.append("")
    lines.append(f"- **Single judge** (`{judge_model}`); judge-model bias is possible.")
    lines.append("- **Small N** (50 cases per model); treat ranking differences as indicative, not conclusive.")
    lines.append("- **Level-1 XAI coverage** uses heuristic feature-name overlap, not semantic equivalence.")
    lines.append("- **No retrieval execution** in this evaluation; KDB answerability is judged, not measured by actual retrieval success.")
    lines.append("- **Generator temperature** was 0.1; judge temperature was 0.0.")
    lines.append("")

    report_path = out_dir / "query_planning_evaluation_results_for_paper.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
