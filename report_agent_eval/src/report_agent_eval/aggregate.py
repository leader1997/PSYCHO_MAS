"""Aggregate RAGAS per-question scores into by-model tables, charts, and paper md."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from report_agent_eval.config import cfg, path
from report_agent_eval.ragas_score import ANSWER_METRICS

PER_QUESTION_CSV = "ragas_per_question.csv"
BY_MODEL_CSV = "ragas_by_model.csv"
PER_QUESTION_LONG_CSV = "per_question_long.csv"
BY_STRATUM_CSV = "ragas_by_model_stratum.csv"
PAPER_MD = "report_answer_quality_for_paper.md"

# gpt-oss vs itself is trivially high; only grounding metrics are meaningful.
REFERENCE_CEILING_METRICS = ["faithfulness", "answer_relevancy"]


def load_scores() -> pd.DataFrame:
    fp = path("outputs_dir") / PER_QUESTION_CSV
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}. Run run_ragas.py first.")
    return pd.read_csv(fp)


def _present_metrics(df: pd.DataFrame) -> list[str]:
    return [m for m in ANSWER_METRICS if m in df.columns]


def by_model(df: pd.DataFrame) -> pd.DataFrame:
    metrics = _present_metrics(df)
    grouped = df.groupby("model")
    agg = grouped[metrics].agg(["mean", "std"]).round(4)
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg["n_questions"] = grouped.size()
    return agg.reset_index()


def by_model_stratum(df: pd.DataFrame) -> pd.DataFrame:
    metrics = _present_metrics(df)
    if "stratum" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby(["model", "stratum"])[metrics].mean().round(4)
    return agg.reset_index()


def long_form(df: pd.DataFrame) -> pd.DataFrame:
    metrics = _present_metrics(df)
    id_cols = [c for c in ["model", "case_id", "qid", "stratum", "question_type"] if c in df.columns]
    return df.melt(
        id_vars=id_cols,
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )


def _mean_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = _present_metrics(df)
    return df.groupby("model")[metrics].mean().round(4)


def save_charts(df: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = path("charts_dir")
    charts_dir.mkdir(parents=True, exist_ok=True)
    metrics = _present_metrics(df)
    means = _mean_table(df)
    saved: list[Path] = []

    # Grouped bar chart: metrics per model.
    fig, ax = plt.subplots(figsize=(10, 5))
    means.plot.bar(ax=ax, rot=0)
    ax.set_title("RAGAS answer metrics by model (gemma4 plans, gpt-oss reference)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p1 = charts_dir / "ragas_answer_metrics_by_model.svg"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(p1)

    # Leaderboard by mean of answer metrics.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    overall = means[metrics].mean(axis=1).sort_values()
    overall.plot.barh(ax=ax)
    ax.set_title("Mean answer-metric score by model")
    ax.set_xlabel("Mean of RAGAS answer metrics")
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    p2 = charts_dir / "ragas_leaderboard.svg"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(p2)

    return saved


def _md_table(frame: pd.DataFrame, float_cols: list[str]) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            val = row[col]
            if col in float_cols and isinstance(val, (int, float)):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_paper_md(df: pd.DataFrame) -> str:
    metrics = _present_metrics(df)
    reference_model = str(cfg("reference_model", default="gpt-oss:20b"))
    plan_model = str(cfg("plan_model", default="gemma4"))
    means = _mean_table(df)

    candidates_means = means.drop(index=[reference_model], errors="ignore")
    main_tbl = candidates_means.reset_index()
    main_tbl = main_tbl.rename(columns={"model": "model"})

    # Leaderboard by mean answer metric.
    lb = candidates_means[metrics].mean(axis=1).round(4).sort_values(ascending=False)
    lb_tbl = lb.reset_index()
    lb_tbl.columns = ["model", "mean_answer_score"]

    n_by_model = df.groupby("model").size()
    total_q = int(df["qid"].nunique()) if "qid" in df.columns else len(df)

    lines: list[str] = []
    lines.append("# Report Agent Answer Quality - RAGAS (scientific results)")
    lines.append("")
    lines.append(
        "Each evaluated local LLM answered the frozen "
        f"`{plan_model}` query-plan questions from identical contextual-RAG evidence "
        f"(Ollama-context hybrid, K={cfg('retrieval', 'top_k', default=5)}). "
        f"Answers are compared to `{reference_model}` reference answers using RAGAS "
        "answer-only metrics."
    )
    lines.append("")
    lines.append(f"- Plan source (frozen): `{plan_model}`")
    lines.append(f"- Reference generator / ground truth: `{reference_model}`")
    lines.append(f"- RAGAS scorer: `{cfg('ragas', 'model', default='gpt-oss:20b')}` "
                 f"+ `{cfg('ragas', 'embedding_model', default='nomic-embed-text')}` embeddings")
    lines.append(f"- Unique questions: {total_q}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Compared against | Meaning |")
    lines.append("| --- | --- | --- |")
    lines.append("| faithfulness | answer vs contexts | Grounding in retrieved evidence |")
    lines.append("| answer_relevancy | answer vs question | Addresses the planned question |")
    lines.append("| answer_correctness | answer vs gpt-oss | Correctness vs reference answer |")
    lines.append("| semantic_similarity | answer vs gpt-oss | Meaning closeness to reference |")
    lines.append("")
    lines.append("## Main results - answer metrics by model (mean)")
    lines.append("")
    lines.append(_md_table(main_tbl, float_cols=metrics))
    lines.append("")
    lines.append("![Answer metrics by model](charts/ragas_answer_metrics_by_model.svg)")
    lines.append("")
    lines.append("## Leaderboard (mean of answer metrics)")
    lines.append("")
    lines.append(_md_table(lb_tbl, float_cols=["mean_answer_score"]))
    lines.append("")
    lines.append("![Leaderboard](charts/ragas_leaderboard.svg)")
    lines.append("")

    if reference_model in means.index:
        ref_row = means.loc[[reference_model], REFERENCE_CEILING_METRICS].reset_index()
        lines.append("## Reference ceiling (gpt-oss on its own answers)")
        lines.append("")
        lines.append(
            "Grounding metrics only; correctness/similarity vs itself are self-referential."
        )
        lines.append("")
        lines.append(_md_table(ref_row, float_cols=REFERENCE_CEILING_METRICS))
        lines.append("")

    lines.append("## Sample sizes (rows scored per model)")
    lines.append("")
    n_tbl = n_by_model.reset_index()
    n_tbl.columns = ["model", "n_rows"]
    lines.append(_md_table(n_tbl, float_cols=[]))
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Shared-model bias: `gpt-oss` is both the reference generator and the RAGAS "
        "scorer, so answer_correctness / semantic_similarity favor gpt-oss-like phrasing."
    )
    lines.append(
        "- Corpus note: contexts come from the knowledge_db_contextual_eval corpus "
        "(best-benchmarked retrieval), not the full live report-pipeline KB."
    )
    lines.append(
        "- `gemma4` authored the plans and is also a candidate answerer; planning and "
        "answering are distinct tasks but this is noted for transparency."
    )
    lines.append("- N is per-question; treat model differences as indicative. Dispersion in `ragas_by_model.csv`.")
    lines.append("")
    return "\n".join(lines)


def run_aggregate() -> dict[str, Path]:
    df = load_scores()
    out_dir = path("outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    by_model_df = by_model(df)
    p = out_dir / BY_MODEL_CSV
    by_model_df.to_csv(p, index=False)
    written["by_model"] = p

    long_df = long_form(df)
    p = out_dir / PER_QUESTION_LONG_CSV
    long_df.to_csv(p, index=False)
    written["long"] = p

    strat_df = by_model_stratum(df)
    if not strat_df.empty:
        p = out_dir / BY_STRATUM_CSV
        strat_df.to_csv(p, index=False)
        written["by_stratum"] = p

    for chart in save_charts(df):
        written[chart.stem] = chart

    md = build_paper_md(df)
    p = out_dir / PAPER_MD
    p.write_text(md, encoding="utf-8")
    written["paper_md"] = p

    return written
