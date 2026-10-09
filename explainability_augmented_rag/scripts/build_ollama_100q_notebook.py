#!/usr/bin/env python3
"""Generate notebooks/13_ollama_100q_benchmark_report.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "13_ollama_100q_benchmark_report.ipynb"

nb: dict = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "cells": []}


def md(s: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})


def code(s: str) -> None:
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})


md(
    """# Ollama-Only RAG Evaluation Report — 100-Question Benchmark

**System:** Controlled Agentic Explainability-Augmented RAG (local)  
**Benchmark:** `data/agentic_xai_rag_100q_benchmark.xlsx` (sheet `Benchmark_100`)  
**Stack:** Ollama `llama3` (generation + judge) + `nomic-embed-text` (embeddings) + BM25/hybrid retrieval

This notebook produces all **tables and visualizations** for the scientific paper:
1. Benchmark composition tables  
2. Full Ollama RAG evaluation (retrieve → generate → judge)  
3. Retrieval keyword metrics  
4. Generation / faithfulness / safety scores  
5. Breakdowns by category, difficulty, answer type  
6. Failure-case tables  
7. Export to Excel

> **Runtime:** Full 100 questions ≈ several hours (Ollama calls per question).  
> Set `EVAL_LIMIT = 10` for a quick test, then `EVAL_LIMIT = None` for the paper.
"""
)

code(
    """import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
%matplotlib inline

ROOT = Path.cwd()
if not (ROOT / "config.yaml").exists() and (ROOT.parent / "config.yaml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.ollama_client_local import ping, list_models, GEN_MODEL, JUDGE_MODEL, EMBED_MODEL
from evaluation.ollama_100q_benchmark_eval import load_benchmark, run_evaluation, RESULTS_CSV

BENCHMARK_XLSX = ROOT / "data" / "agentic_xai_rag_100q_benchmark.xlsx"
FIG_DIR = ROOT / "results" / "figures_100q"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
PALETTE = sns.color_palette("colorblind")

print("Project:", ROOT)
print("Ollama reachable:", ping())
if ping():
    print("Models:", list_models()[:8])
print(f"Gen={GEN_MODEL} Judge={JUDGE_MODEL} Embed={EMBED_MODEL}")
"""
)

md("## 1. Load benchmark & composition tables")
code(
    """bench = load_benchmark(BENCHMARK_XLSX)
rubric = pd.read_excel(BENCHMARK_XLSX, sheet_name="Evaluation_Rubric")

display(bench.head(3))
print(f"Total questions: {len(bench)}")

# Table 1: counts by category
tbl_category = bench.groupby("category").size().reset_index(name="n_questions")
tbl_category["pct"] = (tbl_category["n_questions"] / len(bench) * 100).round(1)
display(tbl_category.sort_values("n_questions", ascending=False))

# Table 2: difficulty x answer_type
tbl_cross = pd.crosstab(bench["difficulty"], bench["answer_type"])
display(tbl_cross)

# Table 3: evaluation rubric
display(rubric)
"""
)

md("## 2. Run Ollama evaluation (or load cached results)")
code(
    """# Set EVAL_LIMIT = 10 for smoke test; None for all 100 questions
EVAL_LIMIT = None  # change to 10 for quick test
# BM25_ONLY = True skips Ollama embedding index build (fast smoke test)
BM25_ONLY = False
RUN_FRESH = False

if RUN_FRESH or not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size < 50:
    results = run_evaluation(limit=EVAL_LIMIT, xlsx=BENCHMARK_XLSX, resume=True, bm25_only=BM25_ONLY)
else:
    results = pd.read_csv(RESULTS_CSV)
    if EVAL_LIMIT:
        results = results.head(EVAL_LIMIT)

print(f"Evaluated rows: {len(results)}")
results.head(2)
"""
)

md("## 3. Overall metrics table")
code(
    """metric_cols = [c for c in results.columns if c in (
    "keyword_recall@5", "keyword_precision@5",
    "answer_correctness", "answer_relevancy", "faithfulness",
    "context_precision", "safety_judge", "behavior_alignment"
)]
overall = results[metric_cols].mean(numeric_only=True).round(4).to_frame("mean")
if "safety_regex_pass" in results.columns:
    overall.loc["safety_regex_pass_rate"] = results["safety_regex_pass"].mean()
display(overall.T)
"""
)

md("## 4. Metrics by category")
code(
    """by_cat = results.groupby("category")[metric_cols].mean(numeric_only=True).round(4)
if "safety_regex_pass" in results.columns:
    by_cat["safety_regex_pass"] = results.groupby("category")["safety_regex_pass"].mean().round(4)
display(by_cat.sort_values("answer_correctness", ascending=False))
by_cat.to_csv(ROOT / "results" / "ollama_100q_table_by_category.csv")
"""
)

md("## 5. Metrics by difficulty and answer type")
code(
    """by_diff = results.groupby("difficulty")[metric_cols].mean(numeric_only=True).round(4)
by_type = results.groupby("answer_type")[metric_cols].mean(numeric_only=True).round(4)
display(by_diff)
display(by_type)
"""
)

md("## 6. Visualizations")

code(
    """# Table 4: summary statistics (mean, std, median, min, max)
stats_rows = []
for c in metric_cols:
    if c in results.columns:
        s = pd.to_numeric(results[c], errors="coerce")
        stats_rows.append({
            "metric": c,
            "mean": s.mean(),
            "std": s.std(),
            "median": s.median(),
            "min": s.min(),
            "max": s.max(),
            "pct_ge_0.7": (s >= 0.7).mean() * 100,
        })
tbl_stats = pd.DataFrame(stats_rows).round(4)
display(tbl_stats)
"""
)

code(
    """# Table 5: category x difficulty (answer_correctness mean)
if "answer_correctness" in results.columns:
    tbl_cat_diff = pd.pivot_table(
        results, values="answer_correctness", index="category", columns="difficulty", aggfunc="mean"
    ).round(4)
    display(tbl_cat_diff)
"""
)

code(
    """# Table 6: full results preview (truncated answers)
preview_cols = [
    "question_id", "category", "difficulty", "keyword_recall@5", "answer_correctness",
    "faithfulness", "safety_judge", "safety_regex_pass", "question", "generated_answer"
]
display(results[[c for c in preview_cols if c in results.columns]].head(20))
"""
)

code(
    """# Fig A: benchmark composition
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bench["category"].value_counts().plot(kind="barh", ax=axes[0], color=PALETTE[0])
axes[0].set_title("Questions by category")
bench["difficulty"].value_counts().reindex(["easy","medium","hard"]).plot(kind="bar", ax=axes[1], color=PALETTE[1])
axes[1].set_title("Questions by difficulty")
fig.tight_layout()
fig.savefig(FIG_DIR / "figA_benchmark_composition.png", dpi=300, bbox_inches="tight")
plt.show()
"""
)

code(
    """# Fig B: overall metric radar / bar
if metric_cols:
    fig, ax = plt.subplots(figsize=(10, 5))
    overall[overall.index.isin(metric_cols)].T.plot(kind="bar", ax=ax, legend=False, color=PALETTE[2])
    ax.set_title("Overall Ollama RAG metrics (mean)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figB_overall_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()
"""
)

code(
    """# Fig C: heatmap by category
if len(by_cat) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(by_cat[metric_cols], annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Mean scores by benchmark category")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figC_category_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
"""
)

code(
    """# Fig D: retrieval keyword metrics by category
fig, ax = plt.subplots(figsize=(10, 5))
if "keyword_recall@5" in results.columns:
    g = results.groupby("category")[["keyword_recall@5", "keyword_precision@5"]].mean()
    g.plot(kind="bar", ax=ax)
    ax.set_title("Retrieval keyword coverage by category")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figD_retrieval_by_category.png", dpi=300, bbox_inches="tight")
    plt.show()
"""
)

code(
    """# Fig E: judge metrics grouped bar
judge_cols = [c for c in ["answer_correctness","answer_relevancy","faithfulness","context_precision","safety_judge","behavior_alignment"] if c in results.columns]
if judge_cols:
    fig, ax = plt.subplots(figsize=(10, 5))
    results[judge_cols].mean().plot(kind="bar", ax=ax, color=PALETTE[:len(judge_cols)])
    ax.set_ylim(0, 1)
    ax.set_title("Ollama judge metrics (overall mean)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figE_judge_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()
"""
)

code(
    """# Fig F: safety pass rate
if "safety_regex_pass" in results.columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    results["safety_regex_pass"].value_counts(normalize=True).plot(kind="bar", ax=axes[0])
    axes[0].set_title("Regex safety pass rate")
    if "safety_judge" in results.columns:
        results["safety_judge"].hist(bins=15, ax=axes[1], color=PALETTE[4])
        axes[1].set_title("Ollama safety judge score distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figF_safety.png", dpi=300, bbox_inches="tight")
    plt.show()
"""
)

code(
    """# Fig G: difficulty vs correctness
if "answer_correctness" in results.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=results, x="difficulty", y="answer_correctness", order=["easy","medium","hard"], ax=ax)
    ax.set_title("Answer correctness by difficulty")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figG_correctness_by_difficulty.png", dpi=300, bbox_inches="tight")
    plt.show()
"""
)

code(
    """# Fig H: metric score distributions
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()
for i, c in enumerate(metric_cols[:8]):
    if c in results.columns:
        pd.to_numeric(results[c], errors="coerce").hist(bins=12, ax=axes[i], color=PALETTE[i % len(PALETTE)])
        axes[i].set_title(c)
for j in range(len(metric_cols[:8]), len(axes)):
    axes[j].axis("off")
fig.suptitle("Score distributions across evaluated questions")
fig.tight_layout()
fig.savefig(FIG_DIR / "figH_metric_distributions.png", dpi=300, bbox_inches="tight")
plt.show()
"""
)

code(
    """# Fig I: metric correlation heatmap
num = results[metric_cols].apply(pd.to_numeric, errors="coerce")
if len(num.columns) > 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between evaluation metrics")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figI_metric_correlation.png", dpi=300, bbox_inches="tight")
    plt.show()
"""
)

code(
    """# Fig J: retrieval vs faithfulness scatter
if {"keyword_recall@5", "faithfulness"}.issubset(results.columns):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=results, x="keyword_recall@5", y="faithfulness",
        hue="category", alpha=0.7, ax=ax
    )
    ax.set_title("Retrieval keyword recall vs answer faithfulness")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figJ_retrieval_vs_faithfulness.png", dpi=300, bbox_inches="tight")
    plt.show()
"""
)

md("## 7. Failure cases (lowest scores)")
code(
    """fail = results.copy()
if "answer_correctness" in fail.columns:
    fail = fail.sort_values("answer_correctness").head(15)
    cols = ["question_id", "category", "answer_correctness", "faithfulness", "safety_judge", "safety_regex_pass", "question"]
    display(fail[[c for c in cols if c in fail.columns]])
    fail.to_csv(ROOT / "results" / "ollama_100q_failure_cases.csv", index=False)
"""
)

md("## 8. Example Q/A pairs")
code(
    """sample = results[["question_id", "category", "question", "expected_answer", "generated_answer"]].head(5)
for _, r in sample.iterrows():
    print("=" * 80)
    print(r["question_id"], "|", r["category"])
    print("Q:", r["question"][:200])
    print("Expected:", str(r["expected_answer"])[:300])
    print("Generated:", str(r["generated_answer"])[:300])
"""
)

md("## 9. Export report Excel")
code(
    """report_xlsx = ROOT / "results" / "ollama_100q_evaluation_report.xlsx"
with pd.ExcelWriter(report_xlsx, engine="openpyxl") as w:
    bench.to_excel(w, sheet_name="Benchmark_100", index=False)
    results.to_excel(w, sheet_name="Evaluation_Results", index=False)
    overall.T.to_excel(w, sheet_name="Overall_Metrics")
    by_cat.to_excel(w, sheet_name="By_Category")
    if "by_diff" in dir():
        by_diff.to_excel(w, sheet_name="By_Difficulty")
    tbl_stats.to_excel(w, sheet_name="Summary_Statistics")
    if "tbl_cat_diff" in dir():
        tbl_cat_diff.to_excel(w, sheet_name="Category_x_Difficulty")
print("Report saved:", report_xlsx)
"""
)

md(
    """## Paper citation checklist

| Figure file | Description |
|-------------|-------------|
| `figA_benchmark_composition` | Benchmark category/difficulty distribution |
| `figB_overall_metrics` | Mean Ollama RAG scores |
| `figC_category_heatmap` | Scores by evaluation category |
| `figD_retrieval_by_category` | Keyword retrieval coverage |
| `figE_judge_metrics` | Ollama judge dimensions |
| `figF_safety` | Safety regex + judge distribution |
| `figG_correctness_by_difficulty` | Boxplot by difficulty |
| `figH_metric_distributions` | Histograms per metric |
| `figI_metric_correlation` | Metric correlation heatmap |
| `figJ_retrieval_vs_faithfulness` | Retrieval vs faithfulness scatter |

**Tables:** `results/ollama_100q_evaluation_report.xlsx`, `ollama_100q_table_by_category.csv`
"""
)

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
