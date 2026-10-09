"""Generate paper figures from results."""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from local_agentic_xai_rag.config_loader import ensure_dirs, path

sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    fig_dir = path("figures_dir")
    fig.savefig(fig_dir / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def build_all_figures() -> None:
    ensure_dirs()
    fig_dir = path("figures_dir")
    res = path("results_dir")

    # Fig 1 architecture placeholder
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        "Case+XAI → Template Queries → Hybrid Retrieval (llama3)\n"
        "→ Evidence Agent → Report Agent → Safety Rules + Claim Check → Human Review",
        ha="center", va="center", fontsize=11,
    )
    ax.set_title("Fig 1: Human-in-the-loop advisory pipeline (Ollama llama3 only)")
    _save(fig, "fig01_architecture.png")

    # Fig 2 retrieval
    p = res / "retrieval_eval_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(8, 4))
        df.set_index("retriever")[["precision@5", "recall@5", "recall@10"]].plot(kind="bar", ax=ax)
        ax.set_title("Fig 2: Retrieval — BM25 vs hybrid llama3")
        ax.set_ylim(0, max(0.15, df["recall@10"].max() * 1.2))
        _save(fig, "fig02_retrieval_bm25_vs_hybrid.png")

    # Fig 3 claim support
    p = res / "claim_level_faithfulness.csv"
    if p.exists():
        df = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(6, 4))
        df["support_status"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Fig 3: Claim support distribution")
        _save(fig, "fig03_claim_support_distribution.png")

    # Fig 4 safety violations
    p = res / "validation_reports.csv"
    if p.exists():
        df = pd.read_csv(p)
        check_cols = [c for c in df.columns if c not in ("case_id", "overall_safety", "revision_required", "comments")]
        fail_rates = (df[check_cols] == "fail").mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        fail_rates.plot(kind="barh", ax=ax, color="coral")
        ax.set_title("Fig 4: Safety check fail rates")
        _save(fig, "fig04_safety_violation_types.png")

    # Fig 5 system comparison
    p = res / "system_comparison_summary.csv"
    if p.exists():
        df = pd.read_csv(p, index_col=0)
        fig, ax = plt.subplots(figsize=(8, 4))
        df[["safety_pass", "claim_support_rate", "disclaimer_present"]].plot(kind="bar", ax=ax)
        ax.set_title("Fig 5: System comparison B1–B3")
        ax.set_ylim(0, 1.05)
        _save(fig, "fig05_system_comparison_b1_b3.png")

    # Fig 6 category heatmap (retrieval recall@5 by category)
    p = res / "retrieval_eval_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "category" in df.columns:
            pivot = (
                df.groupby(["category", "retriever"])["recall@5"]
                .mean()
                .unstack(fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.35)))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_title("Fig 6: Recall@5 by benchmark category")
            _save(fig, "fig06_benchmark_category_heatmap.png")

    print(f"Figures -> {fig_dir}")


def build_tables_xlsx() -> None:
    ensure_dirs()
    res = path("results_dir")
    out = path("paper_outputs") / "tables_for_paper.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        if (res / "retrieval_eval_summary.csv").exists():
            pd.read_csv(res / "retrieval_eval_summary.csv").to_excel(w, sheet_name="Retrieval", index=False)
        if (res / "system_comparison_summary.csv").exists():
            pd.read_csv(res / "system_comparison_summary.csv").to_excel(w, sheet_name="SystemComparison")
        if (res / "claim_level_faithfulness.csv").exists():
            pd.read_csv(res / "claim_level_faithfulness.csv").to_excel(w, sheet_name="Claims", index=False)
        if (res / "validation_reports.csv").exists():
            pd.read_csv(res / "validation_reports.csv").to_excel(w, sheet_name="Safety", index=False)
        pd.DataFrame([{"ollama_model": "llama3", "note": "chat + embeddings only"}]).to_excel(
            w, sheet_name="Config", index=False
        )
    print(f"Tables -> {out}")


def main() -> int:
    build_all_figures()
    build_tables_xlsx()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
