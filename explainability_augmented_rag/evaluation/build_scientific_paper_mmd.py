#!/usr/bin/env python3
"""Aggregate all Phase 1 + Phase 2 results into one scientific-paper MMD file."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT.parent / "xai_rag_depression"
EXPLORATION = ROOT.parent / "exploration_outputs"
OUT = ROOT / "paper_outputs" / "SCIENTIFIC_PAPER_RESULTS.mmd"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _df_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No data yet — run `python scripts/run_pipeline.py`._\n"
    d = df.head(max_rows)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[c])[:80] for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows omitted)_")
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path, n: int = 1) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:n]


def build_mmd() -> str:
    results = ROOT / "results"
    data = ROOT / "data"
    chunks = _read_csv(data / "processed_chunks.csv")
    meta = _read_csv(data / "corpus_metadata.csv")
    corpus_n = len(chunks)
    doc_n = len(meta)

    ml = _read_csv(EXPLORATION / "all_model_comparison.csv")
    bootstrap = _read_csv(PHASE1 / "outputs/strengthened/table_b_bootstrap_ci.csv")
    calibration = _read_csv(PHASE1 / "outputs/strengthened/table_c_calibration.csv")
    faithfulness_ml = _read_csv(PHASE1 / "outputs/strengthened/table_e_faithfulness_controls.csv")
    phase1_ragas = _read_csv(PHASE1 / "outputs/rag_eval_ragas_summary.csv")
    phase1_ragas_cat = _read_csv(PHASE1 / "outputs/rag_eval_ragas_by_category.csv")
    phase1_retrieval = _read_csv(PHASE1 / "outputs/rag_eval_retrieval_summary.csv")
    phase1_ablation = _read_csv(PHASE1 / "outputs/rag_ablation_evaluation.csv")

    retriever = _read_csv(results / "retriever_ablation_summary.csv")
    ragas = _read_csv(results / "ragas_summary_table.csv")
    ragas_full = _read_csv(results / "ragas_results.csv")
    ablation = _read_csv(results / "ablation_summary_table.csv")
    safety = _read_csv(results / "safety_reports.csv")
    faith = _read_csv(results / "citation_precision_results.csv")
    claim_faith = _read_csv(results / "claim_level_faithfulness.csv")
    errors = _read_csv(results / "error_analysis.csv")

    narratives = _read_jsonl(results / "corrected_narratives.jsonl", 1)
    syntheses = _read_jsonl(results / "evidence_summaries.jsonl", 1)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    api_note = "gpt-4o" if (ROOT / ".env").exists() else "configured in .env"

    parts = [
        "---",
        "title: Explainability-Augmented RAG for Depression Screening — Paper Results",
        f"date: {ts}",
        "---",
        "",
        "# Explainability-Augmented RAG for Depression/Anxiety Screening Communication",
        "",
        "> **Disclaimer:** This system produces **non-clinical screening narratives**. "
        "It does **not** diagnose depression or anxiety. All outputs require human review before any deployment.",
        "",
        f"_Generated: {ts} | LLM backend: OpenAI {api_note} | Pipeline: explainability_augmented_rag Phase 2_",
        "",
        "---",
        "",
        "## 1. Study overview",
        "",
        "```mermaid",
        "flowchart TB",
        "  HMS[HMS 2024-2025 Survey] --> ML[9 ML/DL Models]",
        "  ML --> XAI[SHAP / IG / Coefficients]",
        "  XAI --> Cards[XAI Cards]",
        "  PDFs[27 Literature PDFs] --> KB[Knowledge Base]",
        "  Cards --> RAG[Explainability-Augmented RAG]",
        "  KB --> RAG",
        "  RAG --> Narrative[Safe Screening Narrative]",
        "  Narrative --> Eval[RAGAS + Ablation + Safety]",
        "```",
        "",
        "### 1.1 Cohort and target",
        "",
        "| Item | Value |",
        "|------|-------|",
        "| Dataset | Healthy Minds Study 2024–2025 |",
        "| Cleaned respondents | ~63,581 |",
        "| Features | 25 survey variables |",
        "| Target | `dep_or_anx` (elevated depression/anxiety screening flag) |",
        "| Positive rate | ~44.9% |",
        "| Split | 80% / 10% / 10% stratified |",
        "",
        "---",
        "",
        "## 2. Predictive modeling (Phase 1)",
        "",
        "### Table 1 — Model comparison (test set)",
        "",
        _df_md(ml),
        "",
        "### Table 2 — Bootstrap 95% CI (XGBoost exemplar)",
        "",
        _df_md(bootstrap.head(8)),
        "",
        "### Table 3 — Calibration (if available)",
        "",
        _df_md(calibration.head(10)),
        "",
        "### Table 4 — XAI faithfulness controls (feature deletion AUC drop)",
        "",
        _df_md(faithfulness_ml),
        "",
        "---",
        "",
        "## 3. Knowledge base (Phase 2)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Processed chunks | {corpus_n:,} |",
        f"| Source documents | {doc_n:,} |",
        f"| Literature PDFs | {len(list((ROOT / 'data' / 'raw_pdfs').glob('*.pdf')))} |",
        "| Chunking | ~450 tokens, 75 overlap, page-aware |",
        "| Dense embeddings | text-embedding-3-small (OpenAI) |",
        "",
        "### Source type distribution",
        "",
        _df_md(meta.groupby("source_type").size().reset_index(name="count") if not meta.empty else pd.DataFrame()),
        "",
        "---",
        "",
        "## 4. Retrieval ablation (Phase 2)",
        "",
        "### Table 5 — Retriever performance (100-question benchmark)",
        "",
        _df_md(retriever),
        "",
        "### Table 5b — Phase 1 dense retrieval by category (reference)",
        "",
        _df_md(phase1_retrieval),
        "",
        "```mermaid",
        "xychart-beta",
        '    title "Precision@5 by retriever"',
        '    x-axis ["BM25", "Dense", "Hybrid", "Hybrid+Rerank"]',
    ]

    if not retriever.empty and "precision@5" in retriever.columns:
        vals = []
        for r in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
            row = retriever[retriever["retriever"] == r]
            vals.append(str(round(float(row["precision@5"].iloc[0]), 3)) if not row.empty else "0")
        parts.append(f'    y-axis "P@5" 0 --> 1')
        parts.append(f"    bar [{', '.join(vals)}]")
    parts.append("```")
    parts.extend([
        "",
        "---",
        "",
        "## 5. RAGAS evaluation (Batal et al. 2026 aligned)",
        "",
        "Six metrics: context precision, faithfulness, answer relevancy, context recall, "
        "answer correctness, semantic similarity.",
        "",
        "### Table 6 — RAGAS overall means (Phase 2, systems S1–S6)",
        "",
        _df_md(ragas),
        "",
        "### Table 6b — Phase 1 RAGAS overall (Ollama generation, reference)",
        "",
        _df_md(phase1_ragas),
        "",
        "### Table 6c — Phase 1 RAGAS by category",
        "",
        _df_md(phase1_ragas_cat.head(15)),
        "",
    ])

    if not ragas_full.empty and "note" in ragas_full.columns and (ragas_full["note"] == "stub_no_api").all():
        parts.append(
            "\n> **Note:** Phase 2 RAGAS table used stub scores. Re-run with valid OPENAI_API_KEY.\n"
        )
    elif ragas.empty and (results / "pipeline_resume.log").exists():
        log = (results / "pipeline_resume.log").read_text(encoding="utf-8", errors="ignore")
        if "Running S" in log and "Resume complete" not in log:
            parts.append(
                "\n> **Note:** Phase 2 RAGAS evaluation in progress. "
                "Re-run `python evaluation/build_scientific_paper_mmd.py` after pipeline completes.\n"
            )

    parts.extend([
        "---",
        "",
        "## 6. System ablation",
        "",
        "### Table 7 — Phase 2 ablation A1–A8 (3 de-identified cases, GPT-4o judge)",
        "",
        _df_md(ablation),
        "",
        "### Table 7b — Phase 1 four-system ablation (Ollama + GPT judge, reference)",
        "",
        _df_md(phase1_ablation),
        "",
        "```mermaid",
        "flowchart LR",
        "  A1[A1 Profile] --> A2[A2 + Prediction]",
        "  A2 --> A3[A3 + Local XAI]",
        "  A3 --> A7[A7 + RAG]",
        "  A7 --> A8[A8 + Safety]",
        "```",
        "",
        "---",
        "",
        "## 7. Safety and faithfulness",
        "",
        "### Table 8 — Safety validator results",
        "",
    ])

    if not safety.empty:
        pass_rate = (safety.groupby("case_id")["overall_safety"].last() == "pass").mean()
        parts.append(f"**Overall safety pass rate (final attempt):** {pass_rate:.1%}\n")
    parts.append(_df_md(safety.head(15)))

    parts.extend([
        "",
        "### Table 9 — Citation / claim faithfulness",
        "",
        _df_md(faith),
        "",
        "### Table 9b — Claim-level faithfulness (sample)",
        "",
        _df_md(claim_faith.head(15)),
        "",
        "---",
        "",
        "## 8. Error analysis",
        "",
        "### Table 10 — Failure taxonomy summary",
        "",
        _df_md(errors.head(25)),
        "",
        "---",
        "",
        "## 9. Illustrative outputs",
        "",
        "### 9.1 Example evidence synthesis (JSON excerpt)",
        "",
        "```json",
        json.dumps(syntheses[0], indent=2, ensure_ascii=False)[:2000] if syntheses else "{}",
        "```",
        "",
        "### 9.2 Example screening narrative (JSON excerpt)",
        "",
        "```json",
        json.dumps(narratives[0], indent=2, ensure_ascii=False)[:2000] if narratives else "{}",
        "```",
        "",
        "---",
        "",
        "## 10. Limitations (for Discussion section)",
        "",
        "1. Screening flag `dep_or_anx` is not a clinical diagnosis.",
        "2. XAI features are model-associated contributors, not established causes.",
        "3. Retrieval precision remains modest for some benchmark categories (e.g. case-based explanation).",
        "4. RAGAS scores depend on evaluator LLM and context packaging.",
        "5. Evaluation uses de-identified exemplar cases; generalization to all students is unproven.",
        "6. Human expert review is required before any real-world communication use.",
        "",
        "---",
        "",
        "## 11. Reproducibility",
        "",
        "```bash",
        "cd explainability_augmented_rag",
        "pip install -r requirements.txt",
        "python scripts/run_pipeline.py",
        "python evaluation/build_scientific_paper_mmd.py",
        "```",
        "",
        "| Output file | Description |",
        "|-------------|-------------|",
        "| `data/processed_chunks.csv` | Full knowledge base chunks |",
        "| `data/rag_benchmark_100_questions.csv` | 100-Q evaluation benchmark |",
        "| `results/retriever_ablation_summary.csv` | Retriever comparison |",
        "| `results/ragas_summary_table.csv` | RAGAS by system |",
        "| `results/ablation_summary_table.csv` | System ablation |",
        "| `results/safety_reports.csv` | Safety validation |",
        "| `paper_outputs/SCIENTIFIC_PAPER_RESULTS.mmd` | This document |",
        "",
    ])

    return "\n".join(parts)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = build_mmd()
    OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT} ({len(text):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
