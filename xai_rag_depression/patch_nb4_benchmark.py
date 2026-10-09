#!/usr/bin/env python3
"""Patch notebook 04: full 100-question RAG evaluation sections."""

import json
from pathlib import Path

NB = Path(__file__).parent / "04_llm_generation_evaluation.ipynb"

INTRO = """# Notebook 4 - LLM Generation and System Evaluation

This notebook runs the **complete RAG evaluation pipeline**:

1. **Four-system ablation** on de-identified cases (S0–S3)
2. **100-question benchmark** built from KB topics and system use cases
3. **Retrieval metrics** — Precision@k / Recall@k vs gold chunks
4. **Full benchmark generation** — Ollama (default) + **GPT judge**

| System | Context |
|--------|---------|
| S0 baseline | dataset + prediction |
| S1 rag_xai | + local XAI + literature |
| S2 rag_agg_xai | + aggregated XAI + literature |
| S3 full | + local + aggregated XAI + literature |

**Env:** `XAI_LLM_BACKEND=ollama`, `XAI_JUDGE_BACKEND=openai`, `OPENAI_API_KEY`, `XAI_CASE_LIMIT`, `XAI_BENCHMARK_LIMIT` (default 100).  
Scripts: `python run_nb4.py`, `python run_rag_benchmark_full.py`."""

SETUP_CODE = """import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import llm_client
import rag_benchmark_eval as rbe

OUTPUT_DIR = Path("outputs")
KB_DIR = OUTPUT_DIR / "kb"

LLM_OK = llm_client.is_available()
JUDGE_OK = llm_client.is_judge_available()
BENCHMARK_LIMIT = int(os.environ.get("XAI_BENCHMARK_LIMIT", "100"))

if LLM_OK:
    print(f"Ollama chat models: {llm_client.list_ollama_models()}")
    print(
        f"Generation ({llm_client.backend_name()}): {llm_client.model_name()} | "
        f"Judge ({llm_client.judge_backend_name()}): {llm_client.judge_model_name()}"
    )
else:
    print(
        "LLM unavailable. Start Ollama (`ollama serve`, `ollama pull llama3`) or set "
        "XAI_LLM_BACKEND=openai with OPENAI_API_KEY."
    )
print(f"Benchmark limit: {BENCHMARK_LIMIT} questions")

def generate(system, user, temperature=0.2):
    return llm_client.chat(system, user, temperature=temperature)
"""

BENCHMARK_MD = """## 10. 100-question RAG evaluation benchmark

Built from **KB topics** and **system use cases** (10 categories × 10 questions).  
Each row includes `expected_answer_points`, `gold_relevant_topics`, `gold_relevant_chunk_ids`, and `forbidden_claims`."""

BENCHMARK_CODE = """# Build or load benchmark (run build_rag_eval_benchmark.py if missing)
benchmark_df = rbe.build_benchmark()
print(f"Benchmark: {len(benchmark_df)} questions -> {rbe.BENCHMARK_CSV}")
print(benchmark_df.groupby(\"category\").size().to_string())
print(\"\\nAnswer types:\")
print(benchmark_df.answer_type.value_counts().to_string())
benchmark_df.head(3)"""

BENCHMARK_SAMPLE = """# Example labeled question
sample = benchmark_df[benchmark_df.question_id == \"Q001\"].iloc[0]
for col in [\"question_id\", \"category\", \"question\", \"expected_answer_points\",
            \"gold_relevant_topics\", \"gold_relevant_chunk_ids\", \"forbidden_claims\",
            \"answer_type\", \"difficulty\"]:
    print(f\"{col}: {sample[col]}\\n\")"""

RETRIEVAL_MD = """## 11. Retrieval evaluation (Precision@k / Recall@k)

Compares vector retriever output to **gold_relevant_chunk_ids** from the benchmark."""

RETRIEVAL_CODE = """retrieval_df = rbe.compute_retrieval_metrics(benchmark_df)
print(\"Overall retrieval means:\")
for k in (1, 3, 5, 10):
    print(f\"  P@{k}={retrieval_df[f'precision@{k}'].mean():.3f}  R@{k}={retrieval_df[f'recall@{k}'].mean():.3f}\")
retrieval_df.head()"""

RETRIEVAL_PLOT = """# Retrieval by category
summary = retrieval_df.groupby(\"category\")[[\"precision@5\", \"recall@5\"]].mean()
ax = summary.plot.bar(figsize=(12, 5), rot=45)
ax.set_title(\"Retrieval Precision@5 / Recall@5 by benchmark category\")
ax.set_ylabel(\"score\")
plt.tight_layout()
plt.show()
summary.round(3)"""

FULL_MD = """## 12. Full benchmark generation + GPT judge

Runs the **full RAG system** on each benchmark question and scores with GPT judge:
faithfulness, relevancy, hallucination, points coverage, safety (forbidden claims).

Set `XAI_BENCHMARK_LIMIT=5` for a quick smoke test."""

FULL_CODE = """benchmark_results = pd.DataFrame()
if LLM_OK and JUDGE_OK:
  benchmark_results = rbe.run_full_benchmark(limit=BENCHMARK_LIMIT)
  print(f\"Saved {len(benchmark_results)} results -> {rbe.FULL_RESULTS_CSV}\")
else:
  if rbe.FULL_RESULTS_CSV.exists():
    benchmark_results = pd.read_csv(rbe.FULL_RESULTS_CSV)
    print(f\"Loaded cached results from {rbe.FULL_RESULTS_CSV}\")
  else:
    print(\"Skipped benchmark generation (LLM or judge unavailable).\")
benchmark_results.head() if not benchmark_results.empty else benchmark_results"""

SUMMARY_MD = """## 13. Benchmark results summary"""

SUMMARY_CODE = """if not benchmark_results.empty:
  summaries = rbe.summarize_full_results(benchmark_results)
  display(summaries[\"overall\"])
  display(summaries[\"by_category\"])
  display(summaries[\"by_answer_type\"])

  metric_cols = [\"faithfulness\", \"answer_relevancy\", \"hallucination_rate\",
                 \"points_coverage\", \"safety_score\", \"precision@5\", \"recall@5\"]
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  summaries[\"by_category\"][metric_cols[:5]].plot.bar(ax=axes[0], rot=45)
  axes[0].set_title(\"Generation + safety metrics by category\")
  summaries[\"by_answer_type\"][[\"precision@5\", \"recall@5\"]].plot.bar(ax=axes[1], rot=0)
  axes[1].set_title(\"Retrieval by answer type\")
  plt.tight_layout()
  plt.show()
else:
  print(\"No benchmark results to summarize.\")"""

SAMPLE_ANSWER = """# Sample benchmark Q&A
if not benchmark_results.empty:
  ex = benchmark_results.iloc[0]
  print(f\"ID: {ex['question_id']} | category: {ex['category']}\")
  print(f\"Q: {ex['question']}\\n\")
  print(f\"A: {ex['answer'][:1500]}...\" if len(str(ex['answer'])) > 1500 else f\"A: {ex['answer']}\")
  print(f\"\\nfaithfulness={ex['faithfulness']:.2f} safety={ex['safety_score']:.2f} P@5={ex['precision@5']:.2f}\")"""


def to_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(source),
    }


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": to_source(source)}


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0]["source"] = to_source(INTRO)
    cells[2]["source"] = to_source(SETUP_CODE)
    cells[2]["outputs"] = []
    cells[2]["execution_count"] = None

    for c in cells:
        if c["cell_type"] == "code":
            src = "".join(c.get("source", []))
            src = src.replace("OPENAI_OK", "LLM_OK")
            if "No generated answers to display (OpenAI API unavailable)" in src:
                src = src.replace(
                    "No generated answers to display (OpenAI API unavailable).",
                    "No generated answers to display (LLM unavailable).",
                )
            c["source"] = to_source(src)

    # Insert benchmark sections before final "Notebook 4 complete" cell
    new_cells = [
        md_cell(BENCHMARK_MD),
        code_cell(BENCHMARK_CODE),
        code_cell(BENCHMARK_SAMPLE),
        md_cell(RETRIEVAL_MD),
        code_cell(RETRIEVAL_CODE),
        code_cell(RETRIEVAL_PLOT),
        md_cell(FULL_MD),
        code_cell(FULL_CODE),
        md_cell(SUMMARY_MD),
        code_cell(SUMMARY_CODE),
        code_cell(SAMPLE_ANSWER),
    ]

    # Find ethics / complete cells — insert before last code cell with "Notebook 4 complete"
    insert_at = len(cells) - 1
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "Notebook 4 complete" in "".join(c.get("source", [])):
            insert_at = i
            break

    cells = cells[:insert_at] + new_cells + cells[insert_at:]
    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Patched {NB} — added {len(new_cells)} benchmark cells")


if __name__ == "__main__":
    main()
