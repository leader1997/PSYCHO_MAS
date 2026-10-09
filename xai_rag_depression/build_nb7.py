#!/usr/bin/env python3
"""Build 07_ragas_benchmark_evaluation.ipynb."""

import json
from pathlib import Path

NB = Path(__file__).parent / "07_ragas_benchmark_evaluation.ipynb"


def md(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines or [text]}


def code(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines or [text],
    }


cells = [
    md(
        """# RAGAS Evaluation of the 100-Question Mental-Health RAG Benchmark

**Purpose:** Paper-aligned RAG evaluation using six RAGAS metrics (Batal et al., 2026).

**Generation:** Ollama | **Safety judge:** Ollama | **RAGAS LLM + embeddings:** Ollama (`nomic-embed-text`)

Set `XAI_RAGAS_BACKEND=openai` + `OPENAI_API_KEY` to use GPT-4o for RAGAS instead.

**Companion script:** `python run_nb7.py`

**Outputs:** `outputs/rag_eval_ragas_results.csv`, `outputs/rag_eval_ragas_summary.csv`, `outputs/ragas_eval/figures/`"""
    ),
    md(
        """## 1. References

**Batal, M.; Tace, Y.; Bensag, H.; El Filali, S.; Tabaa, M.** Time-Series Modeling and LLM-Based Agents for Peak Energy Management in Smart Campus Environments. *Sustainability* **2026**, *18*(2), 875. https://doi.org/10.3390/su18020875

**Es, S.; et al.** RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217, 2023.

We adopt the six metrics from Batal et al. §6.3 (implemented via [RAGAS](https://github.com/explodinggradients/ragas)):

| Metric | Definition (summary) |
|--------|----------------------|
| **Context precision** | Relevant retrieved chunks ranked highly |
| **Faithfulness** | Answer claims supported by retrieved context |
| **Answer relevancy** | Answer addresses the question |
| **Context recall** | Retrieved context covers reference facts |
| **Answer correctness** | Factual overlap with ground-truth reference |
| **Semantic similarity** | Embedding similarity to ground-truth reference |

**Supplementary (domain-specific, not in paper):** chunk-level P@k/R@k vs gold chunk IDs; safety/refusal scores for mental-health guardrails."""
    ),
    md("""## 2. Install RAGAS dependencies (run once per kernel)

Pinned versions avoid `langchain_openai` / Pydantic v2 conflicts on Python 3.11."""),
    code(
        """import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", "reproducibility/requirements-ragas.txt",
])
print("RAGAS dependencies installed.")"""
    ),
    md("""## 3. Setup"""),
    code(
        """import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

import llm_client
import rag_benchmark_eval as rbe
import ragas_eval as reval

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
plt.rcParams["figure.dpi"] = 120

ROOT = Path(".")
OUT = ROOT / "outputs"
FIG = OUT / "ragas_eval" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BENCHMARK_LIMIT = int(os.environ.get("XAI_BENCHMARK_LIMIT", "100"))

os.environ.setdefault("XAI_LLM_BACKEND", "ollama")
os.environ.setdefault("XAI_JUDGE_BACKEND", "ollama")
os.environ.setdefault("XAI_RAGAS_BACKEND", "ollama")

print("Ollama LLM stack:")
print(f"  generation: {llm_client.backend_name()}/{llm_client.model_name()}")
print(f"  judge:      {llm_client.judge_backend_name()}/{llm_client.judge_model_name()}")
print(f"  RAGAS LLM:  {reval.ragas_backend_name()}/{reval.ragas_model_name()}")
print(f"  RAGAS embed:{reval.ragas_embedding_model_name()}")
print(f"  installed:  {llm_client.list_ollama_models()}")
print(f"Benchmark limit: {BENCHMARK_LIMIT}")"""
    ),
    md("""## 4. Load benchmark (100 questions, 10 categories)"""),
    code(
        """benchmark_df = rbe.build_benchmark()
print(f"Questions: {len(benchmark_df)}")
display(benchmark_df.groupby("category").size().to_frame("n"))
display(benchmark_df.answer_type.value_counts().to_frame("n"))
benchmark_df.head(2)"""
    ),
    md("""## 5. Chunk retrieval baseline (gold chunk P@k / R@k)"""),
    code(
        """retrieval_df = rbe.compute_retrieval_metrics(benchmark_df)
print("Chunk-level retrieval (gold_relevant_chunk_ids):")
for k in (1, 5, 10):
    print(f"  P@{k}={retrieval_df[f'precision@{k}'].mean():.3f}  R@{k}={retrieval_df[f'recall@{k}'].mean():.3f}")
retrieval_df.head(3)"""
    ),
    md(
        """## 6. Generate answers (Ollama) + RAGAS scoring (Ollama)

Requires Ollama with chat + `nomic-embed-text`. Set `XAI_BENCHMARK_LIMIT=5` for a quick smoke test. Results resume from `outputs/rag_eval_full_results.csv`."""
    ),
    code(
        """results_df = pd.DataFrame()
if llm_client.is_available() and reval.is_ragas_available():
    results_df = rbe.run_full_benchmark(limit=BENCHMARK_LIMIT, run_ragas=True)
    print(f"Saved -> {rbe.FULL_RESULTS_CSV}")
else:
    if rbe.FULL_RESULTS_CSV.exists():
        results_df = pd.read_csv(rbe.FULL_RESULTS_CSV)
        if reval.RAGAS_RESULTS_CSV.exists():
            ragas_part = pd.read_csv(reval.RAGAS_RESULTS_CSV)
            results_df = results_df.merge(ragas_part, on="question_id", how="left", suffixes=("", "_r"))
        print("Loaded cached results.")
    else:
        print("Skipped — start Ollama and pull chat + nomic-embed-text models.")
results_df.head(2) if not results_df.empty else results_df"""
    ),
    md("""## 7. RAGAS results — Table 1 (overall means)"""),
    code(
        """ragas_df = pd.read_csv(reval.RAGAS_RESULTS_CSV) if reval.RAGAS_RESULTS_CSV.exists() else results_df
summaries = reval.summarize_ragas(ragas_df)
if summaries:
    display(Markdown("**Table 1 — RAGAS overall means (Batal et al. 2026 Table 2 layout)**"))
    display(summaries["overall"])
else:
    print("No RAGAS scores yet.")"""
    ),
    md("""## 8. RAGAS results — Table 2 (by category)"""),
    code(
        """if summaries and not summaries["by_category"].empty:
    display(Markdown("**Table 2 — RAGAS means by benchmark category**"))
    display(summaries["by_category"])
else:
    print("No per-category RAGAS scores.")"""
    ),
    md("""## 9. Figure — six RAGAS metrics"""),
    code(
        """if summaries:
    cols = [c for c in reval.RAGAS_METRICS if c in summaries["overall"].columns]
    ax = summaries["overall"][cols].T.plot.bar(figsize=(11, 5), legend=False, rot=45)
    ax.set_title("RAGAS evaluation metrics (Batal et al. 2026)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG / "fig_ragas_overall.png", bbox_inches="tight")
    plt.show()
    print(f"Saved {FIG / 'fig_ragas_overall.png'}")"""
    ),
    md("""## 10. Supplementary — safety and chunk retrieval"""),
    code(
        """if not results_df.empty:
    supp_cols = [c for c in ["safety_score", "refusal_appropriateness", "precision@5", "recall@5"] if c in results_df.columns]
    if supp_cols:
        display(Markdown("**Supplementary domain metrics (not in Batal et al.)**"))
        display(results_df[supp_cols].mean().round(4).to_frame("mean").T)"""
    ),
    md("""## 11. Sample Q&A with RAGAS scores"""),
    code(
        """if not results_df.empty and reval.RAGAS_METRICS[0] in results_df.columns:
    for atype in results_df["answer_type"].dropna().unique():
        ex = results_df[results_df.answer_type == atype].iloc[0]
        print(f"\\n=== {ex['question_id']} ({atype}) ===")
        print(f"Q: {ex['question'][:200]}...")
        print(f"A: {str(ex['answer'])[:400]}...")
        scores = ", ".join(
            f"{m}={ex[m]:.3f}" for m in reval.RAGAS_METRICS if m in ex and pd.notna(ex[m])
        )
        print(f"RAGAS: {scores}")"""
    ),
    md(
        """## 12. Reproducibility

```bash
cd xai_rag_depression
pip install -r reproducibility/requirements-ragas.txt
ollama pull llama3
ollama pull nomic-embed-text
# .env: XAI_LLM_BACKEND=ollama, XAI_RAGAS_BACKEND=ollama
python run_nb7.py
# smoke test: XAI_BENCHMARK_LIMIT=3 python run_nb7.py
```

**Output files:**
- `outputs/rag_eval_benchmark.csv` — 100 labeled questions
- `outputs/rag_eval_full_results.csv` — answers + RAGAS + supplementary
- `outputs/rag_eval_ragas_results.csv` — per-question RAGAS scores
- `outputs/rag_eval_ragas_summary.csv` — overall means
- `outputs/rag_eval_ragas_by_category.csv` — per-category means"""
    ),
    code('print("Notebook 7 complete.")'),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB}")
