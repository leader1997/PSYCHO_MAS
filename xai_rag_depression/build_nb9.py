#!/usr/bin/env python3
"""Build 09_ragas_retriever_evaluation.ipynb."""

import json
from pathlib import Path

NB = Path(__file__).parent / "09_ragas_retriever_evaluation.ipynb"


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
        """# RAGAS Evaluation — Retriever Ablation (3 retrievers)

Compare **three retrieval strategies** on the 100-question benchmark with **fully local Ollama** for generation, safety judge, RAGAS LLM, and RAGAS embeddings.

| Role | Backend | Env / model |
|------|---------|-------------|
| Answer generation | Ollama | `XAI_LLM_BACKEND=ollama`, `XAI_OLLAMA_MODEL` |
| Safety judge | Ollama | `XAI_JUDGE_BACKEND=ollama`, `XAI_OLLAMA_JUDGE_MODEL` |
| RAGAS evaluator LLM | Ollama | `XAI_RAGAS_BACKEND=ollama`, `XAI_RAGAS_MODEL` (defaults to generation model) |
| RAGAS embeddings | Ollama | `XAI_RAGAS_EMBEDDING_MODEL=nomic-embed-text` (`ollama pull nomic-embed-text`) |

| Retriever | Method |
|-----------|--------|
| `dense_minilm` | Sentence-transformer cosine (`all-MiniLM-L6-v2`) — current default |
| `bm25` | Sparse lexical BM25Okapi |
| `hybrid_rrf` | Reciprocal rank fusion (dense + BM25) |

**Companion script:** `python run_nb9.py`

**Outputs:** `outputs/ragas_retriever_eval/<retriever_id>/` + `ragas_retriever_comparison.csv`"""
    ),
    md("""## 1. Install dependencies"""),
    code(
        """import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "rank-bm25",
    "-r", "reproducibility/requirements-ragas.txt",
])
print("Dependencies OK.")"""
    ),
    md("""## 2. Setup"""),
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
import rag_retriever_eval as rre
import ragas_eval as reval
from retrievers import RETRIEVER_SPECS

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
plt.rcParams["figure.dpi"] = 120

os.environ.setdefault("XAI_LLM_BACKEND", "ollama")
os.environ.setdefault("XAI_JUDGE_BACKEND", "ollama")
os.environ.setdefault("XAI_RAGAS_BACKEND", "ollama")
BENCHMARK_LIMIT = int(os.environ.get("XAI_BENCHMARK_LIMIT", "100"))
OUT = Path("outputs/ragas_retriever_eval")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print("Retrievers:")
for rid, label in RETRIEVER_SPECS:
    print(f"  {rid}: {label}")
print("\\nOllama LLM stack:")
print(f"  generation: {llm_client.backend_name()}/{llm_client.model_name()}")
print(f"  judge:      {llm_client.judge_backend_name()}/{llm_client.judge_model_name()}")
print(f"  RAGAS LLM:  {reval.ragas_backend_name()}/{reval.ragas_model_name()}")
print(f"  RAGAS embed:{reval.ragas_embedding_model_name()}")
print(f"  installed:  {llm_client.list_ollama_models()}")
print(f"  benchmark limit={BENCHMARK_LIMIT}")"""
    ),
    md("""## 3. Chunk retrieval P@5 by retriever (gold chunk IDs)"""),
    code(
        """retrieval_parts = []
for rid, _ in RETRIEVER_SPECS:
    m = rre.compute_retrieval_metrics_for_retriever(rid, rbe.build_benchmark().head(BENCHMARK_LIMIT))
    m["retriever"] = rid
    retrieval_parts.append(m)
retrieval_all = pd.concat(retrieval_parts, ignore_index=True)
p5 = retrieval_all.groupby("retriever")["precision@5"].mean().round(4)
print("Mean P@5 by retriever:")
print(p5.to_string())
retrieval_all.head(3)"""
    ),
    md(
        """## 4. Generate answers + RAGAS (3 retrievers)

Requires Ollama with chat + embedding models (`ollama pull nomic-embed-text`). Set `XAI_BENCHMARK_LIMIT=5` for a smoke test."""
    ),
    code(
        """comparison_df = pd.DataFrame()
if llm_client.is_available() and reval.is_ragas_available():
    comparison_df = rre.run_all_retrievers(
        limit=BENCHMARK_LIMIT,
        skip_existing=True,
        run_ragas=True,
    )
    print(f"Comparison -> {rre.COMPARISON_SUMMARY_CSV}")
else:
    comparison_df = rre.summarize_comparison()
    if comparison_df.empty:
        print("Skipped — start Ollama and pull chat + nomic-embed-text models.")
comparison_df"""
    ),
    md("""## 5. Table — RAGAS overall means by retriever"""),
    code(
        """if comparison_df.empty:
    comparison_df = rre.summarize_comparison()
if not comparison_df.empty:
    display(Markdown("**RAGAS means (Batal et al. 2026) by retriever**"))
    cols = ["retriever", "retriever_label"] + [c for c in reval.RAGAS_METRICS if c in comparison_df.columns]
    display(comparison_df[cols])
else:
    print("No RAGAS comparison yet.")"""
    ),
    md("""## 6. Figure — six RAGAS metrics × 3 retrievers"""),
    code(
        """if not comparison_df.empty:
    metrics = [c for c in reval.RAGAS_METRICS if c in comparison_df.columns]
    melted = comparison_df.melt(
        id_vars=["retriever", "retriever_label"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="retriever", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("RAGAS by retriever (fully local Ollama stack)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "fig_ragas_by_retriever.png", bbox_inches="tight")
    plt.show()
    print(f"Saved {FIG / 'fig_ragas_by_retriever.png'}")
else:
    print("No data for figure.")"""
    ),
    md("""## 7. Reproducibility

```bash
cd xai_rag_depression
pip install rank-bm25 -r reproducibility/requirements-ragas.txt
ollama pull llama3
ollama pull nomic-embed-text
# smoke: XAI_BENCHMARK_LIMIT=3 python run_nb9.py
python run_nb9.py
```

Per-retriever outputs under `outputs/ragas_retriever_eval/<retriever_id>/`."""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB}")
