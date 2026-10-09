#!/usr/bin/env python3
"""Build 08_three_system_evaluation.ipynb."""

import json
from pathlib import Path

NB = Path(__file__).parent / "08_three_system_evaluation.ipynb"


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
        """# Three-System RAG Ablation Evaluation

**Purpose:** Compare three deployment configurations on de-identified student cases (Ollama generation).

| System | Context |
|--------|---------|
| **Sys-1** `local_xai` | User profile + prediction + **local SHAP** + RAG |
| **Sys-2** `agg_xai` | User profile + prediction + **aggregated XAI** + RAG |
| **Sys-3** `profile_rag` | User profile + prediction + RAG (**no XAI**) |

**Shared across all systems:** full 25-feature user profile, prediction, ethics block, same literature retrieval.

**Companion script:** `python run_nb8.py`

**Outputs:** `outputs/three_system_eval_answers.csv`, `outputs/three_system_eval_results.csv`"""
    ),
    md("""## 1. Setup"""),
    code(
        """import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

import llm_client
import three_system_eval as tse
from case_profiles import load_case_profiles

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
plt.rcParams["figure.dpi"] = 120

os.environ.setdefault("XAI_LLM_BACKEND", "ollama")

OUT = Path("outputs")
FIG = OUT / "three_system_eval" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print(f"Generation: {llm_client.backend_name()}/{llm_client.model_name()}")
print(f"Judge: {llm_client.judge_backend_name()}/{llm_client.judge_model_name()}")
print(f"Judge available: {llm_client.is_judge_available()}")"""
    ),
    md("""## 2. User profiles (25 HMS features, all systems)"""),
    code(
        """profiles = load_case_profiles()
for case_id, prof in profiles.items():
    print(f"\\n=== {case_id} ===")
    print(f"Prediction: {prof['predicted_class']} (p={prof['predicted_probability']})")
    shown = list(prof["feature_levels"].items())[:8]
    for f, lvl in shown:
        print(f"  {f}: {lvl}")
    print(f"  ... ({len(prof['feature_levels'])} features total)")"""
    ),
    md("""## 3. Run evaluation (3 cases × 3 systems)

Requires Ollama. GPT judge (`OPENAI_API_KEY`) scores faithfulness, relevancy, hallucination, context precision/recall."""),
    code(
        """results_df = pd.DataFrame()
answers_df = pd.DataFrame()

if llm_client.is_available():
    results_df = tse.run_three_system_eval(skip_existing=True)
    if tse.ANSWERS_CSV.exists():
        answers_df = pd.read_csv(tse.ANSWERS_CSV)
    print(f"Answers -> {tse.ANSWERS_CSV}")
    print(f"Results -> {tse.RESULTS_CSV}")
else:
    if tse.ANSWERS_CSV.exists():
        answers_df = pd.read_csv(tse.ANSWERS_CSV)
    if tse.RESULTS_CSV.exists():
        results_df = pd.read_csv(tse.RESULTS_CSV)
    print("Skipped live run — load cached CSVs or start Ollama.")

answers_df.head(3) if not answers_df.empty else answers_df"""
    ),
    md("""## 4. Table 1 — Overall means (LLM-as-judge)"""),
    code(
        """summaries = tse.summarize(results_df)
if summaries:
    display(Markdown("**Table 1 — Mean scores by system (higher faithfulness/relevancy/recall = better; lower hallucination = better)**"))
    display(summaries["overall"])
else:
    print("No judge scores yet.")"""
    ),
    md("""## 5. Table 2 — Per-case breakdown"""),
    code(
        """if summaries and "by_case" in summaries:
    display(Markdown("**Table 2 — Scores by case and system**"))
    display(summaries["by_case"])
else:
    print("No per-case scores.")"""
    ),
    md("""## 6. Figure — system comparison"""),
    code(
        """if summaries and not summaries["overall"].empty:
    df = summaries["overall"].reset_index()
    plot_metrics = [c for c in ["faithfulness", "answer_relevancy", "context_recall"] if c in df.columns]
    melted = df.melt(id_vars="system", value_vars=plot_metrics, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=melted, x="metric", y="score", hue="system", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Three-system ablation (Ollama answers, GPT judge)")
    plt.tight_layout()
    plt.savefig(FIG / "fig_three_system_comparison.png", bbox_inches="tight")
    plt.show()
    print(f"Saved {FIG / 'fig_three_system_comparison.png'}")
else:
    print("No data for figure.")"""
    ),
    md("""## 7. Sample answers — high-risk case"""),
    code(
        """if not answers_df.empty:
    sub = answers_df[answers_df.case_id == "high_risk_case"]
    for _, row in sub.iterrows():
        print(f"\\n{'='*60}")
        print(f"{row['system_label']}")
        print(row["answer"][:1200])
        if len(str(row["answer"])) > 1200:
            print("...")
else:
    print("No answers loaded.")"""
    ),
    md("""## 8. Reproducibility

```bash
cd xai_rag_depression
# .env: XAI_LLM_BACKEND=ollama, OPENAI_API_KEY (for judge)
python run_nb8.py
```

**Files:**
- `outputs/case_profiles.json` — full 25-feature profiles
- `outputs/three_system_eval_answers.csv` — long format
- `outputs/three_system_eval_answers_wide.csv` — one row per case
- `outputs/three_system_eval_results.csv` — judge scores
- `outputs/three_system_eval_summary.csv` — overall means"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB}")
