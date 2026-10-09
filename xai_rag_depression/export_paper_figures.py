#!/usr/bin/env python3
"""Re-export all paper figures to outputs/figures/paper/."""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from math import pi

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")

perf = pd.read_csv(OUTPUT_DIR / "model_performance.csv")
agg_imp = pd.read_csv(OUTPUT_DIR / "aggregated_importance.csv")
stab = pd.read_csv(OUTPUT_DIR / "stability_scores.csv")
faith = pd.read_csv(OUTPUT_DIR / "faithfulness.csv")
global_imp = pd.read_csv(OUTPUT_DIR / "global_importance_per_model.csv", index_col=0)
rag_eval = pd.read_csv(OUTPUT_DIR / "rag_ablation_evaluation.csv")
pdf_inv = pd.read_csv(OUTPUT_DIR / "kb" / "pdf_inventory.csv")
kb_stats = json.load(open(OUTPUT_DIR / "kb" / "kb_stats.json"))
agg_card = json.load(open(OUTPUT_DIR / "cards" / "aggregated_explanation.json"))
local_cases = json.load(open(OUTPUT_DIR / "local_explanations.json"))

SYSTEM_ORDER = ["baseline", "rag_xai", "rag_agg_xai", "full"]
SYSTEM_LABELS = {
    "baseline": "S0: Baseline RAG",
    "rag_xai": "S1: RAG + local XAI",
    "rag_agg_xai": "S2: RAG + aggregated XAI",
    "full": "S3: Full system",
}
family_color = dict(zip(perf["model"], perf["family"].map({"ML": "#4C72B0", "DL": "#DD8452"})))
metric_cols = ["faithfulness", "answer_relevancy", "hallucination_rate", "context_precision", "context_recall"]

# Fig 1
perf_sorted = perf.sort_values("auc", ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(perf_sorted["model"], perf_sorted["auc"],
        color=perf_sorted["family"].map({"ML": "#4C72B0", "DL": "#DD8452"}))
ax.set_xlim(0.84, 0.856); ax.set_xlabel("Test-set AUC")
ax.set_title("Figure 1. Predictive performance across 9 models")
plt.tight_layout(); savefig("fig01_auc_by_model.png"); plt.close()

# Fig 4
top_feats = agg_imp.nlargest(12, "final_importance")["feature"].tolist()
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(global_imp.loc[top_feats], annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
ax.set_title("Figure 4. Top-12 features: importance heatmap")
plt.tight_layout(); savefig("fig04_importance_heatmap.png"); plt.close()

# Fig 14
plot_df = rag_eval.copy()
plot_df["system"] = plot_df["system"].map(SYSTEM_LABELS)
heat = plot_df.set_index(["case", "system"])[metric_cols]
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
ax.set_title("Figure 14. Four-system ablation heatmap")
plt.tight_layout(); savefig("fig14_ablation_heatmap.png"); plt.close()

# Fig 15
summary = rag_eval.groupby("system")[metric_cols].mean().reindex(SYSTEM_ORDER)
summary.index = [SYSTEM_LABELS[s] for s in summary.index]
fig, ax = plt.subplots(figsize=(12, 6))
summary[["faithfulness", "answer_relevancy", "hallucination_rate"]].plot.bar(ax=ax, rot=0)
ax.set_ylim(0, 1); ax.set_title("Figure 15. System-level mean scores")
plt.tight_layout(); savefig("fig15_ablation_means.png"); plt.close()

print(f"Done. {len(list(FIG_DIR.glob('*.png')))} PNG files in {FIG_DIR}")
