#!/usr/bin/env python3
"""Generate 05_paper_results_report.ipynb with all pipeline visualizations."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "05_paper_results_report.ipynb"

CELLS = []

def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "outputs": [], "source": text, "execution_count": None})

# --- Title ---
md("""# Pipeline Results Report for Paper Writing

**Project:** Explainability-Augmented RAG for Depression Risk Classification (HMS 2024–2025)

This notebook consolidates **all quantitative results, figures, and interpretive notes** from the end-to-end pipeline. It is intended for a paper writer or co-author who needs publication-ready tables and charts with plain-language explanations.

**Pipeline stages covered:**
0. **Global methodology architecture** (five layers + data flow)
1. Data & predictive models (9 ML + DL models)
2. Explainable AI (SHAP / Integrated Gradients)
3. Stability & weighted aggregation
4. RAG knowledge base (27 PDFs + project cards)
5. LLM generation, **recommendation examples**, & four-system ablation

**Execution mode:** FAST (`XAI_FAST=1`, seed=42) | **Artifacts:** `outputs/`""")

md("## How to use this notebook\n\nRun all cells top-to-bottom. Figures are saved to `outputs/figures/paper/` for direct inclusion in the manuscript. Each section includes:\n- **What was measured**\n- **How to interpret it**\n- **Key numbers for the paper**")

code("""import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.figsize"] = (10, 6)

OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")

# Load all artifacts
perf = pd.read_csv(OUTPUT_DIR / "model_performance.csv")
agg_imp = pd.read_csv(OUTPUT_DIR / "aggregated_importance.csv")
stab = pd.read_csv(OUTPUT_DIR / "stability_scores.csv")
faith = pd.read_csv(OUTPUT_DIR / "faithfulness.csv")
global_imp = pd.read_csv(OUTPUT_DIR / "global_importance_per_model.csv", index_col=0)
xai_methods = pd.read_csv(OUTPUT_DIR / "xai_method_per_model.csv", index_col=0)
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
print("Loaded all pipeline artifacts.")""")

# --- Section 0: Global methodology architecture ---
md("""---
## 0. Global methodology architecture (for Methods / Figure 1 in paper)

This section describes the **full end-to-end methodology** so a paper writer can draft the Methods and System Architecture subsections.

### Overview (five layers)

| Layer | Component | Input | Output |
|-------|-----------|-------|--------|
| **L1 — Data** | HMS 2024–2025 CSV, cleaning, imputation, scaling | 84,735 raw rows | 63,581 × 25 features, stratified splits |
| **L2 — Prediction** | 6 ML + 3 DL models, weighted training | Feature matrix + `dep_or_anx` | 9 trained models, test AUC ≈ 0.85 |
| **L3 — Explainability** | SHAP / IG / coefficients per model | Trained models + test batch | Global + local attributions, faithfulness scores |
| **L4 — Aggregation** | Stability testing, model weights, semantic groups | Per-model importances | JSON explanation cards, aggregated groups |
| **L5 — RAG + LLM** | Vector KB (PDFs + cards), retrieval, Ollama Llama 3 | Query + context bundles | Grounded narratives + ablation metrics |

### Four ablation systems (L5)

| System | Context fed to LLM |
|--------|-------------------|
| S0 baseline | Dataset description + prediction only |
| S1 RAG + local XAI | S0 + CatBoost SHAP (case) + 6 literature chunks |
| S2 RAG + aggregated XAI | S0 + cross-model semantic groups + 6 literature chunks |
| S3 Full | S0 + local + aggregated XAI + 6 literature chunks |

**Evaluation:** LLM-as-judge (faithfulness, relevancy, hallucination, context precision/recall).""")

code("""# Figure 20: Detailed global methodology architecture (layered diagram)
fig, ax = plt.subplots(figsize=(16, 14))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(x, y, w, h, title, lines, fc, ec):
    p = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                 facecolor=fc, edgecolor=ec, linewidth=2, alpha=0.85)
    ax.add_patch(p)
    ax.text(x + w/2, y + h - 0.025, title, ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(x + w/2, y + h/2 - 0.01, "\\n".join(lines), ha="center", va="center", fontsize=8.5)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.8, color="#444"))

# L1 Data
box(0.05, 0.88, 0.9, 0.09, "L1 — DATA LAYER",
    ["HMS 2024–2025 CSV → clean (Finished=1, ≥60% features) → 63,581 rows",
     "Median imputation + StandardScaler | Target: dep_or_anx | Weights: nrweight"],
    "#E8F4FD", "#4C72B0")
arrow(0.5, 0.88, 0.5, 0.84)

# L2 Models — split ML / DL
box(0.05, 0.72, 0.43, 0.11, "L2a — ML MODELS (6)",
    ["Logistic Regression, Random Forest, Extra Trees",
     "XGBoost, LightGBM, CatBoost | TreeSHAP / coefficients"],
    "#EAF2E8", "#55A868")
box(0.52, 0.72, 0.43, 0.11, "L2b — DL MODELS (3)",
    ["FT-Transformer, TabTransformer, DCNv2",
     "Integrated Gradients (Captum) | Weighted BCE loss"],
    "#FFF0E8", "#DD8452")
arrow(0.27, 0.72, 0.27, 0.68); arrow(0.73, 0.72, 0.73, 0.68)

# L3 XAI
box(0.05, 0.56, 0.9, 0.11, "L3 — EXPLAINABILITY (XAI)",
    ["Global: mean |SHAP| or |IG| per feature, normalized | Local: CatBoost SHAP top-5 per case",
     "Faithfulness: mask top-5 features → measure AUC drop on test set"],
    "#F5E8FF", "#9467BD")
arrow(0.5, 0.56, 0.5, 0.52)

# L4 Aggregation
box(0.05, 0.40, 0.9, 0.11, "L4 — STABILITY & AGGREGATION",
    ["Stability: Spearman + Kendall + Jaccard(top-k) across seeds/folds/perturbations",
     "Model weights → weighted importance → semantic groups → JSON explanation cards"],
    "#FFF8E1", "#BCAE6B")
arrow(0.5, 0.40, 0.5, 0.36)

# L5 RAG
box(0.05, 0.24, 0.9, 0.11, "L5 — RAG KNOWLEDGE BASE",
    ["27 PDFs → 2,278 chunks + project cards (2,321 docs total)",
     "Embeddings: all-MiniLM-L6-v2 (384-d) | Cosine similarity retrieval"],
    "#E8F8F5", "#17BECF")
arrow(0.5, 0.24, 0.5, 0.20)

# L6 LLM — four systems
box(0.02, 0.04, 0.22, 0.14, "S0: Baseline",
    ["Dataset +", "prediction only"], "#FFE8E8", "#C44E52")
box(0.27, 0.04, 0.22, 0.14, "S1: Local XAI",
    ["+ CatBoost SHAP", "+ literature"], "#FFE8E8", "#C44E52")
box(0.52, 0.04, 0.22, 0.14, "S2: Agg. XAI",
    ["+ semantic groups", "+ literature"], "#FFE8E8", "#C44E52")
box(0.77, 0.04, 0.21, 0.14, "S3: Full",
    ["local + agg.", "+ literature"], "#FFE8E8", "#C44E52")

for xc in [0.13, 0.38, 0.63, 0.875]:
    arrow(xc, 0.20, xc, 0.18)

ax.text(0.5, 0.015, "LLM: Ollama Llama 3 | Strict prompt | Non-clinical screening narratives",
        ha="center", fontsize=9, style="italic", color="#555")
ax.set_title("Figure 20. Global methodology architecture (five layers + four ablation systems)",
             fontsize=14, fontweight="bold", pad=16)
plt.tight_layout(); savefig("fig20_methodology_architecture.png"); plt.show()""")

code("""# Figure 21: Data-flow diagram (inputs → outputs per stage)
fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

nodes = [
    (1, 8.5, "HMS CSV\\n84,735 rows", "#4C72B0"),
    (1, 6.5, "Cleaned data\\n63,581 × 25", "#4C72B0"),
    (3, 7.5, "9 trained\\nmodels", "#55A868"),
    (3, 5.5, "Global + local\\nXAI", "#9467BD"),
    (5, 7.5, "Stability +\\naggregation", "#BCAE6B"),
    (5, 5.5, "JSON cards\\n(explanation)", "#BCAE6B"),
    (7, 6.5, "RAG KB\\n2,321 docs", "#17BECF"),
    (9, 7, "S0–S3\\nLLM answers", "#C44E52"),
    (9, 5, "LLM-as-judge\\nmetrics", "#C44E52"),
]
for x, y, label, c in nodes:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.7, y-0.5), 1.4, 1.0, boxstyle="round,pad=0.02",
                  facecolor=c, alpha=0.25, edgecolor=c, lw=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")

flows = [(1,8.5,1,6.5), (1,6.5,3,7.5), (3,7.5,3,5.5), (3,5.5,5,7.5), (5,7.5,5,5.5),
         (5,5.5,7,6.5), (3,5.5,7,6.5), (7,6.5,9,7), (9,7,9,5), (1,6.5,7,6.5)]
for x1,y1,x2,y2 in flows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5,
                                connectionstyle="arc3,rad=0.1"))
ax.set_title("Figure 21. End-to-end data flow (inputs and artifacts per stage)", fontsize=13, pad=12)
plt.tight_layout(); savefig("fig21_data_flow.png"); plt.show()""")

md("""### Mermaid diagram (for paper / presentation tools)

```mermaid
flowchart TB
    subgraph L1["L1 Data"]
        A[HMS 2024-2025 CSV] --> B[Clean & preprocess]
        B --> C[Train/Val/Test split]
    end
    subgraph L2["L2 Prediction"]
        C --> D[6 ML models]
        C --> E[3 DL models]
    end
    subgraph L3["L3 XAI"]
        D --> F[TreeSHAP / coefficients]
        E --> G[Integrated Gradients]
        F --> H[Global + local explanations]
        G --> H
    end
    subgraph L4["L4 Aggregation"]
        H --> I[Stability testing]
        I --> J[Weighted aggregation]
        J --> K[Semantic groups + JSON cards]
    end
    subgraph L5["L5 RAG + LLM"]
        K --> L[Vector KB: 27 PDFs + cards]
        L --> M{4 ablation systems}
        M --> N[S0 baseline]
        M --> O[S1 local XAI]
        M --> P[S2 aggregated XAI]
        M --> Q[S3 full]
        N & O & P & Q --> R[LLM-as-judge evaluation]
    end
```""")

# --- Section 1: Study design ---
md("""---
## 1. Study design & data (for Methods section)

| Item | Value |
|------|-------|
| Dataset | Healthy Minds Study (HMS) 2024–2025 |
| Target | `dep_or_anx` (elevated depression and/or anxiety) |
| Cleaned sample | 63,581 respondents (44.9% positive) |
| Split | Train 50,864 / Val 6,358 / Test 6,359 (80/10/10, stratified) |
| Features | 25 survey variables (demographics, financial, sleep, social, academic, MH, substance, discrimination, institution) |
| Weights | HMS non-response weight `nrweight` in training |
| Excluded | `gender_nonbin`, `gender_trans` (aligned with DL notebook) |

**Preprocessing:** median imputation + StandardScaler (fit on train only). Same scaled feature space for all 9 models.""")

code("""# Table 1-style sample summary for the paper
sample_summary = pd.DataFrame([
    {"Statistic": "Raw respondents", "Value": "84,735"},
    {"Statistic": "After cleaning", "Value": "63,581"},
    {"Statistic": "Positive class rate", "Value": "44.9%"},
    {"Statistic": "Test set size", "Value": "6,359"},
    {"Statistic": "Number of features", "Value": "25"},
    {"Statistic": "Random seed", "Value": "42"},
])
display(sample_summary)""")

# --- Section 2: Model performance ---
md("""---
## 2. Predictive performance (Notebook 1)

**What we measured:** Discrimination (AUC, PR-AUC), classification metrics (accuracy, F1, recall), and calibration (Brier score) on the held-out test set.

**Paper takeaway:** All nine models achieve **AUC 0.847–0.853** — strong and tightly clustered performance. XGBoost is best by AUC (0.853); FT-Transformer is best among DL models (0.852). No single architecture dominates; this motivates **multi-model XAI aggregation** rather than relying on one black-box model.""")

code("""# Figure 1: AUC by model (color by family)
perf_sorted = perf.sort_values("auc", ascending=True)
colors = perf_sorted["family"].map({"ML": "#4C72B0", "DL": "#DD8452"})

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(perf_sorted["model"], perf_sorted["auc"], color=colors)
ax.set_xlim(0.84, 0.856)
ax.set_xlabel("Test-set AUC")
ax.set_title("Figure 1. Predictive performance across 9 models (test set, n=6,359)")
ax.axvline(0.85, color="gray", ls="--", alpha=0.5, label="AUC = 0.85")
for bar, val in zip(bars, perf_sorted["auc"]):
    ax.text(val + 0.0003, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
ax.legend(handles=[mpatches.Patch(color="#4C72B0", label="ML"), mpatches.Patch(color="#DD8452", label="DL")])
plt.tight_layout(); savefig("fig01_auc_by_model.png"); plt.show()

# Figure 2: Multi-metric comparison (normalized for radar-style bar groups)
metrics = ["auc", "macro_f1", "recall", "pr_auc"]
metric_labels = ["AUC", "Macro F1", "Recall", "PR-AUC"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, m, lbl in zip(axes.flat, metrics, metric_labels):
    d = perf.sort_values(m, ascending=False)
    ax.bar(range(len(d)), d[m], color=d["family"].map({"ML": "#4C72B0", "DL": "#DD8452"}))
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["model"], rotation=45, ha="right", fontsize=8)
    ax.set_title(lbl)
    ax.set_ylim(d[m].min() - 0.02, d[m].max() + 0.02)
fig.suptitle("Figure 2. Test-set metrics by model", y=1.02)
plt.tight_layout(); savefig("fig02_multi_metrics.png"); plt.show()

# Table for paper: full performance
display(perf.sort_values("auc", ascending=False).round(4))""")

code("""# Figure 3: ML vs DL boxplot (AUC distribution across families)
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=perf, x="family", y="auc", palette={"ML": "#4C72B0", "DL": "#DD8452"}, ax=ax)
sns.stripplot(data=perf, x="family", y="auc", color="black", size=8, ax=ax)
ax.set_title("Figure 3. AUC distribution: ML (n=6) vs DL (n=3) models")
ax.set_ylabel("AUC"); ax.set_xlabel("")
plt.tight_layout(); savefig("fig03_auc_ml_vs_dl.png"); plt.show()

print(f"ML AUC range: {perf[perf.family=='ML'].auc.min():.3f} – {perf[perf.family=='ML'].auc.max():.3f}")
print(f"DL AUC range: {perf[perf.family=='DL'].auc.min():.3f} – {perf[perf.family=='DL'].auc.max():.3f}")""")

# --- Section 3: XAI ---
md("""---
## 3. Explainable AI — global feature importance (Notebook 1)

**Methods per model:**
- ML (trees): TreeSHAP (mean |SHAP|)
- Logistic regression: standardized coefficients + LinearSHAP
- DL: Integrated Gradients (Captum)

**Normalization:** $\\text{norm\\_imp}_{m,j} = |\\text{imp}_{m,j}| / \\sum_k |\\text{imp}_{m,k}|$

**Paper takeaway:** **aca_impa** (academic impairment), **lonely**, **flourish**, and **fincur** appear in the top-5 for every model. High cross-model consensus supports using aggregated explanations for policy-facing narratives.

**Caveat for paper:** `aca_impa` measures academic impairment *due to* mental health — partially tautological with the target. Frame as a strong marker, not an independent upstream cause.""")

code("""# Figure 4: Heatmap — normalized importance across 9 models (top 12 features)
top_feats = agg_imp.nlargest(12, "final_importance")["feature"].tolist()
hm = global_imp.loc[top_feats]
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(hm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Normalized importance"})
ax.set_title("Figure 4. Top-12 features: normalized importance heatmap (9 models)")
ax.set_ylabel("Feature"); ax.set_xlabel("Model")
plt.tight_layout(); savefig("fig04_importance_heatmap.png"); plt.show()""")

code("""# Figure 5: Aggregated weighted importance + agreement
top = agg_imp.nlargest(12, "final_importance").sort_values("final_importance")
agree_colors = top["agreement_level"].map({"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"})

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top["feature"], top["final_importance"], color=agree_colors)
ax.set_xlabel("Weighted aggregated importance (across 9 models)")
ax.set_title("Figure 5. Aggregated feature importance with cross-model agreement color")
ax.legend(handles=[
    mpatches.Patch(color="#2ca02c", label="High agreement"),
    mpatches.Patch(color="#ff7f0e", label="Medium agreement"),
    mpatches.Patch(color="#d62728", label="Low agreement"),
], loc="lower right")
plt.tight_layout(); savefig("fig05_aggregated_importance.png"); plt.show()

display(agg_imp.nlargest(10, "final_importance").round(4))""")

code("""# Figure 6: Semantic feature groups (from aggregation card)
groups = pd.DataFrame(agg_card["top_feature_groups"])
agree_map = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"}
colors = groups["agreement"].map(agree_map)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(groups["group"], groups["importance"], color=colors)
ax.set_xlabel("Aggregated group importance")
ax.set_title("Figure 6. Semantic feature groups (weighted cross-model aggregation)")
ax.invert_yaxis()
plt.tight_layout(); savefig("fig06_semantic_groups.png"); plt.show()

print("Final insight:", agg_card["final_insight"])""")

code("""# Figure 7: Cross-model rank agreement — how many models put each feature in top-10
top10_counts = (global_imp.rank(ascending=False) <= 10).sum(axis=1).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top10_counts.index[::-1], top10_counts.values[::-1], color="#4C72B0")
ax.set_xlabel("Number of models (out of 9) with feature in top-10")
ax.set_title("Figure 7. Cross-model top-10 agreement count")
plt.tight_layout(); savefig("fig07_top10_agreement.png"); plt.show()""")

# --- Section 4: Stability ---
md("""---
## 4. Explanation stability & model weights (Notebook 2)

**Stability score:** Mean of Spearman ρ, Kendall τ, and Jaccard(top-k) across random seeds / CV folds / perturbations (ML: 7 runs; DL: 4 perturbation runs).

**Model weight:** Combines stability, calibration (1 − Brier), and macro-F1 to weight each model in the aggregation.

**Paper takeaway:** Tree-based ML models show **stability ≥ 0.93**; DL models are lower (0.72–0.90) but still contribute. XGBoost receives the highest aggregation weight (0.631).""")

code("""# Figure 8: Stability vs model weight
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

family_color = dict(zip(perf["model"], perf["family"].map({"ML": "#4C72B0", "DL": "#DD8452"})))
s = stab.sort_values("stability_score", ascending=True)
axes[0].barh(s["model"], s["stability_score"], color=s["model"].map(family_color))
axes[0].set_xlabel("Stability score"); axes[0].set_title("Explanation stability")
axes[0].set_xlim(0.65, 1.0)

w = stab.sort_values("model_weight", ascending=True)
axes[1].barh(w["model"], w["model_weight"], color=w["model"].map(family_color))
axes[1].set_xlabel("Aggregation weight"); axes[1].set_title("Model weights for XAI aggregation")
plt.suptitle("Figure 8. Stability scores and aggregation weights", y=1.02)
plt.tight_layout(); savefig("fig08_stability_weights.png"); plt.show()

display(stab.round(4))""")

code("""# Figure 9: Stability vs calibration scatter
merged = stab.merge(perf[["model", "family", "macro_f1"]], on="model")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=merged, x="stability_score", y="calibration_score", hue="family",
                style="family", s=150, palette={"ML": "#4C72B0", "DL": "#DD8452"}, ax=ax)
for _, row in merged.iterrows():
    ax.annotate(row["model"], (row["stability_score"], row["calibration_score"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_title("Figure 9. Stability vs calibration by model")
plt.tight_layout(); savefig("fig09_stability_calibration.png"); plt.show()""")

# --- Section 5: Faithfulness ---
md("""---
## 5. Explanation faithfulness — deletion test (Notebook 1)

**Method:** Mask the top-5 globally important features (set to median), re-score test set, measure AUC drop.

$$\\text{AUC drop} = \\text{AUC}_{\\text{base}} - \\text{AUC}_{\\text{masked}}$$

**Paper takeaway:** Large AUC drops (0.13–0.21) confirm that highlighted features are **functionally important** to predictions, not arbitrary attributions. DCNv2 shows the largest drop (0.206); FT-Transformer the smallest among models (0.126).""")

code("""# Figure 10: Faithfulness (AUC drop when masking top-5 features)
f = faith.sort_values("auc_drop", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(f["model"], f["auc_drop"], color="#C44E52")
ax.set_xlabel("AUC drop (base AUC − masked top-5 AUC)")
ax.set_title("Figure 10. Explanation faithfulness via feature-masking deletion test")
for y, (_, row) in enumerate(f.iterrows()):
    ax.text(row["auc_drop"] + 0.003, y, f"{row['auc_drop']:.3f}", va="center", fontsize=9)
plt.tight_layout(); savefig("fig10_faithfulness_auc_drop.png"); plt.show()

display(faith.round(4))""")

# --- Section 6: Local explanations ---
md("""---
## 6. Local (case-level) explanations (Notebook 1 → Notebook 2 cards)

Three de-identified test cases illustrate how **CatBoost TreeSHAP** explains individual predictions:
- **high_risk_case** — probability 0.979
- **low_risk_case** — probability 0.05
- **borderline_case** — probability 0.50

These feed the **S1 (local XAI)** and **S3 (full)** RAG systems.""")

code("""# Figure 11: Local SHAP factors for each case (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, case in zip(axes, local_cases):
    factors = pd.DataFrame(case["top_contributing_factors"])
    colors = ["#C44E52" if d == "increases risk" else "#4C72B0" for d in factors["direction"]]
    ax.barh(factors["feature"], factors["shap_value"].abs(), color=colors)
    ax.set_title(f"{case['case_id']}\\np={case['predicted_probability']}")
    ax.set_xlabel("|SHAP value|")
fig.suptitle("Figure 11. Local XAI — top-5 CatBoost SHAP factors per case", y=1.05)
plt.tight_layout(); savefig("fig11_local_shap_cases.png"); plt.show()

for case in local_cases:
    print(f"\\n=== {case['case_id']} ===")
    print(f"Prediction: {case['predicted_class']} (p={case['predicted_probability']})")
    display(pd.DataFrame(case["top_contributing_factors"]))""")

# --- Section 7: RAG KB ---
md("""---
## 7. RAG knowledge base (Notebook 3)

**Contents:** 2,321 embedded documents — 2,278 literature chunks from **27 PDFs**, plus project cards (dataset overview, XAI cards, ethics, feature definitions).

**Embeddings:** `all-MiniLM-L6-v2`, dimension 384, cosine similarity retrieval.""")

code("""# Figure 12: KB composition pie chart
kb_df = pd.Series(kb_stats).drop(["total_documents", "embedding_dim"])
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(kb_df.values, labels=kb_df.index, autopct="%1.1f%%", startangle=140)
ax.set_title(f"Figure 12. Knowledge base composition (n={kb_stats['total_documents']} documents)")
plt.tight_layout(); savefig("fig12_kb_composition.png"); plt.show()

print(f"Embedding dimension: {kb_stats['embedding_dim']}")""")

code("""# Figure 13: Literature PDFs by chunk count (top 15)
top_pdfs = pdf_inv.nlargest(15, "chunks").sort_values("chunks")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_pdfs["file"].str.replace(".pdf", "", regex=False), top_pdfs["chunks"], color="#55A868")
ax.set_xlabel("Number of text chunks")
ax.set_title("Figure 13. Top-15 literature PDFs by chunk count in RAG KB")
plt.tight_layout(); savefig("fig13_pdf_chunks.png"); plt.show()

print(f"Total PDFs: {len(pdf_inv)} | Total literature chunks: {pdf_inv.chunks.sum()}")""")

# --- Section 8: LLM evaluation ---
md("""---
## 8. LLM generation & four-system ablation (Notebook 4)

**Systems compared:**

| ID | System | Local XAI | Aggregated XAI | Literature |
|----|--------|-----------|----------------|------------|
| S0 | baseline | ✗ | ✗ | ✗ |
| S1 | rag_xai | ✓ | ✗ | ✓ |
| S2 | rag_agg_xai | ✗ | ✓ | ✓ |
| S3 | full | ✓ | ✓ | ✓ |

**LLM:** Ollama, Llama 3 (`llama3:latest`), T=0.2 (generation), T=0.0 (judge)

**Metrics (LLM-as-judge, scores ∈ [0,1]):**
- **Faithfulness** — answer grounded in context
- **Answer relevancy** — answer addresses the question
- **Hallucination rate** — fraction unsupported (lower is better)
- **Context precision / recall** — retrieval quality (S1–S3 only)""")

code("""# Figure 14: Ablation heatmap (all metrics, all cases)
plot_df = rag_eval.copy()
plot_df["system"] = plot_df["system"].map(SYSTEM_LABELS)
metric_cols = ["faithfulness", "answer_relevancy", "hallucination_rate", "context_precision", "context_recall"]
heat = plot_df.set_index(["case", "system"])[metric_cols]
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
ax.set_title("Figure 14. Four-system ablation — all metrics by case")
plt.tight_layout(); savefig("fig14_ablation_heatmap.png"); plt.show()""")

code("""# Figure 15: System-level mean comparison (grouped bar)
summary = rag_eval.groupby("system")[metric_cols].mean().reindex(SYSTEM_ORDER)
summary.index = [SYSTEM_LABELS[s] for s in summary.index]

fig, ax = plt.subplots(figsize=(12, 6))
summary[["faithfulness", "answer_relevancy", "hallucination_rate"]].plot.bar(ax=ax, rot=0, width=0.8)
ax.set_ylabel("LLM-as-judge score (0–1)")
ax.set_title("Figure 15. System-level mean scores (3 cases)")
ax.legend(loc="upper right")
ax.set_ylim(0, 1)
plt.tight_layout(); savefig("fig15_ablation_means.png"); plt.show()

display(summary.round(3))""")

code("""# Figure 16: Per-case grouped comparison (faithfulness & relevancy)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in zip(axes, ["faithfulness", "answer_relevancy"],
                              ["Faithfulness", "Answer relevancy"]):
    pivot = rag_eval.pivot(index="case", columns="system", values=metric)[SYSTEM_ORDER]
    pivot.columns = [SYSTEM_LABELS[s] for s in pivot.columns]
    pivot.plot.bar(ax=ax, rot=0)
    ax.set_title(title); ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="lower right")
fig.suptitle("Figure 16. Ablation metrics by case", y=1.02)
plt.tight_layout(); savefig("fig16_ablation_by_case.png"); plt.show()""")

code("""# Figure 17: Radar chart — system-level profile (3 core metrics)
from math import pi

categories = ["Faithfulness", "Answer relevancy", "1 − Hallucination"]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for sid in SYSTEM_ORDER:
    row = summary.loc[SYSTEM_LABELS[sid]]
    values = [row["faithfulness"], row["answer_relevancy"], 1 - row["hallucination_rate"]]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=SYSTEM_LABELS[sid])
    ax.fill(angles, values, alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title("Figure 17. System profile radar (higher = better)", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.tight_layout(); savefig("fig17_ablation_radar.png"); plt.show()""")

code("""# Figure 18: Context precision vs recall (XAI systems only)
xai_eval = rag_eval[rag_eval.system != "baseline"].copy()
xai_eval["system"] = xai_eval["system"].map(SYSTEM_LABELS)
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=xai_eval, x="context_precision", y="context_recall",
                hue="system", style="case", s=150, ax=ax)
ax.set_xlim(0, 0.5); ax.set_ylim(0, 1)
ax.set_xlabel("Context precision"); ax.set_ylabel("Context recall")
ax.set_title("Figure 18. RAG retrieval quality (context precision vs recall)")
plt.tight_layout(); savefig("fig18_context_prec_recall.png"); plt.show()""")

# --- Section 8b: Recommendation examples ---
md("""---
## 8b. LLM recommendation examples (for Results / Qualitative section)

The LLM generates **non-clinical, supportive recommendations** grounded in retrieved literature and XAI evidence. Each answer follows a fixed structure:

`Prediction → Main contributing factors → Model agreement → Uncertainty → Interpretation → **Recommendation**`

Below we show **extracted Recommendation sections** for the three de-identified cases, comparing all four ablation systems on the **high-risk case** and showing full-system examples for low-risk and borderline cases.

**For the paper:** Emphasize that S1/S3 cite **case-specific SHAP factors** (e.g., aca_impa, lonely), while S2 cites **semantic groups** (Academic impact, Financial stress). S0 gives generic advice without factor-level grounding.""")

code("""import re

def extract_section(text, section_name):
    \"\"\"Extract a named section from structured LLM output.\"\"\"
    if not text or not isinstance(text, str):
        return "(not available — run `python run_nb4.py`)"
    pattern = rf"(?i){re.escape(section_name)}\\s*:?\\s*\\n(.*?)(?=\\n\\s*[A-Z][A-Za-z /&]+:|\\Z)"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else "(section not found in output)"

llm_path = OUTPUT_DIR / "llm_example_answers.json"
if llm_path.exists():
    llm_answers = json.load(open(llm_path, encoding="utf-8"))
    print(f"Loaded {sum(len(v) for v in llm_answers.values())} generated answers from {llm_path.name}")
else:
    llm_answers = {}
    print("WARNING: llm_example_answers.json not found. Run: python run_nb4.py")

# Table: Recommendation comparison — high_risk_case across 4 systems
if llm_answers and "high_risk_case" in llm_answers:
    rec_rows = []
    for sid in SYSTEM_ORDER:
        ans = llm_answers["high_risk_case"].get(sid, "")
        rec_rows.append({
            "System": SYSTEM_LABELS[sid],
            "Recommendation (extracted)": extract_section(ans, "Recommendation"),
            "Faithfulness": rag_eval[(rag_eval.case == "high_risk_case") & (rag_eval.system == sid)]["faithfulness"].values[0],
        })
    rec_high = pd.DataFrame(rec_rows)
    print("\\n=== Table 7. Recommendation comparison — high_risk_case (p=0.979) ===")
    display(rec_high)

    # Side-by-side print for paper writer
    for _, row in rec_high.iterrows():
        print("\\n" + "=" * 70)
        print(row["System"], f"| faithfulness={row['Faithfulness']}")
        print("=" * 70)
        print(row["Recommendation (extracted)"])
else:
    print("No high_risk_case answers available.")""")

code("""# Figure 22: Recommendation length & specificity proxy (word count by system)
if llm_answers:
    rows = []
    for case_id, systems in llm_answers.items():
        for sid, ans in systems.items():
            rec = extract_section(ans, "Recommendation")
            rows.append({
                "case": case_id,
                "system": SYSTEM_LABELS[sid],
                "word_count": len(rec.split()),
                "mentions_literature": int(any(w in rec.lower() for w in ["literature", "campus", "counsel", "support", "resource"])),
            })
    rec_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=rec_df, x="system", y="word_count", hue="case", ax=ax)
    ax.set_title("Figure 22. Recommendation length (words) by system and case")
    ax.set_xlabel(""); ax.set_ylabel("Word count in Recommendation section")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); savefig("fig22_recommendation_length.png"); plt.show()
    display(rec_df.pivot(index="case", columns="system", values="word_count"))""")

code("""# Full narrative examples — one per case (S3 full system)
if llm_answers:
    for case_id in ["high_risk_case", "low_risk_case", "borderline_case"]:
        if case_id in llm_answers and "full" in llm_answers[case_id]:
            print("\\n" + "#" * 80)
            print(f"FULL SYSTEM (S3) EXAMPLE — {case_id}")
            print("#" * 80)
            print(llm_answers[case_id]["full"])
            print()
else:
    print("Run python run_nb4.py to populate full narrative examples.")""")

code("""# Baseline vs Full — recommendation contrast (high-risk)
if llm_answers and "high_risk_case" in llm_answers:
    base_rec = extract_section(llm_answers["high_risk_case"].get("baseline", ""), "Recommendation")
    full_rec = extract_section(llm_answers["high_risk_case"].get("full", ""), "Recommendation")
    contrast = pd.DataFrame([
        {"System": "S0 Baseline (no XAI)", "Recommendation": base_rec},
        {"System": "S3 Full (local + aggregated XAI + literature)", "Recommendation": full_rec},
    ])
    print("\\n=== Qualitative contrast for Discussion ===")
    display(contrast.T)""")

# --- Section 9: Pipeline diagram ---
md("""---
## 9. Pipeline architecture (for Figure / Methods diagram)

```
HMS 2024-2025 CSV (63,581 rows, 25 features)
        │
        ▼
┌───────────────────────────────────────┐
│  Stage 1: 9 models + per-model XAI    │
│  ML: TreeSHAP / LR coefficients       │
│  DL: Integrated Gradients             │
│  → global + local explanations        │
│  → faithfulness (deletion test)       │
└───────────────┬───────────────────────┘
                ▼
┌───────────────────────────────────────┐
│  Stage 2: Stability + aggregation     │
│  → stability scores, model weights    │
│  → semantic groups, JSON cards        │
└───────────────┬───────────────────────┘
                ▼
┌───────────────────────────────────────┐
│  Stage 3: RAG knowledge base          │
│  → 27 PDFs (2,278 chunks) + cards     │
│  → MiniLM-L6-v2 embeddings (384-d)    │
└───────────────┬───────────────────────┘
                ▼
┌───────────────────────────────────────┐
│  Stage 4: LLM generation + ablation   │
│  → 4 systems (baseline → full)        │
│  → Llama 3 + LLM-as-judge metrics     │
└───────────────────────────────────────┘
```""")

code("""# Figure 19: End-to-end pipeline flow (schematic)
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")
stages = [
    ("Stage 1\\n9 Models + XAI", "AUC 0.847–0.853\\nSHAP / IG attributions"),
    ("Stage 2\\nStability & Aggregation", "Stability ≥ 0.93 (ML)\\nWeighted semantic groups"),
    ("Stage 3\\nRAG Knowledge Base", "2,321 documents\\n27 PDFs vectorized"),
    ("Stage 4\\nLLM Ablation", "Relevancy 0.90 (XAI)\\nvs 0.77 (baseline)"),
]
x = [0.12, 0.38, 0.64, 0.90]
colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
for i, (title, detail) in enumerate(stages):
    rect = mpatches.FancyBboxPatch((x[i]-0.1, 0.35), 0.2, 0.35, boxstyle="round,pad=0.02",
                                    facecolor=colors[i], alpha=0.3, edgecolor=colors[i], linewidth=2)
    ax.add_patch(rect)
    ax.text(x[i], 0.58, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x[i], 0.42, detail, ha="center", va="center", fontsize=9)
    if i < len(stages) - 1:
        ax.annotate("", xy=(x[i+1]-0.1, 0.52), xytext=(x[i]+0.1, 0.52),
                    arrowprops=dict(arrowstyle="->", lw=2, color="gray"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Figure 19. End-to-end pipeline schematic", fontsize=14, pad=20)
plt.tight_layout(); savefig("fig19_pipeline_schematic.png"); plt.show()""")

# --- Section 10: Paper tables ---
md("""---
## 10. Ready-to-use summary tables for the manuscript

Copy the displayed DataFrames below into the paper. Suggested table numbering:
- **Table 1:** Sample & study design (Section 1)
- **Table 2:** Model performance (Section 2)
- **Table 3:** Top aggregated features (Section 3)
- **Table 4:** Stability & weights (Section 4)
- **Table 5:** Faithfulness AUC drop (Section 5)
- **Table 6:** Four-system RAG ablation (Section 8)
- **Table 7:** Recommendation comparison — high-risk case (Section 8b)""")

code("""# Table 6: Ablation summary for paper
paper_table6 = summary.copy()
paper_table6.columns = ["Faithfulness", "Answer relevancy", "Hallucination rate",
                         "Context precision", "Context recall"]
display(paper_table6.round(3))

# Key bullet points for Discussion section
findings = pd.DataFrame([
    {"Finding": "Predictive performance", "Result": "All 9 models AUC 0.847–0.853; XGBoost best (0.853)"},
    {"Finding": "Top drivers", "Result": "aca_impa, lonely, flourish, fincur — 9/9 model consensus"},
    {"Finding": "Explanation stability", "Result": "Tree ML ≥ 0.93; DL 0.72–0.90"},
    {"Finding": "Faithfulness", "Result": "AUC drop 0.13–0.21 when masking top-5 features"},
    {"Finding": "RAG ablation", "Result": "XAI systems relevancy 0.90 vs baseline 0.77"},
    {"Finding": "Best hallucination", "Result": "Aggregated XAI & full system: 0.40 vs baseline 0.50"},
    {"Finding": "Limitation", "Result": "LLM-as-judge on 3 cases; not clinical validation"},
], )
print("\\n=== KEY FINDINGS FOR DISCUSSION ===")
display(findings)""")

md("""---
## 11. Ethical limitations (must appear in paper)

1. **Not a diagnostic tool** — screening signals only, not clinical diagnosis.
2. **Does not replace professionals** — must not substitute licensed mental-health providers.
3. **Observational associations** — XAI shows statistical drivers, not causal mechanisms.
4. **aca_impa circularity** — academic impairment co-measures mental health impact.
5. **LLM-as-judge** — automated evaluation proxy; human expert review recommended.
6. **Small LLM eval sample** — 3 de-identified cases; results are illustrative.

---
*Generated from pipeline artifacts in `outputs/`. Re-run upstream notebooks or `run_nb1.py`–`run_nb4.py` to refresh.*""")

# Write notebook
notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH} ({len(CELLS)} cells)")
