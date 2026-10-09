# Local Agentic XAI-RAG (Ollama llama3 only)

Controlled local pipeline that produces **advisory screening reports for human review** from HMS screening predictions, XAI cards, and retrieved literature.

> **Not a diagnostic tool.** All LLM and embedding calls use **Ollama `llama3` only** — no cloud APIs, no nomic-embed-text, no separate judge model.

## Prerequisites

- Phase 2 corpus built: [`../explainability_augmented_rag/data/processed_chunks.csv`](../explainability_augmented_rag/data/processed_chunks.csv)
- Ollama running with `llama3`: `ollama pull llama3`

## Setup

```powershell
cd "local_agentic_xai_rag"
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## Streamlit UI

Interactive dashboard for the full system (pipeline, reports, retrieval, evaluation, figures):

```powershell
pip install -e .
streamlit run streamlit_app.py
```

Pages: **Overview** · **Run pipeline** · **Case reports** · **Retrieval explorer** · **Evaluation dashboard** · **Figures & paper**

## Run (3 cases)

```powershell
# Fast smoke (BM25 only, no llama3 embed index build):
python -m local_agentic_xai_rag.main --all-cases --bm25-only

# Full hybrid (BM25 + llama3 embeddings — builds FAISS on first run):
python -m local_agentic_xai_rag.main --all-cases
```

## Pipeline

```
Case + XAI → Query Planning Agent (llama3) → hybrid retrieval
  → evidence agent (llama3) → report agent (llama3)
  → regex safety + claim verification (llama3) → final_narratives.jsonl
```

Use `--templates-only` to skip the query planning agent and use fixed template queries.
Use `--no-query-planning` to disable the agent via CLI.

## Outputs

- `results/evidence_summaries.jsonl`
- `results/generated_narratives.jsonl`
- `results/final_narratives.jsonl`
- `results/validation_reports.csv`
- `results/claim_level_faithfulness.csv`

## Configuration

[`config.yaml`](config.yaml): `ollama.model: llama3` for chat **and** embeddings.

Set `retrieval.bm25_only: true` to skip dense index during development.

## Evaluation

```powershell
jupyter notebook notebooks/01_paper_results_report.ipynb
python -m local_agentic_xai_rag.evaluation.retrieval_eval --limit 100
python -m local_agentic_xai_rag.evaluation.retrieval_eval --limit 100 --bm25-only-hybrid  # fast hybrid slot
python -m local_agentic_xai_rag.evaluation.system_comparison --bm25-only
python -m local_agentic_xai_rag.visualization.build_all_figures
python -m local_agentic_xai_rag.evaluation.expert_review_export
python -m local_agentic_xai_rag.evaluation.acceptance
```

## Baseline comparison (Phase 2)

| Metric | Phase 2 baseline | This project (see `results/`) |
|--------|------------------|-------------------------------|
| BM25 P@5 (gold 100Q) | 0.128 | `retrieval_eval_summary.csv` |
| Hybrid P@5 (gold 100Q) | 0.088 | `retrieval_eval_summary.csv` |
| Claim fully supported (3 cases) | 54.5% | `claim_level_faithfulness.csv` |
| Safety pass (3 cases) | 2/3 | `validation_reports.csv` |
| Keyword recall@5 (100Q advisory) | 0.157 | optional benchmark |
| Regex safety (100Q) | 61% | optional benchmark |

## Acceptance gates (G1–G6)

| Gate | Criterion |
|------|-----------|
| G1 | 3 cases → valid `final_narratives.jsonl` with all schema fields |
| G2 | Disclaimer in ≥ 90% of final reports |
| G3 | No unsupported/contradicted claims after citation check |
| G4 | Retrieval eval on 100 gold-chunk questions completes |
| G5 | All local; Ollama llama3 only |
| G6 | Results under `results/` |

Run `python -m local_agentic_xai_rag.evaluation.acceptance` → `results/acceptance_gates.json`.
