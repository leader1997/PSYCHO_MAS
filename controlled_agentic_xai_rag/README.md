# Controlled Agentic XAI-RAG

> **Not a diagnostic tool.** Generates evidence-grounded, non-clinical screening narratives for human review.

Greenfield implementation of the controlled local Agentic XAI-RAG architecture using **Ollama llama3 only** and the **knowledge_db** corpus.

## Prerequisites

- Python 3.10+
- Ollama with `llama3`: `ollama pull llama3`
- Phase 2 inputs (seeded automatically)

## Setup

```powershell
cd controlled_agentic_xai_rag
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

## Build corpus and inputs

```powershell
python scripts/seed_inputs.py
python scripts/download_pdfs.py
python scripts/build_corpus.py
# If PDFs unavailable:
python scripts/build_corpus.py --fallback-phase2
```

## Run pipeline

```powershell
# BM25-only smoke (no FAISS embed build):
python -m controlled_agentic_xai_rag.main --all-cases --bm25-only --skip-ollama-check

# Full hybrid retrieval:
python -m controlled_agentic_xai_rag.main --all-cases
```

## Ollama call policy

All LLM/embed calls are **sequential** with `inter_call_delay_seconds: 1` after each completed response. See `ARCHITECTURE.md`.

## Evaluation

**Paper-ready notebook** (tables + figures):

```powershell
jupyter notebook notebooks/08_scientific_evaluation.ipynb
```

CLI helpers:

```powershell
python -m controlled_agentic_xai_rag.evaluation.eval_retrieval --limit 100
python -m controlled_agentic_xai_rag.evaluation.expert_review_export
```

## Streamlit UI

```powershell
streamlit run streamlit_app.py
```

## Outputs

All artifacts under `results/`: formatted cases, query plans, retrieval audit, narratives, validation, `final_narratives.jsonl`.
