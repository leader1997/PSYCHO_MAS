# Explainability-Augmented RAG — Phase 2

**Scientific RAG pipeline** for non-clinical depression/anxiety **screening communication** using the Healthy Minds Study (HMS) 2024–2025 cohort.

> **This system does NOT diagnose depression or anxiety.** It generates evidence-grounded, non-clinical screening narratives from model predictions and XAI explanations.

## Setup

```powershell
cd "explainability_augmented_rag"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Set OPENAI_API_KEY in .env
python scripts/seed_from_phase1.py
```

## Run order

```powershell
python retrieval/build_corpus.py
python xai/build_xai_cards.py
python generation/query_builder.py
python retrieval/retriever_eval.py --limit 100
python generation/evidence_synthesis.py --limit 3
python generation/narrative_generator.py --limit 3
python safety/safety_validator.py
python safety/faithfulness_checker.py
python evaluation/build_benchmark.py
python -m evaluation.ragas_evaluation --limit 10
python -m evaluation.ablation_evaluation
python evaluation/error_analysis.py
python evaluation/generate_paper_outputs.py
```

Smoke test: append `--limit 3` or `--limit 5` to evaluation scripts.

## Folder structure

- `data/` — corpus, benchmark, seeded phase-1 inputs
- `xai/` — standardized XAI cards
- `retrieval/` — corpus build, dense/BM25/hybrid/reranker
- `generation/` — query builder, evidence synthesis, narratives
- `safety/` — validator, faithfulness checker
- `evaluation/` — RAGAS, ablation, error analysis
- `notebooks/` — 01–12 reproducible notebooks
- `results/` — all intermediate CSV/JSONL outputs
- `paper_outputs/` — draft methods/results, tables, figures

## LLM backend

Default: **OpenAI** (`gpt-4o` generation/judge, `text-embedding-3-small` embeddings). Configure in `config.yaml`.

## Scientific constraints

1. No clinical diagnosis or treatment advice
2. XAI features are model contributors, not causes
3. Every factual claim must cite retrieved evidence
4. Elevated-risk outputs must recommend professional support
5. All student data remain de-identified

## Phase 1 relationship

Inputs are **copied once** from `../xai_rag_depression/` via `scripts/seed_from_phase1.py`. Phase 2 runs standalone after seeding.
