# Results — Local Agentic XAI-RAG

## Acceptance gates (G1–G6)

All gates **passed** (`results/acceptance_gates.json`):

| Gate | Result |
|------|--------|
| G1 | 3/3 cases with valid schema in `final_narratives.jsonl` |
| G2 | 100% disclaimer rate (vs 0% single-shot baseline) |
| G3 | 0 unsupported/contradicted claims after citation gate |
| G4 | 100-question gold-chunk retrieval eval completed |
| G5 | Ollama llama3 only (chat + embeddings) |
| G6 | All artifacts under `results/` |

## Retrieval (100 gold-chunk questions)

| Retriever | P@5 | R@5 | R@10 | MRR | nDCG@10 |
|-----------|-----|-----|------|-----|---------|
| BM25 | 0.128 | 0.130 | 0.179 | 0.300 | 0.174 |
| Hybrid (llama3) | see `retrieval_eval_summary.csv` | | | | |

BM25 matches Phase 2 baseline P@5 = 0.128. Full hybrid dense index builds on first run (`data/faiss_llama3/`); use `--bm25-only-hybrid` for fast eval slot during development.

## Advisory reports (3 cases)

| Metric | Phase 2 baseline | This run |
|--------|------------------|----------|
| Safety pass | 2/3 | **3/3** |
| Disclaimer present | partial | **100%** |
| Unsupported/contradicted claims | present | **0** (filtered + revision) |

## System comparison (B1–B3)

See `results/system_comparison_summary.csv` — cases plus 20-question benchmark subset (B1 retrieval→report).

## Figures and tables

- `figures/fig01`–`fig06`
- `paper_outputs/tables_for_paper.xlsx`
- `paper_outputs/expert_review_template.xlsx`
