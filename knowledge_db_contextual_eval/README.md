# Knowledge DB Contextual Retrieval Evaluation

Faithful port of `knowledge_db/logs/contextual_embeddings_ollama_evaluation.ipynb` into a runnable package.

Evaluates **plain / metadata / Ollama-context** chunk representations with **dense** and **weighted hybrid** retrieval (70% dense + 30% BM25) on the original **knowledge_db** corpus (338 chunks, category-level gold).

## Data (reused from knowledge_db)

- Metadata: `../knowledge_db/metadata/selected_documents.jsonl`
- Queries: `../knowledge_db/logs/search_queries_used.txt`
- PDFs: paths in metadata under `../knowledge_db/`
- Legacy cache (optional): `../knowledge_db/logs/contextual_eval_cache/`

## Setup

```bash
cd knowledge_db_contextual_eval
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:latest
```

## Run

```bash
# Fast: reuse cached chunks + embeddings from knowledge_db/logs (plain + metadata only)
python scripts/run_eval.py --skip-ollama-context --generate-report

# Full paper replication (all 3 conditions, uses cached Ollama contexts if present)
python scripts/run_eval.py --generate-report

# Regenerate report only
python scripts/generate_report.py
```

## Outputs

- `outputs/paper_contextual_retrieval_eval_<timestamp>_summary.csv`
- `outputs/paper_contextual_retrieval_eval_<timestamp>_details.csv`
- `outputs/result_latest.json`
- `outputs/contextual_retrieval_results_for_paper.md`
- `outputs/charts/*.svg`

## Expected results (full run)

When using the legacy cache on the full corpus, numbers should match run `20260630_013954`:

- Plain hybrid Recall@5 ≈ 0.794
- Ollama hybrid Recall@5 ≈ 0.825

## vs `ollama_contextual_eval`

| | This package | `ollama_contextual_eval` |
|--|--------------|--------------------------|
| Corpus | knowledge_db PDFs (338 chunks) | controlled_agentic_xai_rag CSV (5740 chunks) |
| Gold | Category / doc_id | Chunk IDs (100Q benchmark) |
| Hybrid | 0.7 dense + 0.3 BM25 | RRF fusion |
