# Ollama Contextual Retrieval Evaluation

Standalone evaluation of **contextual embeddings** using **only local Ollama** — no Claude, Voyage, or cloud APIs.

Methodology follows Anthropic's [Contextual Embeddings cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide):

1. For each chunk, optionally generate a short **situating context** with an Ollama chat model.
2. Prepend that context to the chunk text before embedding (**contextualized chunk**).
3. Index the same contextualized text for **both** dense vectors and **BM25** (Contextual BM25).
4. Compare retrieval quality across representations and retriever types.

## Compared conditions

| Representation | Description |
|----------------|-------------|
| `plain` | Raw chunk text only |
| `metadata` | Title, expertise, category, page prefix + chunk |
| `ollama_context` | Ollama-generated situating sentence + chunk (cookbook-style) |

| Retrieval | Description |
|-----------|-------------|
| `dense` | Cosine similarity on `nomic-embed-text` embeddings |
| `hybrid` | RRF fusion of dense + BM25 on the **same** indexed text |

## Data sources (reused)

- Corpus: `../controlled_agentic_xai_rag/data/processed_chunks.csv` (5,740 chunks)
- Benchmark: `../controlled_agentic_xai_rag/data/benchmarks/rag_benchmark_100_questions.csv`
- Gold resolution: `controlled_agentic_xai_rag.evaluation.benchmark_gold`

## Setup

```bash
cd ollama_contextual_eval
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:latest   # or set context_model in config.yaml
```

Ensure Ollama is running at `http://localhost:11434`.

## Quick smoke test (~few minutes)

Skips LLM context generation; uses 400 chunks and 10 questions:

```bash
python scripts/run_eval.py --chunk-limit 400 --question-limit 10 --skip-ollama-context
```

## Full ablation (slow)

Generating Ollama context for all 5,740 chunks is **one chat call per chunk** (cached in `cache/ollama_context_*.jsonl`). Embedding all three representations is also expensive but cached in `cache/embeddings_*.npy`.

```bash
# Step 1: plain + metadata only (no LLM context) — still useful baseline
python scripts/run_eval.py --skip-ollama-context

# Step 2: include ollama_context (uses cache after first run)
python scripts/run_eval.py
```

Options:

- `--representations plain,metadata,ollama_context`
- `--modes dense,hybrid`
- `--force-regenerate-context` — rebuild Ollama situating contexts
- `--force-reembed` — recompute embeddings

## Outputs

Written to `outputs/`:

- `summary_latest.csv` — mean hit@k, recall@k, MRR, nDCG per condition
- `per_question_latest.csv` — per-question breakdown
- `report_*.json` — full run metadata + context examples

## Interpretation (from prior smaller-corpus runs)

On a smaller psychology corpus, **Ollama context + hybrid** tended to beat plain hybrid on acronym-heavy queries (PHQ-9, GAD-7, SHAP). Dense-only contextualization showed smaller gains. See `../knowledge_db/logs/contextual_retrieval_results_for_paper.md` for the earlier write-up.

This folder replicates that design on the **full 100-question scientific benchmark** with chunk-level gold labels.

## Project layout

```
ollama_contextual_eval/
├── config.yaml
├── requirements.txt
├── scripts/run_eval.py
├── src/ollama_contextual_eval/
│   ├── contextualize.py   # plain / metadata / ollama context builders
│   ├── ollama_client.py   # embed + chat
│   ├── retriever.py       # dense + hybrid RRF
│   └── evaluate.py        # ablation runner
├── cache/                 # contexts + embedding matrices
└── outputs/               # CSV + JSON results
```
