# Report Agent Evaluation via RAGAS

Evaluates the **report/answer LLM** by comparing each local model's answers to
`gpt-oss` reference answers, over the **same frozen `gemma4` query-plan questions**
and the **same frozen contextual-RAG evidence**. Scored with RAGAS **answer-only**
metrics (retrieval is fixed, so context metrics are excluded).

## Pipeline

```
gemma4 plans (frozen)  ->  contextual RAG (Ollama-context hybrid, K=5)  ->  frozen contexts
                                                     |
                       gpt-oss -> reference answer   |   each LLM -> candidate answer
                                                     |
                RAGAS answer metrics (gpt-oss + nomic-embed-text), ground_truth = gpt-oss
                                                     |
                     aggregate -> CSVs + charts + report_answer_quality_for_paper.md
```

## Requirements

```bash
pip install -r requirements.txt
# Ollama models:
ollama pull gpt-oss:20b && ollama pull nomic-embed-text
ollama pull gemma4 && ollama pull llama3 && ollama pull qwen2.5 && ollama pull phi4 && ollama pull mistral
```

RAGAS uses the pinned set in `../xai_rag_depression/reproducibility/requirements-ragas.txt`.
The contextual retriever reuses `../knowledge_db_contextual_eval` (cached embeddings/contexts).

## Usage

```bash
# Smoke test (2 cases x 2 candidate models)
python scripts/run_all.py --limit 2 --models mistral llama3

# Full run (all candidates, all gemma4 cases)
python scripts/run_all.py

# Or step by step
python scripts/run_retrieval.py     # Phase 1a: freeze contexts
python scripts/run_answers.py       # Phase 1b: gpt-oss + candidate answers
python scripts/run_ragas.py         # Phase 2: RAGAS answer metrics
python scripts/aggregate.py         # tables + charts + paper md
```

All phases cache to disk and resume automatically.

## Metrics (answer only)

| Metric | Compared against | Meaning |
| --- | --- | --- |
| faithfulness | answer vs contexts | Grounding in retrieved evidence |
| answer_relevancy | answer vs question | Addresses the planned question |
| answer_correctness | answer vs gpt-oss | Correctness vs reference answer |
| semantic_similarity | answer vs gpt-oss | Meaning closeness to reference |

## Outputs

- `outputs/contexts/{case_id}/{qid}.json` - frozen retrieved evidence (shared by all models)
- `outputs/answers/{model}/{case_id}.json` - per-model answers
- `outputs/ragas_per_question.csv` - one row per (model, case, qid)
- `outputs/ragas_by_model.csv` - mean / std per model
- `outputs/ragas_by_model_stratum.csv` - per risk stratum
- `outputs/per_question_long.csv` - tidy long form
- `outputs/charts/*.svg`
- `outputs/report_answer_quality_for_paper.md`

## Caveats

- Shared-model bias: `gpt-oss` is both reference generator and RAGAS scorer.
- Corpus note: contexts are from the knowledge_db_contextual_eval corpus, not the full live KB.
- `gemma4` authored the plans and is also a candidate answerer (planning vs answering are distinct).
- N is per-question; treat differences as indicative.
