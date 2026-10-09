# Report Agent Answer Quality - RAGAS (scientific results)

Each evaluated local LLM answered the frozen `gemma4` query-plan questions from identical contextual-RAG evidence (Ollama-context hybrid, K=5). Answers are compared to `gpt-oss:20b` reference answers using RAGAS answer-only metrics.

- Plan source (frozen): `gemma4`
- Reference generator / ground truth: `gpt-oss:20b`
- RAGAS scorer: `gpt-oss:20b` + `nomic-embed-text` embeddings
- Unique questions: 6

## Metrics

| Metric | Compared against | Meaning |
| --- | --- | --- |
| faithfulness | answer vs contexts | Grounding in retrieved evidence |
| answer_relevancy | answer vs question | Addresses the planned question |
| answer_correctness | answer vs gpt-oss | Correctness vs reference answer |
| semantic_similarity | answer vs gpt-oss | Meaning closeness to reference |

## Main results - answer metrics by model (mean)

| model | faithfulness | answer_relevancy | answer_correctness | semantic_similarity |
| --- | --- | --- | --- | --- |
| gemma4 | 0.841 | 0.560 | 0.554 | 0.929 |
| llama3 | 0.459 | 0.704 | 0.548 | 0.937 |
| mistral | 0.624 | 0.720 | 0.515 | 0.934 |
| phi4 | 0.549 | 0.681 | 0.603 | 0.944 |
| qwen2.5 | 0.366 | 0.723 | 0.539 | 0.927 |

![Answer metrics by model](charts/ragas_answer_metrics_by_model.svg)

## Leaderboard (mean of answer metrics)

| model | mean_answer_score |
| --- | --- |
| gemma4 | 0.721 |
| mistral | 0.698 |
| phi4 | 0.694 |
| llama3 | 0.662 |
| qwen2.5 | 0.639 |

![Leaderboard](charts/ragas_leaderboard.svg)

## Sample sizes (rows scored per model)

| model | n_rows |
| --- | --- |
| gemma4 | 18 |
| llama3 | 18 |
| mistral | 18 |
| phi4 | 18 |
| qwen2.5 | 18 |

## Caveats

- Shared-model bias: `gpt-oss` is both the reference generator and the RAGAS scorer, so answer_correctness / semantic_similarity favor gpt-oss-like phrasing.
- Corpus note: contexts come from the knowledge_db_contextual_eval corpus (best-benchmarked retrieval), not the full live report-pipeline KB.
- `gemma4` authored the plans and is also a candidate answerer; planning and answering are distinct tasks but this is noted for transparency.
- N is per-question; treat model differences as indicative. Dispersion in `ragas_by_model.csv`.
