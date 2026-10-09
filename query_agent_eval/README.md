# Query Planning Agent Evaluation Harness

Evaluates the quality of question plans produced by the autonomous **Query Planning Agent**
across **7 local Ollama LLMs**, on **50 real HMS-derived case profiles**
(20 elevated / 20 low / 10 borderline), using **Level-1 automatic metrics** and a
**Level-2 LLM judge (`gpt-oss:20b`)**.

## Pipeline

```
HMS CSV + CatBoost/TreeSHAP  ->  50 structured profiles
                                     |
                    5 Ollama generators (llama3, qwen2.5, mistral, gemma2, phi4)
                                     |
                          250 question plans (JSON, cached)
                            /                      \
              Level-1 auto metrics          Level-2 gpt-oss:20b judge
                            \                      /
                          Aggregate -> CSVs + report.md
```

## Setup

```bash
pip install -r requirements.txt
# Pull the models once (7 generators + judge):
ollama pull gemma4 && ollama pull mistral-small && ollama pull llama3 && \
ollama pull qwen2.5 && ollama pull gemma2 && ollama pull phi4 && \
ollama pull mistral && ollama pull gpt-oss:20b
```

Profile generation reuses the `controlled_agentic_xai_rag` CatBoost + TreeSHAP stack
(Phase-1 model artifacts and the HMS CSV must be present).

## Usage

```bash
# 1. Build the 50 stratified profiles
python scripts/build_profiles.py

# 2a. Full run (generate + judge + aggregate)
python scripts/run_all.py

# 2b. Or step-by-step
python scripts/run_generation.py
python scripts/run_judge.py
python scripts/aggregate.py

# Smoke test (2 cases x 2 models)
python scripts/run_all.py --limit 2 --models gemma4 mistral-small
```

Both `--models` and `--limit` are supported everywhere; all plans/judgments are cached
per file so interrupted runs resume automatically.

## Metrics

### Level 1 — automatic (`metrics_auto.py`)
| Metric | Meaning |
| --- | --- |
| Schema validity | Valid JSON that passes `schemas/question_plan.json` |
| Mean # questions | `len(question_plan)`; % within the 5-9 band |
| XAI factor coverage | Fraction of case XAI contributors linked/mentioned |
| Protective factor coverage | ≥1 protective question when protective factors exist |
| Required question-type coverage | Expected types by strategy present / expected |
| Unsafe wording rate | Questions using diagnosis/treatment/causal language |
| Redundancy rate | Question pairs with token-set Jaccard ≥ 0.8 |

### Level 2 — LLM judge (`judge.py`, `gpt-oss:20b`, 1-5 each)
relevance, XAI alignment, KDB answerability, report usefulness,
probability adaptation, safety, clarity, non-redundancy.

Reported as **percentages** (score / 5 x 100) in `level2_by_model.csv` and `report.md`.
Level-1 rate/coverage metrics are also reported as percentages.

## Outputs
- `data/case_profiles_50.jsonl`
- `outputs/plans/{model}/{case_id}.json`
- `outputs/judge/{model}/{case_id}.json`
- `outputs/level1_by_model.csv`, `outputs/level2_by_model.csv`, `outputs/per_case_long.csv`
- `outputs/report.md`

## Caveats
- Single-judge design (judge is excluded from generators to avoid self-judging bias).
- Small N (50 cases/model) — differences are indicative, not conclusive.
- Auto metrics use heuristic matching and are proxies, not ground truth.
