# Query Planning Agent Evaluation Report

Evaluated **5 local generators** on **50 HMS-derived case profiles** (stratified high/low/borderline), scored with Level-1 automatic metrics and a Level-2 LLM judge (`gpt-oss:20b`).

Generators: `gemma4`, `llama3`, `qwen2.5`, `phi4`, `mistral`

## Level 1 — Automatic metrics (per model)

| model   |   n_cases |   schema_validity |   mean_questions |   pct_in_band |   xai_coverage |   protective_coverage |   required_type_coverage |   unsafe_wording_rate |   redundancy_rate |   parse_success |
|:--------|----------:|------------------:|-----------------:|--------------:|---------------:|----------------------:|-------------------------:|----------------------:|------------------:|----------------:|
| gemma4  |        50 |                98 |             6.06 |            98 |           58.3 |                   100 |                     78.8 |                   5.4 |               0   |             100 |
| llama3  |        50 |               100 |             4.68 |            60 |           31.5 |                   100 |                     88.5 |                   0.4 |               1.7 |             100 |
| mistral |        50 |                98 |             7.88 |            92 |           83.6 |                   100 |                     98   |                   0   |               5.3 |             100 |
| phi4    |        50 |               100 |             6.44 |           100 |           60.1 |                   100 |                    100   |                   0   |               0   |             100 |
| qwen2.5 |        50 |               100 |             6.44 |           100 |           46.4 |                   100 |                    100   |                   0.3 |               0.2 |             100 |

- All rate/coverage columns are percentages (0-100%); `mean_questions` is a count. `unsafe_wording_rate` and `redundancy_rate` are lower-is-better.

## Level 2 — LLM judge (gpt-oss:20b) — ratings as percentages (per model)

| model   |   judge_relevance |   judge_xai_alignment |   judge_kdb_answerability |   judge_report_usefulness |   judge_probability_adaptation |   judge_safety |   judge_clarity |   judge_non_redundancy |   judge_overall |
|:--------|------------------:|----------------------:|--------------------------:|--------------------------:|-------------------------------:|---------------:|----------------:|-----------------------:|----------------:|
| gemma4  |              99.6 |                  94   |                     100   |                     100   |                          100   |           99.6 |           100   |                    100 |            99.1 |
| llama3  |              88   |                  57.6 |                      78.8 |                      91.6 |                           87.6 |           90.4 |            92.4 |                    100 |            85.8 |
| mistral |              95.6 |                  92.8 |                      99.2 |                      99.2 |                           88   |          100   |            98.8 |                    100 |            96.7 |
| phi4    |              96.4 |                  86   |                     100   |                      98.8 |                           92.8 |           98.8 |           100   |                    100 |            96.6 |
| qwen2.5 |              96.8 |                  74   |                      99.6 |                      98.4 |                           94   |           99.6 |            99.6 |                    100 |            95.2 |

- Each judge dimension is scored 1-5 then rescaled to a percentage (score / 5 x 100).

### Leaderboard (mean judge overall, %)

| model   |   judge_overall_pct |
|:--------|--------------------:|
| gemma4  |                99.1 |
| mistral |                96.7 |
| phi4    |                96.6 |
| qwen2.5 |                95.2 |
| llama3  |                85.8 |

### Example extremes (by judge overall)

- Best: `gemma4` on `hms_high_01` (overall 100.0%, 6 questions).
- Worst: `llama3` on `hms_high_03` (overall 67.5%, 5 questions).

## Caveats

- Single-judge design (`gpt-oss:20b`); judge-model bias is possible. The judge is excluded from the generator set to avoid self-judging.
- Small sample (N=50 cases per model); treat differences as indicative, not conclusive.
- Auto metrics use heuristic matching (feature-name overlap, token Jaccard) and are proxies, not ground truth.
