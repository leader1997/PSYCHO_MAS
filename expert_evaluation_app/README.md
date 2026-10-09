# HMS Expert Evaluation App

React (Vite + TypeScript) interface for expert review of AI-assisted XAI + RAG mental-health **screening** outputs. Ratings are logged for manuscript reporting.

**Not a diagnostic tool.** Stimuli use de-identified HMS-derived evaluation profiles (`query_agent_eval/data/case_profiles_50.jsonl`) plus frozen report-agent evidence/answers where available.

## Run

```powershell
cd expert_evaluation_app
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

Rebuild the case bank from profiles (optional):

```powershell
python scripts/build_cases_from_profiles.py
```

## Study flow

1. Enter a stable **Expert ID** (e.g. `E01`).
2. For each of **9 cases** (3 high / 3 borderline / 3 low), review left-panel stimuli (dataset feature levels, probability, XAI, evidence, report).
3. Answer **Q1–Q20** (1–5 Likert; Appendix sections A–E). **Q21–Q23** free text optional.
4. **Save Progress** anytime; **Next Case** requires all 20 Likert scores.
5. After the last case, click **Finish study**, then **Export CSV** and **Export JSONL**.

Progress persists in `localStorage` under key `expert_eval_v2_store`.

## Case bank

- Version: `case_bank_v4_9_real`
- Sources: `hms_high_01..03`, `hms_borderline_01..03`, `hms_low_01..03`
- Evidence: top hits from `report_agent_eval/outputs/contexts/<case_id>/`
- Reports: `gemma4` answers when present, else `gpt-oss_20b`; two low cases without stored answers use a profile+evidence fallback summary (`report_source` field documents this)
- Academic impairment (`aca_impa`) is hidden from XAI / profile / evidence title display

## Export schema

### JSONL

One object per `save` / `complete_case` / `finish_study` with `scores` (`q1`…`q20`), `qualitative` (`q21`…`q23`), and `stimulus_snapshot`.

### CSV

Wide table: one row per case with `q1`…`q20` and `q21`…`q23`, plus `source_case_id`, stratum, probability, etc.

### Manuscript analysis

1. Pool expert CSVs (same `case_bank_version` / `questions_version`).
2. Per item `q1`…`q20`: report **mean ± SD** (optionally by stratum).
3. Agreement: ICC on expert×case matrices per item.
4. Summarize Q21–Q23 thematically.

## Build

```powershell
npm run build
```
