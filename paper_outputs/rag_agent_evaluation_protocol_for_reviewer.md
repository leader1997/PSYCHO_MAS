# RAG and Agent Evaluation Protocol — Reviewer Response Package

**Purpose.** Address the review request for reproducibility/transparency on annotations, sample sizes, prompts, model versions, and statistical uncertainty — **without changing the RAG architecture**.

**Related files**
- Prompt appendix: [`supplement_rag_agent_prompts.md`](supplement_rag_agent_prompts.md)
- Bootstrap CIs: [`query_agent_bootstrap_ci.csv`](query_agent_bootstrap_ci.csv), [`report_agent_ragas_bootstrap_ci.csv`](report_agent_ragas_bootstrap_ci.csv)
- Bootstrap script: [`bootstrap_rag_agent_uncertainty.py`](bootstrap_rag_agent_uncertainty.py)

---

## 1. Reviewer concern

> RAG and agent evaluations lack sufficient details about annotations, sample sizes, prompts, model versions, and statistical uncertainty.

This is a **reporting** gap. The evaluations already exist; the manuscript must document the five elements below.

---

## 2. Protocol summary table

| Element | Query Planning Agent | Report Agent (answer RAGAS) | Contextual retrieval (related) |
|---|---|---|---|
| **Sample size** | 50 cases (20 elevated / 20 low / 10 borderline); 5 generators → **250 plans**; **1,575 questions** | **3 cases** (`hms_low_01`, `hms_borderline_01`, `hms_high_01`); frozen `gemma4` plans → **18 questions** per candidate; 5 candidates + 1 reference | Collection queries **n=63**; manual challenge **n=7**; depth **K=5** |
| **Annotations / reference** | **No human multi-rater annotation.** Level-1 automatic rules + single LLM judge (`gpt-oss:20b`) | **No human reference answers.** `gpt-oss:20b` generates reference answers used as RAGAS ground truth; faithfulness uses frozen retrieved contexts | Category/corpus-construction relevance for collection queries; 7 hand-written challenge queries |
| **Prompts** | See Supplement (Query Planning system + structured-case user JSON) | See Supplement (answer system + question/contexts user wrapper) | Context-generation model `qwen2.5` (chunk contextualization), separate from agent prompts |
| **Models** | Generators: `gemma4`, `llama3`, `qwen2.5`, `phi4`, `mistral` (Ollama tags). Judge: `gpt-oss:20b` | Same 5 candidates; reference + RAGAS LLM: `gpt-oss:20b`; embeddings: `nomic-embed-text` | Embeddings: `nomic-embed-text`; context LLM: `qwen2.5` |
| **Generation params** | Temperature **0.1**; judge **0.0**; inter-call delay **1.0 s**; `top_p` / `max_tokens` **not overridden** (Ollama defaults) | Temperature **0.1**; RAGAS LLM temp **0**; retrieval **hybrid Ollama-context, K=5**; delay **0.5 s**; `top_p` / `max_tokens` not overridden | Hybrid = dense + BM25; K=5 |
| **Uncertainty** | Mean ± SD and **95% bootstrap CI** across cases (B=2000, seed=42) | Mean ± SD and **95% bootstrap CI** across questions (B=2000, seed=42) | Point estimates in existing retrieval tables (N small for manual set) |

**Also in manuscript (brief):** end-to-end S0–S3 ablation on **3 preset cases** (pipeline report) — exploratory due to small N.

---

## 3. Exact model versions (local Ollama, recorded 2026-09-22)

Short names in configs resolve to these installed tags:

| Role | Config name | Installed tag | Ollama ID | Size / notes |
|---|---|---|---|---|
| Generator | `gemma4` | `gemma4:latest` | `c6eb396dbd59` | 8.0B, Q4_K_M, ~9.6 GB |
| Generator | `llama3` | `llama3:latest` | `365c0bd3c000` | 8.0B, Q4_0, ~4.7 GB |
| Generator | `qwen2.5` | `qwen2.5:latest` (= `qwen2.5:7b`) | `845dbda0ea48` | 7.6B, Q4_K_M, ~4.7 GB |
| Generator | `phi4` | `phi4:latest` | `ac896e5b8b34` | 14.7B, Q4_K_M, ~9.1 GB |
| Generator | `mistral` | `mistral:latest` | `f974a74358d6` | 7.2B, Q4_0, ~4.1 GB |
| Judge / reference / RAGAS LLM | `gpt-oss:20b` | `gpt-oss:20b` (= `gpt-oss:latest`) | `17052f91a42e` | 20.9B, MXFP4, ~13 GB |
| Embeddings | `nomic-embed-text` | `nomic-embed-text:latest` | `0a109f422b47` | 137M, F16, ~274 MB |

**Backend:** local Ollama at `http://localhost:11434`.  
**Bias disclosure:** `gpt-oss:20b` is both the Report Agent **reference answer generator** and the **RAGAS scoring LLM**.

---

## 4. Annotation / reference-construction procedure (honest)

### Query Planning Agent
1. **Level-1:** deterministic code metrics (schema validity, question-count band 5–9, XAI/protective/type coverage, unsafe-wording heuristics, redundancy).  
2. **Level-2:** single LLM judge (`gpt-oss:20b`, temperature 0) scores each plan on 8 dimensions (1–5), reported as percentages.  
3. **Not used:** human expert multi-annotator labels; no Cohen’s κ / adjudication protocol.

### Report Agent (RAGAS answer quality)
1. Frozen `gemma4` query plans define the questions.  
2. Identical contextual-RAG passages (Ollama-context hybrid, K=5) are frozen per question.  
3. `gpt-oss:20b` writes a **reference answer** on those passages.  
4. Each candidate LLM answers the same question on the same passages.  
5. RAGAS metrics: faithfulness (answer vs contexts); answer_relevancy (answer vs question); answer_correctness & semantic_similarity (answer vs gpt-oss reference).  
6. **Not used:** human-written reference answers or passage relevance adjudication for this arm.

### Contextual retrieval
- Collection-query relevance follows corpus-construction / category targeting.  
- Manual challenge set: **7** authored evaluation queries (not a multi-rater panel).

---

## 5. Statistical uncertainty (computed from existing outputs)

Bootstrap: **B = 2000**, seed **42**.

### Query Planning — judge overall (%), n = 50 cases per model

| Model | Mean ± SD | 95% CI |
|---|---|---|
| gemma4 | 99.2 ± 1.4 | 98.8–99.5 |
| mistral | 96.7 ± 3.3 | 95.8–97.6 |
| phi4 | 96.6 ± 3.9 | 95.5–97.6 |
| qwen2.5 | 95.3 ± 4.6 | 93.9–96.4 |
| llama3 | 85.8 ± 5.7 | 84.2–87.3 |

### Report Agent RAGAS — faithfulness (0–1), n = 18 questions per model

| Model | Mean ± SD | 95% CI |
|---|---|---|
| gemma4 | 0.841 ± 0.206 | 0.744–0.926 |
| mistral | 0.624 ± 0.262 | 0.506–0.736 |
| phi4 | 0.549 ± 0.247 | 0.437–0.659 |
| llama3 | 0.459 ± 0.283 | 0.334–0.582 |
| qwen2.5 | 0.366 ± 0.309 | 0.237–0.500 |

Full metric-wise CIs: `report_agent_ragas_bootstrap_ci.csv`, `query_agent_bootstrap_ci.csv`.

**Caveat:** Report Agent n=18 questions (3 cases) → CIs are wide; treat as exploratory sensitivity, not definitive ranking.

---

## 6. Paste-ready manuscript subsection

**RAG and Agent Evaluation Protocol.**  
The Query Planning Agent evaluation was conducted on **50** HMS-derived student cases (20 elevated / 20 low / 10 borderline), producing **250** model–case plans and **1,575** generated questions across five local Ollama generators (`gemma4:latest`, `llama3:latest`, `qwen2.5:latest`, `phi4:latest`, `mistral:latest`). The Report Agent answer-quality evaluation used frozen `gemma4` plans for **3** stratified cases (`hms_low_01`, `hms_borderline_01`, `hms_high_01`), yielding **18** question–answer instances per candidate under identical Ollama-context hybrid retrieval (**K = 5**). All candidates received identical case information and frozen retrieved passages. Exact prompt templates and generation parameters (generator temperature **0.1**; judge/RAGAS temperature **0.0**; `top_p` and `max_tokens` left at Ollama defaults) are provided in Supplementary Appendix X. The judge and Report Agent reference/RAGAS model was **`gpt-oss:20b`** (Ollama ID `17052f91a42e`); embeddings used **`nomic-embed-text:latest`**. Retrieval relevance for the contextual ablation used corpus-construction/category labels (collection **n = 63**) plus **7** manual challenge queries. Query plans were scored with automatic Level-1 rules and a single LLM judge; Report Agent correctness/similarity used `gpt-oss:20b` reference answers rather than human annotation (shared judge/reference bias disclosed). For principal metrics we report the mean ± SD and **95% bootstrap confidence intervals** across evaluation instances (B = 2000).

---

## 7. Paste-ready reviewer response

**Response:** Thank you for highlighting this issue. We have substantially expanded the description of the RAG and agent evaluation protocol. The revised manuscript now reports the exact evaluation sample sizes (Query Planning: 50 cases / 250 plans / 1,575 questions; Report Agent RAGAS: 3 cases / 18 questions per model; contextual retrieval: 63 collection + 7 manual queries), the annotation/reference-construction procedure (automatic Level-1 metrics and LLM-as-judge / `gpt-oss` reference answers — not human multi-rater annotation), prompt templates and generation parameters, exact Ollama model tags and IDs, and statistical uncertainty (mean ± SD with 95% bootstrap CIs). The complete prompts and configuration details have also been added to the Supplementary Material to improve reproducibility.

---

## 8. Suggested manuscript edits (minimal)

1. Add subsection **“RAG and Agent Evaluation Protocol”** (paragraph in §6 above).  
2. Add **Supplementary Appendix X** = content of `supplement_rag_agent_prompts.md`.  
3. In Results tables for query-agent / report-agent, append **± SD** and/or **95% CI** from the bootstrap CSVs (at least for judge overall and faithfulness).  
4. In Limitations, keep: single LLM judge; small Report Agent N; no expert multi-annotator gold; local Ollama version dependence.

---

## 9. What we did / did not invent

| Claim | Status |
|---|---|
| Sample sizes above | From configs + paper outputs + `ragas_by_model.csv` (`n_questions=18`) |
| Human annotation | **Not claimed** — procedure stated as automatic / LLM reference |
| Prompts | Verbatim from source files |
| Model versions | From `ollama list` / `ollama show` on the evaluation machine |
| Uncertainty | Bootstrap on existing `per_case_long.csv` and `ragas_per_question.csv` |

---

## 10. One-sentence bottom line

**We fixed the reviewer’s transparency request by documenting real Ns, honest non-human reference construction, full prompts/parameters, pinned Ollama tags, and bootstrap 95% CIs — without changing the RAG system.**
