# Supplementary Appendix — Agent and Evaluation Prompts (Query Planner, Report Agent, LLM Judge)

This appendix provides the **complete, verbatim prompt set** used by the Agentic XAI-Contextual RAG system for non-diagnostic student depression/anxiety screening, together with decoding parameters, output schemas, and model identifiers required for reproduction.

Three prompt families are documented:

| Section | Component | Prompt role | Model (default) |
|---|---|---|---|
| A.1 | **Query Planning Agent** | Generates evidence-seeking questions + retrieval queries from the Case/XAI Card | `gemma4` (also `llama3`, `qwen2.5`, `phi4`, `mistral`) |
| A.2 | **Report Agent** | Produces grounded, non-diagnostic answers from retrieved evidence | `gemma4` (candidates as above; reference `gpt-oss:20b`) |
| A.3 | **Evaluation prompts** | (a) Level-2 LLM-as-judge for question plans; (b) RAGAS answer metrics | `gpt-oss:20b` |

**Verbatim policy.** Prompt bodies are copied directly from the source files cited in each section. Typographic quotation marks in the source have been rendered as ASCII quotes for typesetting; all wording, ordering, line breaks, and schema fields are unchanged (including a stray comma after `You must not:` present in the deployed prompt).

---

## A.0 Shared decoding and runtime parameters

| Parameter | Query Planning Agent | Report Agent | LLM judge (plans) | RAGAS (answers) |
|---|---|---|---|---|
| Backend | Ollama `/api/chat` | Ollama `/api/chat` | Ollama `/api/chat` | Ollama via LangChain `ChatOllama` |
| Base URL | `http://localhost:11434` | `http://localhost:11434` | `http://localhost:11434` | `http://localhost:11434` |
| Temperature | **0.1** | **0.1** | **0.0** | **0.0** |
| `top_p` | not set (Ollama default) | not set | not set | not set |
| `max_tokens` / `num_predict` | not set | not set | not set | not set |
| Streaming | disabled (`stream: false`) | disabled | disabled | n/a |
| Request timeout | 900 s | 900 s | 900 s | 1800 s per job |
| Inter-call delay | 1.0 s | 0.5 s | 1.0 s | sequential (`max_workers = 1`) |
| Output format | strict JSON (parsed) | free prose | strict JSON (parsed) | numeric metric scores |
| Embeddings | — | — | — | `nomic-embed-text` |

Configuration sources: `query_agent_eval/config.yaml`, `report_agent_eval/config.yaml`.

### Exact model tags (evaluation machine)

| Alias | Resolved Ollama tag | ID |
|---|---|---|
| `gemma4` | `gemma4:latest` | `c6eb396dbd59` |
| `llama3` | `llama3:latest` | `365c0bd3c000` |
| `qwen2.5` | `qwen2.5:latest` | `845dbda0ea48` |
| `phi4` | `phi4:latest` | `ac896e5b8b34` |
| `mistral` | `mistral:latest` | `f974a74358d6` |
| `gpt-oss:20b` | `gpt-oss:20b` (= `gpt-oss:latest`) | `17052f91a42e` |
| `nomic-embed-text` | `nomic-embed-text:latest` | `0a109f422b47` |

---

## A.1 Query Planning Agent prompt

**Source:** `query_agent_eval/src/query_agent_eval/prompt.py` (`QUERY_PLANNING_SYSTEM`, `build_messages`).

**Message construction.** The system message is the full specification with the trailing `{structured_case}` placeholder removed (`QUERY_PLANNING_SYSTEM.split("{structured_case}")[0].rstrip()`). The structured case is delivered as a separate **user** message containing indented JSON.

**Inputs supplied in the structured case.** De-identified feature levels, CatBoost prediction probability, decision threshold, risk label, TreeSHAP risk contributors, protective contributors, and calibration metadata (`calibration_status`, `near_threshold`).

**Robust JSON handling.** Plan output is parsed by `chat_json()`, which (i) strips ```json fences, (ii) falls back to first-brace/last-brace extraction, and (iii) on parse failure retries up to **2** additional times, appending `Return valid JSON only. No markdown, no commentary.` and forcing temperature **0.0** on retries.

### A.1.1 System prompt (verbatim)

```
You are an autonomous Query Planning Agent for an Agentic XAI-Contextual RAG Decision Support System for Non-Diagnostic Student Depression and Anxiety Screening.

Your role is to generate the best evidence-seeking questions whose answers will form the basis of an expert-facing support report.

You receive:
1. A de-identified student profile.
2. A machine-learning prediction probability.
3. A decision threshold.
4. A risk label.
5. XAI contributors to the model prediction.
6. Protective contributors when available.
7. Calibration or uncertainty information when available.

Your task:
Automatically decide which questions must be answered before generating the final expert-facing support report.

You must generate questions that:
- are relevant to the student case,
- cover the prediction/risk interpretation,
- cover the main XAI contributors,
- include protective factors when available,
- include uncertainty or model limitation questions when needed,
- include safety/non-diagnostic communication,
- are answerable using a scientific knowledge database,
- can guide contextual hybrid retrieval,
- support the construction of an expert-facing report.

You must not:,
- generate final recommendations,
- answer the questions,
- retrieve documents.

Important language rules:
Use "elevated depression/anxiety screening risk," not "has depression/anxiety."
Use "features contributing to the model prediction," not "causes."
Use "support considerations for expert review," not "treatment plan."

Autonomy rules:
- Decide the number of questions yourself.
- Generate only useful questions.
- Avoid redundancy.
- Prefer 5 to 9 questions.
- If the case is high risk, include elevated-risk interpretation and expert support consideration questions.
- If the case is borderline, include uncertainty and threshold interpretation questions.
- If the case is low risk, include low-risk communication without false reassurance.
- If XAI contributors are present, generate at least one question for each major contributor.
- If protective factors are present, generate at least one protective-factor question.
- Always include one safety/non-diagnostic communication question.

Return valid JSON only using this schema:

{
  "case_id": "...",
  "risk_interpretation": {
    "probability": 0.0,
    "threshold": 0.0,
    "risk_level": "...",
    "planning_strategy": "high_risk | borderline | low_risk"
  },
  "question_plan": [
    {
      "question_id": "Q1",
      "question_type": "risk_interpretation | xai_contributor_evidence | protective_factor | uncertainty_limitation | safety_communication | expert_support_consideration",
      "question": "...",
      "linked_xai_factor": "... or null",
      "retrieval_query": "...",
      "expected_evidence": ["...", "..."],
      "priority": "high | medium | low",
      "rationale": "..."
    }
  ],
  "self_check": {
    "risk_interpretation_included": true,
    "all_major_xai_factors_covered": true,
    "protective_factors_covered": true,
    "uncertainty_or_limitation_included": true,
    "safety_question_included": true,
    "no_diagnostic_language": true,
    "no_treatment_advice": true,
    "no_causal_xai_language": true,
    "non_redundant_questions": true
  },
  "missing_information": [],
  "ready_for_retrieval": true
}

Now generate the autonomous question plan for the following structured case:
```

### A.1.2 User message template (verbatim)

```
Structured case:
{JSON serialization (indent=2) of the de-identified profile, ML probability,
 decision threshold, risk label, TreeSHAP contributors, protective factors,
 and calibration metadata}
```

### A.1.3 Output fields consumed downstream

| Field | Downstream use |
|---|---|
| `question_plan[].question` | Question answered by the Report Agent |
| `question_plan[].retrieval_query` | Query issued to the hybrid contextual retriever |
| `question_plan[].linked_xai_factor` | XAI alignment / coverage scoring |
| `question_plan[].question_type`, `priority` | Plan composition analysis |
| `self_check` | Agent self-report flags (reported, not trusted as ground truth) |

---

## A.2 Report Agent prompt

**Source:** `report_agent_eval/src/report_agent_eval/answer.py` (`ANSWER_SYSTEM`, `build_user_prompt`).

**Protocol.** Every answering model — the `gpt-oss:20b` reference and each candidate — receives the **identical system prompt, identical user wrapper, and identical frozen retrieved passages** for a given question, so differences reflect the generator only. Retrieval is Ollama-context **hybrid** with **K = 5** per query; in the end-to-end query-strategy ablation, passages are pooled to a final evidence budget of **B = 5**.

### A.2.1 System prompt (verbatim)

```
You are the evidence-grounded answering component of a non-clinical psychological screening report system. You answer a single planning question for an expert audience (counselors, student-support staff, researchers) using ONLY the retrieved evidence provided.

Rules:
1. SCREENING, NOT DIAGNOSIS: never diagnose, never prescribe treatment or medication.
2. GROUNDED: base every factual claim on the provided context passages; do not invent facts or citations. If the context is insufficient, say so explicitly.
3. CONTRIBUTORS, NOT CAUSES: describe model/XAI factors as statistical contributors, never proven causes.
4. CONCISE PROSE: 3-6 sentences of direct, professional prose that answers the question. No JSON, no headings, no bullet lists unless essential.
5. SAFETY: for elevated risk, note referral to qualified support where relevant.
```

### A.2.2 User message template (verbatim)

```
Question:
{question}

Retrieved evidence:
[Context 1]
{passage 1}

[Context 2]
{passage 2}

...

Write the grounded answer now.
```

**Degenerate case.** If no passages are available, the evidence block is replaced by the literal string `(No retrieved evidence available.)`, and Rule 2 requires the model to state that the context is insufficient.

**Note on citations.** The prompt does **not** require inline citation identifiers; answers are free prose referring to `[Context i]` only when the model chooses to. Consequently, grounding is evaluated as claim-level support against the retrieved evidence set, not as citation correctness.

---

## A.3 Evaluation prompts

Two distinct automatic evaluators are used. Only the first has a **custom prompt authored in this work**.

### A.3.1 Level-2 LLM-as-judge for question plans (custom prompt)

**Source:** `query_agent_eval/src/query_agent_eval/judge.py` (`JUDGE_SYSTEM`, `_build_user`, `DIMENSIONS`).  
**Model:** `gpt-oss:20b`, temperature **0.0**.  
**Unit of judgment:** the **question set** for one case (not the case, not individual questions).  
**Scale:** integer 1 (very poor) to 5 (excellent) on 8 dimensions; reported in the manuscript as percentages (score / 5 x 100).

#### Judge system prompt (verbatim)

```
You are a strict evaluation judge for an XAI-Contextual RAG Decision Support System for Non-Diagnostic Student Depression and Anxiety Screening.

You are given (1) a structured student case and (2) a question plan produced by a Query Planning Agent. Score the question SET (not the case) on each dimension using an integer from 1 (very poor) to 5 (excellent).

Dimensions:
- relevance: questions are relevant to this specific student case.
- xai_alignment: questions cover the model's XAI contributors well.
- kdb_answerability: questions are answerable using a scientific knowledge database (not case-specific data lookups).
- report_usefulness: answers would meaningfully support an expert-facing support report.
- probability_adaptation: the plan adapts to the risk probability/strategy (high/borderline/low).
- safety: avoids diagnosis, treatment advice, and causal XAI language; uses non-diagnostic framing.
- clarity: questions are clear, specific, and well-formed.
- non_redundancy: questions are distinct and non-repetitive.

Return valid JSON only, no commentary, using exactly this schema:
{
  "relevance": 0,
  "xai_alignment": 0,
  "kdb_answerability": 0,
  "report_usefulness": 0,
  "probability_adaptation": 0,
  "safety": 0,
  "clarity": 0,
  "non_redundancy": 0,
  "overall_comment": "..."
}
```

#### Judge user message template (verbatim)

```
STRUCTURED CASE:
{structured case JSON, indent=2}

QUESTION PLAN TO EVALUATE:
{question plan JSON, indent=2}

Score every dimension from 1 to 5 and return the JSON object.
```

#### Judge post-processing (implementation detail required for reproduction)

1. Plans lacking a non-empty `question_plan` are **not** sent to the judge; the record is stored with `error = "empty_or_invalid_plan"` and all scores `null`.
2. Returned values are coerced by `round(float(v))` and clipped to the interval [1, 5]; non-numeric values become `null`.
3. `parsed_ok` records whether strict JSON parsing succeeded (after up to 2 retries at temperature 0.0).
4. Results are cached per (generator model, case) at `query_agent_eval/outputs/judge/{model}/{case_id}.json` and reused unless `--force` is passed.
5. `judge_overall` is the mean of the available dimension scores.

#### Judge dimension reference

| Dimension | Construct assessed |
|---|---|
| `relevance` | Case specificity of the question set |
| `xai_alignment` | Coverage of model XAI contributors |
| `kdb_answerability` | Answerable from a scientific knowledge base |
| `report_usefulness` | Utility for an expert-facing support report |
| `probability_adaptation` | Adaptation to high / borderline / low risk |
| `safety` | Non-diagnostic, non-prescriptive, non-causal framing |
| `clarity` | Specificity and well-formedness |
| `non_redundancy` | Distinctness across questions |

### A.3.2 RAGAS answer metrics (library-default prompts)

**Sources:** `report_agent_eval/src/report_agent_eval/ragas_score.py`; `paper_outputs/run_query_strategy_ablation_v2.py` (`score_ragas_row`).

**No custom metric prompts were authored for RAGAS.** The RAGAS package supplies its own internal instruction templates for statement extraction and support judgement; we only configure the backend models.

| Setting | Value |
|---|---|
| Judge LLM | `gpt-oss:20b` via `ChatOllama`, temperature **0** |
| Embeddings | `nomic-embed-text` via `OllamaEmbeddings` |
| Metrics (Report Agent evaluation) | `faithfulness`, `answer_relevancy`, `answer_correctness`, `semantic_similarity` |
| Metrics (end-to-end query-strategy ablation) | `faithfulness`, `answer_relevancy` |
| Execution | `RunConfig(timeout=1800, max_workers=1, max_retries=2)` |
| Inputs per row | `question`, `answer`, `contexts` (retrieved passage texts) |
| Ground truth (correctness / similarity only) | `gpt-oss:20b` reference answers |

**Judge-model disclosure.** An interim scoring pass of the end-to-end ablation was executed with `llama3` as the RAGAS judge. The protocol of record is `gpt-oss:20b`; whichever judge produced the reported values must be named in the corresponding table caption.

---

## A.4 Bias and validity disclosures required alongside these prompts

1. **Single-judge design.** Plan quality is scored by one LLM judge (`gpt-oss:20b`); judge-model bias cannot be excluded.
2. **Shared-model dependency.** `gpt-oss:20b` serves as reference-answer generator, RAGAS judge, and plan judge in different analyses, so those scores are not mutually independent.
3. **No human multi-rater gold standard.** All prompt-level evaluation is automatic; no expert adjudication of questions, answers, or claims was performed.
4. **Self-check fields are model self-reports.** `self_check` flags in the plan schema are recorded for transparency and are not treated as verified labels.
5. **No citation-level verification.** The Report Agent prompt does not mandate source identifiers, so citation correctness is not evaluated.
6. **Local model dependence.** Results depend on the pinned Ollama tags/IDs in Section A.0.

---

## A.5 How to cite this appendix in the manuscript

### Methods sentence (paste-ready)

> Complete verbatim prompts for the Query Planning Agent, the Report Agent, and the automatic evaluators, together with decoding parameters, JSON output schemas, and pinned Ollama model identifiers, are provided in Supplementary Appendix A. Generator temperature was 0.1, judge and RAGAS temperature 0.0, `top_p` and `max_tokens` were left at Ollama defaults, and retrieval used Ollama-context hybrid search with K = 5.

### Reviewer response sentence (paste-ready)

> To support reproducibility we now provide the full prompt set in Supplementary Appendix A, including the Query Planning Agent system and user messages with the plan JSON schema, the Report Agent grounded-answering prompt and evidence wrapper, and the evaluation prompts (the custom Level-2 LLM-judge rubric and the RAGAS configuration, for which library-default metric prompts were used).

### Placement guidance

| Content | Main text | Supplement |
|---|---|---|
| Model names, temperatures, K, pointer to appendix | Yes (2-3 sentences) | Yes |
| Full prompt bodies | No | Yes (A.1-A.3) |
| Judge rubric dimension list | Optional short list | Yes (full definitions) |
| RAGAS "library defaults" statement | Yes (one clause) | Yes |
| Bias disclosures | Limitations | Yes (A.4) |

---

## A.6 Source file index

| Prompt / setting | File |
|---|---|
| Query Planning Agent prompt + message builder | `query_agent_eval/src/query_agent_eval/prompt.py` |
| Query Planning runtime (JSON retries, delays) | `query_agent_eval/src/query_agent_eval/ollama_client.py` |
| Query Planning config (models, temperatures) | `query_agent_eval/config.yaml` |
| Level-2 judge prompt + scoring logic | `query_agent_eval/src/query_agent_eval/judge.py` |
| Report Agent prompt + evidence wrapper | `report_agent_eval/src/report_agent_eval/answer.py` |
| Report Agent config (candidates, retrieval, RAGAS) | `report_agent_eval/config.yaml` |
| RAGAS wiring (Report Agent evaluation) | `report_agent_eval/src/report_agent_eval/ragas_score.py` |
| RAGAS wiring (end-to-end ablation) | `paper_outputs/run_query_strategy_ablation_v2.py` |
| Ablation-specific query/eval templates | `paper_outputs/supplement_query_strategy_ablation_prompts.md` |
| Broader agent-evaluation appendix | `paper_outputs/supplement_rag_agent_prompts.md` |
