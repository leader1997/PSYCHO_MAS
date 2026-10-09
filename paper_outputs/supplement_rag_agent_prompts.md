# Supplementary Appendix X — RAG and Agent Prompts & Configuration

This appendix supports the manuscript subsection **RAG and Agent Evaluation Protocol**. Prompt text is copied verbatim from the evaluation codebase.

**Related:** For the end-to-end **query-strategy ablation (v2)** — Direct / Template / LLM-plan query templates, independent E1–E6 evaluation questions, Report Agent prompts as used in that run, and RAGAS (`gpt-oss:20b`) disclosure — see [`supplement_query_strategy_ablation_prompts.md`](supplement_query_strategy_ablation_prompts.md) (Supplementary Appendix Y).

**Consolidated prompt appendix:** For the three core prompt families (Query Planning Agent, Report Agent, and the evaluation prompts including the Level-2 LLM-judge rubric and RAGAS configuration) with decoding parameters and output schemas in one place, see [`appendix_prompts_planner_report_judge.md`](appendix_prompts_planner_report_judge.md) (Supplementary Appendix A).

---

## X.1 Generation and retrieval parameters

| Parameter | Query Planning Agent | Report Agent answers | RAGAS scoring | Contextual retrieval |
|---|---|---|---|---|
| Backend | Ollama `http://localhost:11434` | Ollama | Ollama (ChatOllama) | Ollama embeddings + hybrid BM25 |
| Temperature | **0.1** | **0.1** | **0.0** | N/A (retrieval) |
| Judge / reference temperature | **0.0** (`gpt-oss:20b`) | Reference answers: **0.1** (same answer API) | **0.0** | — |
| `top_p` | Not set (Ollama default) | Not set | Not set | — |
| `max_tokens` / `num_predict` | Not set (Ollama default) | Not set | Not set | — |
| Inter-call delay | 1.0 s | 0.5 s | sequential (`max_workers=1`) | — |
| Request timeout | 900 s | 900 s | 1800 s | — |
| Retrieval | N/A (planning only) | Ollama-context **hybrid**, **K = 5** | Uses frozen contexts | Dense + hybrid; **K = 5** |
| Embedding model | — | — | `nomic-embed-text:latest` | `nomic-embed-text:latest` |

Config sources: `query_agent_eval/config.yaml`, `report_agent_eval/config.yaml`.

---

## X.2 Exact model tags (evaluation machine)

| Config alias | Resolved Ollama tag | ID |
|---|---|---|
| `gemma4` | `gemma4:latest` | `c6eb396dbd59` |
| `llama3` | `llama3:latest` | `365c0bd3c000` |
| `qwen2.5` | `qwen2.5:latest` / `qwen2.5:7b` | `845dbda0ea48` |
| `phi4` | `phi4:latest` | `ac896e5b8b34` |
| `mistral` | `mistral:latest` | `f974a74358d6` |
| `gpt-oss:20b` | `gpt-oss:20b` / `gpt-oss:latest` | `17052f91a42e` |
| `nomic-embed-text` | `nomic-embed-text:latest` | `0a109f422b47` |

---

## X.3 Query Planning Agent — system prompt

Source: `query_agent_eval/src/query_agent_eval/prompt.py` (`QUERY_PLANNING_SYSTEM`).

At runtime, the trailing `{structured_case}` placeholder is removed from the system message; the structured case is sent as the **user** message: `Structured case:\n{JSON}`.

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

**User message template**

```
Structured case:
{JSON serialization of the de-identified profile, ML probability, threshold,
 risk label, TreeSHAP contributors, protective factors, calibration metadata}
```

---

## X.4 Query Planning Agent — Level-2 judge (summary)

- Model: `gpt-oss:20b`, temperature **0.0**
- Input: case metadata + generated question plan
- Output: 8 dimensions scored 1–5 (relevance, XAI alignment, KDB answerability, report usefulness, probability adaptation, safety, clarity, non-redundancy)
- Reported in the paper as percentages (score × 20)
- Implementation: `query_agent_eval/src/query_agent_eval/judge.py`

---

## X.5 Report Agent — answer system prompt

Source: `report_agent_eval/src/report_agent_eval/answer.py` (`ANSWER_SYSTEM`).

```
You are the evidence-grounded answering component of a non-clinical psychological screening report system. You answer a single planning question for an expert audience (counselors, student-support staff, researchers) using ONLY the retrieved evidence provided.

Rules:
1. SCREENING, NOT DIAGNOSIS: never diagnose, never prescribe treatment or medication.
2. GROUNDED: base every factual claim on the provided context passages; do not invent facts or citations. If the context is insufficient, say so explicitly.
3. CONTRIBUTORS, NOT CAUSES: describe model/XAI factors as statistical contributors, never proven causes.
4. CONCISE PROSE: 3-6 sentences of direct, professional prose that answers the question. No JSON, no headings, no bullet lists unless essential.
5. SAFETY: for elevated risk, note referral to qualified support where relevant.
```

**User message template** (`build_user_prompt`)

```
Question:
{planned question text}

Retrieved evidence:
[Context 1]
{passage 1}

[Context 2]
{passage 2}
...

Write the grounded answer now.
```

If no passages are available, the evidence block is `(No retrieved evidence available.)`.

Reference answers (`gpt-oss:20b`) and all candidate answers use this **identical** wrapper on the **same frozen contexts**.

---

## X.6 Report Agent evaluation design (for reproducibility)

1. Plan source (frozen): `gemma4` plans from `query_agent_eval/outputs/plans`  
2. Cases scored in the reported RAGAS table: `hms_low_01`, `hms_borderline_01`, `hms_high_01`  
3. Retrieval: `condition=ollama_context`, `mode=hybrid`, `top_k=5`  
4. Candidates: `gemma4`, `llama3`, `qwen2.5`, `phi4`, `mistral`  
5. RAGAS metrics: faithfulness, answer_relevancy, answer_correctness, semantic_similarity  
6. Ground truth for correctness/similarity: `gpt-oss:20b` reference answers  
7. Shared-model bias: same model family used as reference and RAGAS judge — disclosed in Limitations  

---

## X.7 Sample-size checklist (do not alter without re-running)

| Quantity | Value |
|---|---|
| Query Planning cases | 50 |
| Query Planning generators | 5 |
| Query Planning model–case runs | 250 |
| Query Planning questions generated | 1,575 |
| Report Agent cases (RAGAS table) | 3 |
| Report Agent questions scored per model | 18 |
| Contextual retrieval collection queries | 63 |
| Contextual retrieval manual challenge queries | 7 |
| End-to-end S0–S3 ablation cases | 3 |

---

## X.8 Uncertainty method

- Resampling unit: **case** (Query Planning); **question** (Report RAGAS)  
- Bootstrap replicates: **B = 2000**  
- Seed: **42**  
- Reported: mean, SD, 95% percentile CI  
- Script: `paper_outputs/bootstrap_rag_agent_uncertainty.py`  
- Outputs: `paper_outputs/query_agent_bootstrap_ci.csv`, `paper_outputs/report_agent_ragas_bootstrap_ci.csv`
