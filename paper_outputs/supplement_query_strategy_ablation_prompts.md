# Supplementary Appendix — Query-Strategy Ablation Prompts & Templates (v2)

This appendix documents **every prompt and query template** used in the strengthened end-to-end query-strategy ablation (v2). Prompt bodies are copied **verbatim** from the evaluation codebase. For the broader Query Planning / Report Agent evaluation stack (50-case plans; 3-case Report RAGAS), see also [`supplement_rag_agent_prompts.md`](supplement_rag_agent_prompts.md). For the consolidated planner / report / judge prompt appendix with decoding parameters and JSON schemas, see [`appendix_prompts_planner_report_judge.md`](appendix_prompts_planner_report_judge.md) (Supplementary Appendix A).

**Manuscript placement.** Put a short Methods pointer in the main text; put this full appendix in the Supplementary Material. Do not paste full prompts into Results.

---

## Y.0 Scope: what is ablated vs held fixed

| Component | Role in ablation |
|---|---|
| Case/XAI card (probability, threshold, risk label, TreeSHAP risk/protective factors) | **Shared** — identical for all arms |
| Query formulation (Direct / Template / LLM planning) | **Ablated** — only difference across arms |
| Knowledge base + hybrid Ollama-context retriever (`nomic-embed-text`, K=5) | **Held fixed** |
| Passage pooling to final evidence budget B=5 | **Held fixed** |
| Independent eval questions E1–E6 | **Held fixed** (not taken from the planner) |
| Report Agent (`gemma4`, temperature 0.1, same answer prompt) | **Held fixed** |
| RAGAS faithfulness + answer_relevancy (`gpt-oss:20b`, temperature 0; embeddings `nomic-embed-text`) | **Held fixed** |

Runner: `paper_outputs/run_query_strategy_ablation_v2.py`  
Results summary: `paper_outputs/end_to_end_query_strategy_ablation_v2_for_paper.md`

---

## Y.1 Generation and scoring parameters

| Parameter | Query Planning (LLM arm) | Direct / Template queries | Report Agent answers | RAGAS scoring |
|---|---|---|---|---|
| Model | `gemma4` (frozen plans on disk) | Deterministic string templates (no LLM) | `gemma4` | `gpt-oss:20b` (ChatOllama) |
| Temperature | **0.1** (when plans were generated) | N/A | **0.1** | **0.0** |
| Embeddings | — | — | — | `nomic-embed-text` |
| Retrieval | — | Hybrid Ollama-context, **K = 5** per query | Uses pooled B=5 contexts | Uses answer + pooled contexts |
| Final evidence budget | — | Dedup + rank → **B = 5** | B = 5 | — |
| Cases | — | **20** stratified (7 high / 6 borderline / 7 low), seed **42** | 20 × 3 arms × 6 questions = **360** | 360 question–arm rows |
| Bootstrap | — | Case-level paired, B=2000, seed=42 | Same | Same |

**Judge note for manuscript consistency.** The ablation protocol reports RAGAS with **`gpt-oss:20b`**, matching the Report Agent reference/RAGAS stack elsewhere in the manuscript. If a local `ragas_scores.csv` was produced under a different judge (e.g. an interim `llama3` pass), re-run `python paper_outputs/run_query_strategy_ablation_v2.py score` with `gpt-oss:20b` before citing Table X as gpt-oss-judged.

---

## Y.2 Query Planning Agent — system + user (LLM planning arm)

**Source:** `query_agent_eval/src/query_agent_eval/prompt.py` (`QUERY_PLANNING_SYSTEM`, `build_messages`).

**Use in ablation:** Frozen plans are loaded from `query_agent_eval/outputs/plans/gemma4/{case_id}.json`. The first six `retrieval_query` fields populate the LLM planning arm (truncate/pad to exactly 6). Plans are **not** regenerated during the ablation scoring run.

At runtime, the trailing `{structured_case}` placeholder is removed from the system message; the structured case is sent as the **user** message.

### System prompt (verbatim)

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

### User message template (verbatim)

```
Structured case:
{JSON serialization of the de-identified profile, ML probability, threshold,
 risk label, TreeSHAP contributors, protective factors, calibration metadata}
```

---

## Y.3 Arm A — Direct retrieval query template (no LLM)

**Source:** `arm_direct_queries()` in `paper_outputs/run_query_strategy_ablation_v2.py`.  
**Role:** system / user N/A — a single deterministic retrieval string.  
**Budget:** 1 query (`D1`).

### Construction (verbatim pattern)

```
college student {risk_label_with_underscores_as_spaces} screening probability {prediction_probability} XAI contributors: {top5_display_names_comma_separated}; protective factors; screening communication
```

Where `top5_display_names` come from the first five TreeSHAP risk contributors (`display_name` or `feature`).

---

## Y.4 Arm B — Template retrieval queries T1–T6 (no LLM)

**Source:** `arm_template_queries()` in `paper_outputs/run_query_strategy_ablation_v2.py`.  
**Budget:** exactly 6 queries. Placeholders filled from the same Case/XAI card (`xs[0..2]` = top-3 risk contributors; `prot_name` = top protective factor display name).

| ID | Retrieval query template (verbatim) |
|---|---|
| T1 | `interpreting college student depression anxiety screening probability threshold non-diagnostic guidelines` |
| T2 | `{display(xs[0])} college student depression anxiety screening evidence` |
| T3 | `{display(xs[1])} college student mental health screening evidence` |
| T4 | `{display(xs[2])} student depression anxiety screening contributor evidence` |
| T5 | `{prot_name} protective factors college student mental health screening` |
| T6 | `non-diagnostic screening communication limitations referral college counseling support` |

---

## Y.5 Independent evaluation questions E1–E6 (not planner text)

**Source:** `independent_eval_questions()` in `paper_outputs/run_query_strategy_ablation_v2.py`.  
These questions are answered by the Report Agent under every arm. They are **deterministic Case/XAI fill-ins** and are **independent of** Gemma plan question text (avoids biasing evaluation toward the LLM planner).

| ID | Type | Question template (verbatim) |
|---|---|---|
| E1 | screening_interpretation | `How should an expert interpret a non-diagnostic college-student depression/anxiety screening probability of {prob} relative to threshold {thr} (risk label: {risk}; near_threshold={cal.near_threshold})?` |
| E2 | xai_contributor_1 | `What evidence is relevant to understanding how {display(xs[0])} ({xs[0].feature}) can contribute to an elevated screening-model prediction in college students, without treating it as a causal diagnosis?` |
| E3 | xai_contributor_2 | `What evidence is relevant to understanding how {display(xs[1])} ({xs[1].feature}) relates to student depression/anxiety screening risk as a statistical contributor rather than a cause?` |
| E4 | xai_contributor_3 | `What evidence is relevant to understanding how {display(xs[2])} ({xs[2].feature}) may appear among model contributors for student mental-health screening support?` |
| E5 | protective_factors | `What evidence discusses protective or lower-risk factors relevant to {prot_names} in college student mental-health screening contexts?` |
| E6 | limitations_safety_support | `What evidence-based guidance exists for communicating non-diagnostic screening results, stating limitations, and suggesting referral/support pathways for college students without providing treatment prescriptions?` |

---

## Y.6 Report Agent — answer system + user prompts

**Source:** `report_agent_eval/src/report_agent_eval/answer.py` (`ANSWER_SYSTEM`, `build_user_prompt`).  
**Ablation call:** imported by `run_query_strategy_ablation_v2.py`; model `gemma4`, temperature **0.1**.

### System prompt (verbatim)

```
You are the evidence-grounded answering component of a non-clinical psychological screening report system. You answer a single planning question for an expert audience (counselors, student-support staff, researchers) using ONLY the retrieved evidence provided.

Rules:
1. SCREENING, NOT DIAGNOSIS: never diagnose, never prescribe treatment or medication.
2. GROUNDED: base every factual claim on the provided context passages; do not invent facts or citations. If the context is insufficient, say so explicitly.
3. CONTRIBUTORS, NOT CAUSES: describe model/XAI factors as statistical contributors, never proven causes.
4. CONCISE PROSE: 3-6 sentences of direct, professional prose that answers the question. No JSON, no headings, no bullet lists unless essential.
5. SAFETY: for elevated risk, note referral to qualified support where relevant.
```

### User message template (verbatim)

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

If no passages are available, the evidence block is `(No retrieved evidence available.)`.

---

## Y.7 RAGAS scoring disclosure (`gpt-oss:20b`)

**Source:** `score_ragas_row()` in `paper_outputs/run_query_strategy_ablation_v2.py`.

- Metrics: RAGAS **faithfulness** and **answer_relevancy** only (library defaults).  
- LLM: `ChatOllama(model="gpt-oss:20b", base_url="http://localhost:11434", temperature=0)`.  
- Embeddings: `OllamaEmbeddings(model="nomic-embed-text", ...)`.  
- **No custom RAGAS metric prompts are defined in this repository**; the RAGAS package supplies its default metric instruction templates.  
- Inputs per row: `question` (E1–E6), `answer` (Report Agent), `contexts` (pooled B=5 passage texts).

---

## Y.8 Out of scope for this ablation

The Query Planning Agent **Level-2 LLM judge** prompt (`query_agent_eval/.../judge.py`, historically `gpt-oss:20b`) is used in the separate 50-case plan-quality evaluation. It is **not** invoked during the query-strategy ablation run. See [`supplement_rag_agent_prompts.md`](supplement_rag_agent_prompts.md) Section X.4.

---

## Y.9 Worked example (`hms_high_02`)

Values taken from the ablation outputs (`queries_by_arm.csv`, `answers_index.csv`). Passage bodies omitted.

### Direct query (D1)

```
college student elevated screening risk screening probability 0.9743 XAI contributors: Academic impairment, Flourishing, Current financial situation, Food insecurity worry, Loneliness; protective factors; screening communication
```

### Template query (T2)

```
Academic impairment college student depression anxiety screening evidence
```

### Independent eval question (E1)

```
How should an expert interpret a non-diagnostic college-student depression/anxiety screening probability of 0.9743 relative to threshold 0.5 (risk label: elevated screening risk; near_threshold=False)?
```

### Report Agent user-message skeleton (E1 + pooled contexts)

```
Question:
How should an expert interpret a non-diagnostic college-student depression/anxiety screening probability of 0.9743 relative to threshold 0.5 (risk label: elevated screening risk; near_threshold=False)?

Retrieved evidence:
[Context 1]
{pooled passage 1}

[Context 2]
{pooled passage 2}

[Context 3]
{pooled passage 3}

[Context 4]
{pooled passage 4}

[Context 5]
{pooled passage 5}

Write the grounded answer now.
```

(System message = Section Y.6; model = `gemma4`, temperature 0.1.)

---

## Y.10 How to report this in the manuscript

### Main text (Methods) — keep short

Name: models, temperatures, K/B, that only query formulation was ablated, and that **verbatim prompts/templates are in Supplementary Appendix Y**. Do **not** reproduce full prompt text in Results or Discussion.

### Supplementary Material

Include this file (or its journal-formatted equivalent) in full. Cross-reference figure captions to Appendix Y for prompt text.

### Figure caption (example)

> **Figure X. End-to-end query-strategy ablation.** Identical Case/XAI inputs feed three query strategies (direct; six templates; six LLM-plan queries). Downstream retrieval (K=5), evidence pooling (B=5), Report Agent, and independent evaluation questions (E1–E6) are held fixed. Verbatim prompts and templates are provided in Supplementary Appendix Y.

### Paste-ready Methods paragraph (prompts / transparency)

**Prompt transparency (query-strategy ablation).** Exact system and user prompts for the Query Planning Agent and Report Agent, deterministic Direct and Template retrieval-query constructions, independent evaluation-question templates (E1–E6), generation/scoring parameters (Query Planning and Report Agent temperature 0.1; RAGAS judge `gpt-oss:20b` at temperature 0; embeddings `nomic-embed-text`; hybrid retrieval K=5; pooled evidence budget B=5), and a worked example for case `hms_high_02` are provided verbatim in Supplementary Appendix Y. No custom RAGAS metric prompts were authored; library defaults were used.

### Paste-ready reviewer response (prompt transparency)

Thank you for requesting prompt transparency. We now provide the complete verbatim prompt and template set for the end-to-end query-strategy ablation in Supplementary Appendix Y, including Query Planning and Report Agent prompts, Direct/Template retrieval-query constructions, independent evaluation questions E1–E6, RAGAS judge configuration (`gpt-oss:20b`, temperature 0), and a worked example. Main-text Methods summarize models and parameters and point to the appendix for full wording.

### Author checklist (before submission)

| Item | Report in Methods? | Report in Supplement? |
|---|---|---|
| System vs user message | Name both | Full verbatim text |
| Which arm uses the prompt | Yes (Direct / Template / LLM / Report / RAGAS) | Yes |
| Temperature | Yes | Yes (table) |
| Exact model tag | Yes (`gemma4`, `gpt-oss:20b`, `nomic-embed-text`) | Yes |
| Source file path | Optional | Yes |
| Custom vs library RAGAS prompts | Yes (library defaults) | Yes |
| Judge used for cited Table X | Must match the scored CSV | Disclose any interim alternate judge |

---

## Y.11 Source file index

| Artifact | Path |
|---|---|
| Ablation runner | `paper_outputs/run_query_strategy_ablation_v2.py` |
| Query Planning prompts | `query_agent_eval/src/query_agent_eval/prompt.py` |
| Report Agent prompts | `report_agent_eval/src/report_agent_eval/answer.py` |
| Frozen plans (`gemma4`) | `query_agent_eval/outputs/plans/gemma4/` |
| Ablation queries / answers | `paper_outputs/query_strategy_ablation_v2/` |
| Related agent-eval prompt appendix | `paper_outputs/supplement_rag_agent_prompts.md` |
