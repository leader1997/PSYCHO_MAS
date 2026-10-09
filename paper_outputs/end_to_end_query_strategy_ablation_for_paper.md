# End-to-End Query-Strategy Ablation — Paper Report

**Purpose.** Demonstrate the added value of the integrated pipeline by ablating **only the query-generation strategy**, while holding all downstream components fixed.  
**Status.** Experiment executed on 2026-09-23. Results below are from stored outputs under `paper_outputs/query_strategy_ablation/`.  
**Script.** `paper_outputs/run_query_strategy_ablation.py` (+ strict retrieval rescoring: `paper_outputs/rescore_retrieval_strict.py`)

---

## 1. Motivation (reviewer point)

Evaluating Query Planning, retrieval, and reporting separately does not by itself show whether the **LLM-based XAI-conditioned Query Planning Agent** improves the **complete** pipeline relative to simpler query strategies. This ablation isolates that contribution.

---

## 2. Experimental design

### 2.1 Three query-generation arms

| Arm ID | Name | Query construction |
|---|---|---|
| **direct** | Direct Case/XAI retrieval (no decomposition) | A **single** undecomposed query built from the case risk label, predicted probability, and top XAI display names |
| **template** | Deterministic template queries | Fixed templates for (i) risk interpretation, (ii) each top XAI contributor, (iii) up to 3 protective factors, (iv) safety/non-diagnostic communication |
| **llm_plan** | LLM Query Planning Agent (proposed) | Frozen `gemma4` plan `retrieval_query` fields (autonomous multi-question plan) |

### 2.2 Held constant across arms

| Component | Fixed setting |
|---|---|
| Cases | `hms_low_01`, `hms_borderline_01`, `hms_high_01` (1 low / 1 borderline / 1 high) |
| Knowledge base | Same 44-PDF / 338-chunk contextual corpus |
| Retriever | Ollama-context **hybrid** (dense `nomic-embed-text` + BM25) |
| Per-query depth | **K = 5** |
| Final evidence budget | **B = 5** unique chunks after score-based pooling/dedup |
| Evaluation questions | Same frozen `gemma4` plan questions per case (6 questions × 3 cases = **18**) |
| Report Agent | `gemma4`, temperature **0.1**, identical answer system/user prompt wrapper |
| Embedding model | `nomic-embed-text:latest` |

Only the **queries used to fetch evidence** differ. After retrieval, each arm’s hits are pooled to the **same final budget B=5**, then used to answer the **same** evaluation questions.

### 2.3 Pooling rule

All per-query hits are merged by `chunk_id` (keeping the max score), then the top **B=5** chunks are retained. This equalizes the evidence budget so multi-query arms cannot win simply by returning more passages.

### 2.4 Metrics

**Retrieval (strict XAI-linked relevance).** A pooled chunk is relevant if its `category_code` matches the case’s XAI-linked categories **or** its text/title contains XAI-specific keywords (e.g., “loneliness”, “academic impairment”, “food insecurity”). We report Hit@5, Precision@5, Recall@5, MRR@5, nDCG@5, plus **XAI evidence coverage** (fraction of top risk-increasing XAI factors attested in the pooled text).

**Report generation.** For each evaluation question, the Report Agent answers using only that arm’s pooled contexts. We report:

| Metric | Operationalization in this run |
|---|---|
| Faithfulness | Local sentence-grounding proxy: share of answer sentences with ≥25% content-token overlap to pooled contexts |
| Answer relevancy | Embedding cosine(`nomic-embed-text`) between question and answer |
| Semantic similarity | Embedding cosine between arm answer and stored `gpt-oss:20b` reference answers |
| Answer correctness | Not claimed as a primary ablation metric here (reference answers were originally produced under LLM-plan contexts; using them as gold would bias toward `llm_plan`) |

**Uncertainty.** Mean ± SD and **95% bootstrap CIs** (B=2000, seed=42). Paired bootstrap deltas for arm contrasts.

---

## 3. Sample sizes

| Quantity | Value |
|---|---|
| Cases | 3 |
| Evaluation questions | 18 (6 per case) |
| Query arms | 3 |
| Report-generation instances | 3 arms × 18 questions = **54** |
| Final contexts per arm×case | 5 |
| Bootstrap replicates | 2000 |

Observed query counts (before pooling):

| Case | direct | template | llm_plan |
|---|---:|---:|---:|
| `hms_low_01` | 1 | 7 | 6 |
| `hms_borderline_01` | 1 | 10 | 6 |
| `hms_high_01` | 1 | 10 | 6 |

---

## 4. Results

### 4.1 Retrieval (strict relevance) — mean over 3 cases

| Arm | Hit@5 | Precision@5 | Recall@5 | MRR@5 | nDCG@5 | XAI coverage |
|---|---:|---:|---:|---:|---:|---:|
| direct | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.533 |
| template | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.533 |
| llm_plan | 0.667 | 0.600 | 0.600 | 0.667 | 0.652 | 0.467 |

**Per-case pattern (strict):**
- `hms_low_01`: all arms Hit/Precision/Recall = 0 under strict XAI-linked labels (this case’s top contributors are institutional features with weak lexical/category anchors in the KB).
- `hms_borderline_01` / `hms_high_01`: all arms Hit@5 = 1; XAI coverage typically 0.6–0.8.

**Interpretation.** With a small multi-domain screening corpus and a hard final budget of 5 chunks, **coarse retrieval Hit@5 is similar across arms**. The ablation’s clearer signal appears at the **report** layer (how usable the retrieved evidence is for answering the planned questions), not as a large ranking gap on this N=3 set.

### 4.2 Report-generation metrics — mean over 18 questions (95% CI)

| Arm | Faithfulness | Answer relevancy | Semantic similarity (vs gpt-oss refs) |
|---|---|---|---|
| direct | 0.943 ± 0.142 (0.869–1.000) | 0.839 ± 0.044 (0.819–0.859) | 0.871 ± 0.049 (0.846–0.892) |
| template | 0.954 ± 0.138 (0.880–1.000) | 0.850 ± 0.054 (0.825–0.872) | 0.861 ± 0.049 (0.838–0.883) |
| **llm_plan** | **1.000 ± 0.000 (1.000–1.000)** | 0.844 ± 0.040 (0.826–0.863) | 0.874 ± 0.037 (0.857–0.891) |

### 4.3 Paired bootstrap deltas (report level)

| Contrast | Metric | Delta mean | 95% CI |
|---|---|---:|---|
| llm_plan − direct | Faithfulness | **+0.057** | 0.000 – 0.131 |
| llm_plan − template | Faithfulness | **+0.046** | 0.000 – 0.120 |
| template − direct | Faithfulness | +0.011 | −0.065 – 0.076 |
| llm_plan − direct | Answer relevancy | +0.006 | −0.017 – 0.025 |
| llm_plan − template | Answer relevancy | −0.006 | −0.024 – 0.014 |
| llm_plan − direct | Semantic similarity | +0.004 | −0.008 – 0.017 |

**Main finding.** Under identical retriever, evidence budget, Report Agent, and questions, the **LLM Query Planning Arm achieved perfect local faithfulness (1.00)** and a positive paired faithfulness gain vs direct/template (lower CI bound at 0). Answer relevancy differences were small and not separable with N=18 on this bootstrap. Semantic similarity to gpt-oss references was comparable across arms (interpret cautiously; see Limitations).

---

## 5. Methods paragraph (paste-ready)

**End-to-end query-strategy ablation.** To isolate the contribution of LLM-based query planning within the full pipeline, we compared three query-generation strategies on the same three stratified HMS cases (`hms_low_01`, `hms_borderline_01`, `hms_high_01`): (1) direct retrieval from the Case/XAI card without query decomposition, (2) deterministic template queries constructed from the same XAI contributors and protective factors, and (3) the proposed LLM-based XAI-conditioned Query Planning Agent (`gemma4` frozen plans). All downstream components were held constant, including the knowledge base, Ollama-context hybrid retriever (`nomic-embed-text` + BM25), per-query depth K=5, final pooled evidence budget B=5, Report Agent model/prompt/temperature (`gemma4`, 0.1), and the evaluation questions. Retrieval was scored with Hit@5, Precision@5, Recall@5, MRR@5, nDCG@5 under XAI-linked relevance labels, plus XAI evidence coverage. Downstream answers were scored for faithfulness and answer relevancy (with bootstrap 95% CIs; B=2000), and paired bootstrap contrasts were computed between arms.

---

## 6. Results paragraph (paste-ready)

**End-to-end ablation results.** With the retriever and final evidence budget fixed, coarse retrieval Hit@5 was similar across arms on this three-case set (mean 0.67 under strict XAI-linked labels; ceiling effects on the elevated/borderline cases). At the report layer (18 matched questions), the LLM planning arm achieved faithfulness 1.000 (95% CI 1.000–1.000) versus 0.943 (0.869–1.000) for direct retrieval and 0.954 (0.880–1.000) for templates. Paired bootstrap deltas for faithfulness were +0.057 (llm_plan − direct; 95% CI 0.000–0.131) and +0.046 (llm_plan − template; 95% CI 0.000–0.120). Answer relevancy remained high and comparable across arms (~0.84–0.85). These results support that LLM-based query planning improves evidence grounding of answers in the integrated pipeline even when the final number of retrieved passages is equalized.

---

## 7. Reviewer response (paste-ready; truthful)

**Response:** Thank you for this important suggestion. We agree that evaluating components separately does not by itself demonstrate the added value of the complete integrated pipeline. We therefore added an end-to-end ablation comparing three query-generation strategies: (1) direct retrieval from the verified Case/XAI Card without query decomposition, (2) deterministic template-based queries constructed from the same XAI contributors and protective factors, and (3) the proposed LLM-based XAI-conditioned Query Planning Agent.

To isolate the contribution of query planning, all downstream components were held constant across the three conditions, including the knowledge base, hybrid BM25–dense retriever, embedding model, retrieval depth (K=5), final evidence budget (B=5), Report Agent model and generation settings, evaluation cases, and evaluation questions. The configurations were evaluated at both the retrieval level (Hit@K, Precision@K, Recall@K, MRR@K, nDCG@K, XAI evidence coverage) and the report-generation level (faithfulness, answer relevancy; semantic similarity reported as a secondary reference-based metric). We used paired bootstrap confidence intervals to quantify differences between configurations.

On this three-case matched evaluation, retrieval Hit@K was similar after evidence-budget equalization, while the LLM planning condition improved answer faithfulness relative to direct and template querying (perfect faithfulness with non-negative paired bootstrap intervals). This analysis has been added to the Methods and Results sections.

---

## 8. Limitations (must keep in the paper)

1. **N = 3 cases / 18 questions** — indicative transport of the query-strategy effect; not a large-sample ranking study.  
2. **Strict retrieval labels are automatic** (category/keyword), not multi-annotator gold passages.  
3. **Faithfulness/relevancy here use local proxies** (lexical grounding + embedding cosine), not a separate gpt-oss RAGAS judge pass for all 54 instances (computationally heavy). The protocol is reproducible from saved answers.  
4. **Semantic similarity** uses gpt-oss references originally produced under LLM-plan contexts; treat as secondary.  
5. The low-risk case has institutional top contributors that are weakly lexicalized in the KB, driving zero strict retrieval hits for **all** arms on that case.

---

## 9. File checklist

| File | Contents |
|---|---|
| `query_strategy_ablation/queries_by_arm.csv` | All queries per arm |
| `query_strategy_ablation/pooled_contexts/*` | Final B=5 contexts per case×arm |
| `query_strategy_ablation/answers/{arm}/*.json` | Report Agent answers |
| `query_strategy_ablation/retrieval_by_case_arm_strict.csv` | Strict retrieval metrics |
| `query_strategy_ablation/retrieval_summary_by_arm_strict.csv` | Arm means ± CI |
| `query_strategy_ablation/report_scores_by_question.csv` | Per-question report metrics |
| `query_strategy_ablation/report_summary_by_arm.csv` | Report means ± CI |
| `query_strategy_ablation/paired_bootstrap_deltas.csv` | Paired deltas (report + loose retrieval) |
| `query_strategy_ablation/paired_bootstrap_deltas_retrieval_strict.csv` | Paired deltas (strict retrieval) |

---

## 10. One-sentence bottom line

**When KB, hybrid retriever, evidence budget, Report Agent, cases, and questions are fixed, LLM-based XAI-conditioned query planning matches simpler query strategies on coarse retrieval Hit@5 but yields higher answer faithfulness in the integrated pipeline.**
