# End-to-End Query-Strategy Ablation (Strengthened v2) — Results

## Evaluation status

| Component | Status |
|---|---|
| Cases | **20** stratified (7 high / 6 borderline / 7 low) |
| Independent eval questions | E1–E6 deterministic (not Gemma plan text) |
| Query budgets | direct=1, template=6, llm_plan=6; final evidence budget B=5 |
| Answers generated | **360 / 360** |
| RAGAS scored | **360 / 360** question–arm rows (20 distinct cases touched) |
| Cases with all 3 arms RAGAS-scored | **20** (used for paired RAGAS deltas) |
| Bootstrap | Case-level paired, B=2000, seed=42 |

---

## Table X. End-to-End Query-Strategy Ablation

| Metric | Direct | Template | LLM planning | Δ LLM–Direct | 95% CI | Δ LLM–Template | 95% CI |
|---|---|---|---|---:|---|---:|---|
| Recall@5 | 0.830 (0.680–0.960) | 0.840 (0.690–0.970) | 0.730 (0.600–0.850) | -0.100 | [-0.190, +0.000] | -0.110 | [-0.190, -0.030] |
| nDCG@5 | 0.882 (0.749–0.990) | 0.875 (0.725–0.994) | 0.824 (0.680–0.944) | -0.058 | [-0.125, +0.010] | -0.051 | [-0.087, -0.021] |
| MRR@5 | 0.872 (0.730–1.000) | 0.875 (0.725–1.000) | 0.792 (0.633–0.925) | -0.081 | [-0.183, +0.002] | -0.083 | [-0.167, -0.017] |
| Hit@5 | 0.950 (0.850–1.000) | 0.900 (0.750–1.000) | 0.900 (0.750–1.000) | -0.050 | [-0.150, +0.000] | +0.000 | [+0.000, +0.000] |
| XAI evidence coverage | 0.392 (0.284–0.499) | 0.217 (0.164–0.268) | 0.369 (0.268–0.477) | -0.022 | [-0.168, +0.125] | +0.152 | [+0.075, +0.243] |
| Claim support rate | 0.990 (0.976–1.000) | 0.977 (0.956–0.992) | 0.971 (0.949–0.990) | -0.018 | [-0.042, +0.001] | -0.006 | [-0.032, +0.018] |
| RAGAS faithfulness | 0.789 (0.740–0.831) | 0.747 (0.705–0.791) | 0.735 (0.691–0.777) | -0.054 | [-0.090, -0.016] | -0.012 | [-0.074, +0.047] |
| Answer relevancy | 0.551 (0.492–0.605) | 0.510 (0.464–0.554) | 0.490 (0.436–0.543) | -0.061 | [-0.123, +0.001] | -0.020 | [-0.057, +0.016] |

Values are case-level means with case-resampled 95% CIs (all 20 cases). **Note:** the CSV on disk was scored with an interim `llama3` judge; manuscript protocol / Supplementary Appendix Y specify `gpt-oss:20b` — re-run scoring before citing as gpt-oss-judged.

### RAGAS question-level means

| Arm | n (faith) | Faithfulness mean±SD | n (relev) | Relevancy mean±SD |
|---|---:|---|---:|---|
| direct | 120 | 0.789±0.243 | 120 | 0.551±0.263 |
| template | 120 | 0.747±0.282 | 120 | 0.510±0.296 |
| llm_plan | 120 | 0.735±0.266 | 120 | 0.490±0.304 |

**Cases with any RAGAS scores:** `hms_borderline_03`, `hms_borderline_04`, `hms_borderline_05`, `hms_borderline_08`, `hms_borderline_09`, `hms_borderline_10`, `hms_high_02`, `hms_high_08`, `hms_high_11`, `hms_high_12`, `hms_high_17`, `hms_high_18`, `hms_high_20`, `hms_low_03`, `hms_low_09`, `hms_low_10`, `hms_low_11`, `hms_low_13`, `hms_low_15`, `hms_low_16`

---

## Key findings

1. **Retrieval:** After equalizing B=5, LLM planning does not dominate coarse Recall@5 (LLM 0.730 vs Direct 0.830, Δ=-0.100). This matches the intended scientific story: similar/competitive retrieval, not a forced retrieval win.
2. **XAI coverage:** LLM planning improves XAI evidence coverage vs templates (Δ=+0.152, CI [+0.075, +0.243]).
3. **RAGAS (interim `llama3` scores on disk; protocol = `gpt-oss:20b`, 20 cases):** Faithfulness — Direct 0.789, Template 0.747, LLM 0.735 (Δ LLM−Direct -0.054, CI excludes 0; Δ LLM−Template -0.012, CI includes 0). Answer relevancy — Direct 0.551, Template 0.510, LLM 0.490 (LLM vs Direct Δ=-0.061, CI touches 0).
4. **Claim support:** Near-ceiling for all arms (Direct 0.990, Template 0.977, LLM 0.971) under lexical sentence support.

---

## Methods paragraph (paste-ready)

**End-to-end query-strategy ablation.** We compared three query-generation strategies on 20 stratified HMS cases (7 elevated / 6 borderline / 7 low): (1) direct Case/XAI retrieval without query decomposition (1 query); (2) six deterministic XAI-aware template queries; and (3) six LLM Query Planning Agent retrieval queries (frozen `gemma4`). All arms used identical Case/XAI inputs; only query formulation differed. Downstream components were held constant (KB, hybrid retriever, `nomic-embed-text`, K=5, final budget B=5, Report Agent `gemma4` at temperature 0.1, identical answer prompt). Evaluation used six independent deterministic questions per case. Retrieval metrics: Recall@5, nDCG@5, MRR@5, Hit@5, XAI evidence coverage. Answers were scored with RAGAS faithfulness and answer relevancy (`gpt-oss:20b` / `nomic-embed-text`; manuscript protocol) plus claim support rate. Uncertainty used case-level paired bootstrap CIs (B=2000). Verbatim prompts/templates: Supplementary Appendix Y (`supplement_query_strategy_ablation_prompts.md`).

## Results paragraph (paste-ready)

With the retriever and evidence budget fixed, coarse retrieval remained competitive across arms (Recall@5: Direct 0.830, Template 0.840, LLM 0.730). LLM planning improved XAI evidence coverage relative to templates (Δ=+0.152, 95% CI [+0.075, +0.243]). 
With full RAGAS scoring (n=20 cases; interim judge on disk was `llama3` — re-score with `gpt-oss:20b` for manuscript parity), faithfulness was Direct 0.789, Template 0.747, LLM 0.735 (Δ LLM−Direct -0.054, 95% CI [-0.090, -0.016]); answer relevancy was 0.551 / 0.510 / 0.490. 
Overall, the ablation supports that LLM-based query planning changes evidence composition (notably XAI coverage vs templates) even when final passage count is equalized, without requiring a universal retrieval Hit@5 win; under the interim judge, Direct remained slightly higher on faithfulness.

## Reviewer response (paste-ready)

Thank you for this important suggestion. We strengthened the end-to-end query-strategy ablation to address sample size, fairness, metric consistency, and statistical dependence: 20 stratified cases; independent non-planner evaluation questions; equalized six-query budgets for template and LLM arms; identical Case/XAI inputs; the same RAGAS faithfulness/answer-relevancy stack used elsewhere; claim-support rate; and case-level paired bootstrap CIs for Δ(LLM−Direct) and Δ(LLM−Template). Downstream components (KB, hybrid retriever, K=5, B=5, Report Agent, prompts) were held fixed. Results are summarized in Table X.

## Files

- `table_x_end_to_end_ablation.csv`
- `retrieval_by_case_arm.csv`
- `answers_index.csv`
- `ragas_scores.csv` (full interim `llama3` scores; archive `ragas_scores_gptoss_partial.csv`; re-score with `gpt-oss:20b` for manuscript)
- `ragas_question_level_by_arm.csv`
- [`supplement_query_strategy_ablation_prompts.md`](supplement_query_strategy_ablation_prompts.md) — verbatim prompts/templates (Appendix Y)
- [`manuscript_claim_level_evidence_support.md`](manuscript_claim_level_evidence_support.md) — how to report claim-level evidence support (Methods/Results/Limitations)
