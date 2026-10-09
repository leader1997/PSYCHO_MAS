# Strengthening roadmap — implementation status

This document tracks the 20-point reviewer roadmap against what is **implemented**, **partial**, or **planned**.

## Implemented (run now)

| Priority | Task | Script / artifact |
|----------|------|-------------------|
| 1 | Freeze Baseline v1 | `run_freeze_baseline.py` → `outputs/versions/v1_fast/` |
| 2 | Feature ablation E1–E6 | `run_feature_ablation.py` → `outputs/strengthened/table_a_*.csv` |
| 3 | Bootstrap 95% CI | `run_validation_suite.py` → `table_b_bootstrap_ci.csv` |
| 4 | Calibration ECE + curve | `run_validation_suite.py` → `table_c_calibration.csv`, `figures/calibration_curve_xgboost.png` |
| 5 | XAI deletion controls | `run_faithfulness_controls.py` → `table_e_faithfulness_controls.csv` |
| 6 | Feature dictionary | `reproducibility/FEATURE_DICTIONARY.md` |
| 9 | Weighting scheme comparison | `run_weighting_comparison.py` |
| 6b | Threshold screening metrics | `threshold_screening_metrics.csv` |
| 7 | Fairness subgroups (initial) | `table_d_fairness_subgroups.csv` |
| 17 | Reproducibility scaffold | `reproducibility/`, `pipeline_common.py`, `run_strengthened_pipeline.py` |

### One-command analysis (priorities 1–6, no full retrain)

```powershell
cd xai_rag_depression
python run_strengthened_pipeline.py
```

### Full-quality v2 (hours)

```powershell
$env:XAI_FAST="0"
python run_strengthened_pipeline.py --full-pipeline --full-quality
```

---

## Planned (not yet implemented)

| § | Task | Notes |
|---|------|-------|
| 4 | Repeated CV / institution-held-out | Needs `inst_id` or similar in CSV |
| 5 | Platt / isotonic recalibration | After calibration diagnostics |
| 11 | Hybrid BM25 + dense + reranker | RAG v4 |
| 12 | 30–100 LLM eval cases | RAG v5 |
| 13 | Multi-LLM comparison | Extend `run_nb4.py` |
| 14 | Context routing by case type | New routing module |
| 15 | Safety guardrails / forbidden phrases | Post-generation filter |
| 16 | Causal sensitivity (adjusted LR, PSM) | Separate analysis script |
| 18 | Tables F–G (retrieval + 100-case LLM) | After §11–12 |

---

## Paper tables generated

| Table | File |
|-------|------|
| A — Feature ablation | `outputs/strengthened/table_a_feature_ablation_xgb.csv` |
| B — Bootstrap CI | `outputs/strengthened/table_b_bootstrap_ci.csv` |
| C — Calibration | `outputs/strengthened/table_c_calibration.csv` |
| D — Fairness | `outputs/strengthened/table_d_fairness_subgroups.csv` |
| E — Faithfulness controls | `outputs/strengthened/table_e_faithfulness_controls.csv` |
| Version roadmap | `outputs/versions/version_roadmap.csv` |

---

## Recommended contribution framing (post-strengthening)

> Explainability-augmented RAG for survey-based depression/anxiety **screening research** among college students. Nine models show similar AUC (~0.85); explanations agree on academic impairment, loneliness, flourishing, and financial stress. **Feature-ablation** quantifies dependence on outcome-proximal variables (`aca_impa`). RAG ablation shows XAI improves narrative relevancy; retrieval precision and hallucination control remain deployment challenges.
