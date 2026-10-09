# Predictive-model evaluation — manuscript metrics pack

**Purpose.** Respond to the reviewer request for stronger predictive evaluation: confidence intervals, calibration, and screening measures (sensitivity, specificity, PPV, NPV), beyond aggregate scores alone.

**Primary reported model (Case/XAI Card / SHAP pipeline):** XGBoost on HMS `dep_or_anx` (student-level stratified holdout unless noted).  
**Sources:** `xai_rag_depression/outputs/` and `xai_rag_depression/outputs/strengthened/`.

---

## 1. What to include in the manuscript

| Block | Put in | Content |
|---|---|---|
| A. Discrimination + 95% CIs | Main Results table | AUC, PR-AUC, F1, Brier with bootstrap CIs (XGBoost) |
| B. Multi-model comparison | Main or Supplement | AUC / PR-AUC / recall / Brier across candidates |
| C. Calibration | Main short table + reliability curve if available | Brier, ECE, slope, intercept |
| D. Screening operating points | Main or Supplement | Sensitivity, specificity, PPV, NPV at thresholds 0.3–0.7 |
| E. Institution-held-out | Sensitivity / Supplement | Student-level vs leave-campus-out AUC (mean±SD) |

**Note on statistical comparisons.** Models are extremely close on AUC (~0.847–0.853). We report descriptive ΔAUC. Formal pairwise DeLong tests were not part of the saved strengthened suite; if required, they can be added from held-out probabilities.

---

## 2. Table A — XGBoost discrimination with 95% bootstrap CIs

Source: `strengthened/table_b_bootstrap_ci.csv`

| Metric | Mean | 95% CI |
|---|---:|---|
| AUC | 0.853 | [0.844, 0.862] |
| PR-AUC | 0.823 | [0.808, 0.836] |
| F1 (threshold 0.5) | 0.752 | [0.741, 0.764] |
| Brier score | 0.156 | [0.151, 0.161] |

---

## 3. Table B — Multi-model discrimination (student-level holdout)

Source: `outputs/model_performance.csv` (point estimates; threshold 0.5 for classification metrics)

| Model | Family | AUC | PR-AUC | Accuracy | Recall (Sens.) | F1 | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | ML | 0.853 | 0.823 | 0.773 | 0.769 | 0.753 | 0.156 |
| LightGBM | ML | 0.852 | 0.823 | 0.770 | 0.769 | 0.750 | 0.157 |
| CatBoost | ML | 0.852 | 0.825 | 0.772 | 0.772 | 0.752 | 0.157 |
| FT-Transformer | DL | 0.852 | 0.824 | 0.768 | 0.789 | 0.753 | 0.158 |
| TabTransformer | DL | 0.851 | 0.824 | 0.769 | 0.678 | 0.725 | 0.157 |
| DCNv2 | DL | 0.850 | 0.823 | 0.771 | 0.745 | 0.745 | 0.157 |
| Random Forest | ML | 0.849 | 0.820 | 0.769 | 0.759 | 0.747 | 0.158 |
| Extra Trees | ML | 0.848 | 0.819 | 0.768 | 0.764 | 0.747 | 0.158 |
| Logistic Regression | ML | 0.847 | 0.818 | 0.766 | 0.759 | 0.745 | 0.160 |

**Descriptive comparison (vs XGBoost AUC 0.853):** largest gap is Logistic Regression (ΔAUC ≈ −0.006). CatBoost / LightGBM differ by &lt;0.002 AUC.

---

## 4. Table C — Calibration (XGBoost)

Source: `strengthened/table_c_calibration.csv`

| Metric | Value |
|---|---:|
| Brier score | 0.156 |
| Expected calibration error (ECE, 10 bins) | 0.033 |
| Calibration slope | 0.202 |
| Calibration intercept | 0.473 |

**Interpretation for manuscript.** Discrimination is strong (AUC ≈ 0.85), but the calibration slope &lt; 1 indicates probability compression (predicted risks are not perfectly calibrated on an absolute scale). Brier/ECE remain moderate. For screening use, emphasize ranking / thresholded decisions and disclose that local recalibration may be needed before operational deployment.

---

## 5. Table D — Screening metrics (sensitivity, specificity, PPV, NPV)

### D1. All models at threshold 0.5 (manuscript table)

Prevalence on the evaluated test split: **≈ 0.449**.  
Sensitivity = recall from `model_performance.csv`; specificity from accuracy identity  
`acc = sens·p + spec·(1−p)`; PPV/NPV from Bayes formulas (matches XGBoost confusion-matrix metrics).

Source: `strengthened/screening_metrics_all_models_thr0.5.csv`

| Model | Sensitivity | Specificity | PPV | NPV |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.759 | 0.772 | 0.731 | 0.797 |
| Random Forest | 0.759 | 0.776 | 0.735 | 0.798 |
| Extra Trees | 0.764 | 0.771 | 0.731 | 0.800 |
| XGBoost | 0.769 | 0.776 | 0.737 | 0.805 |
| LightGBM | 0.769 | 0.771 | 0.733 | 0.804 |
| CatBoost | 0.772 | 0.772 | 0.734 | 0.806 |
| FT-Transformer | 0.789 | 0.750 | 0.720 | 0.813 |
| TabTransformer | 0.678 | 0.844 | 0.780 | 0.762 |
| DCNv2 | 0.745 | 0.792 | 0.745 | 0.792 |

### D2. XGBoost across thresholds

Source: `strengthened/threshold_screening_metrics.csv` + `threshold_screening_metrics_with_npv.csv`

| Threshold | Sensitivity | Specificity | PPV | NPV | F1 | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.904 | 0.569 | 0.631 | 0.879 | 0.743 | 0.431 | 0.096 |
| 0.40 | 0.848 | 0.676 | 0.681 | 0.845 | 0.755 | 0.324 | 0.152 |
| 0.50 | 0.769 | 0.776 | 0.737 | 0.805 | 0.753 | 0.224 | 0.231 |
| 0.60 | 0.670 | 0.852 | 0.787 | 0.760 | 0.724 | 0.148 | 0.330 |
| 0.70 | 0.546 | 0.917 | 0.844 | 0.712 | 0.663 | 0.083 | 0.454 |

**Default operating point:** threshold **0.5**.

**Screening trade-off.** Lower thresholds raise sensitivity (fewer missed elevated cases) and NPV, at the cost of specificity and PPV.

---

## 6. Table E — Institution-held-out transportability (extra)

Source: `strengthened/table_institution_vs_student_level.csv` / `institution_heldout_summary.csv`  
GroupKFold by `schoolnum` (K=5).

| Setting | Model | Student-level AUC | Held-out AUC (mean±SD) | Δ (student − held-out) |
|---|---|---:|---:|---:|
| E1 full | CatBoost | 0.852 | 0.855 ± 0.004 | −0.003 |
| E1 full | XGBoost | 0.853 | 0.856 ± 0.005 | −0.002 |
| E6 no institution covariates | CatBoost | 0.852 | 0.854 ± 0.004 | −0.002 |
| E6 no institution covariates | XGBoost | 0.853 | 0.855 ± 0.004 | −0.002 |

No AUC degradation under leave-campus-out within HMS.

---

## 7. Paste-ready Methods

**Predictive-model evaluation.** Beyond aggregate discrimination scores, we report (i) 95% bootstrap confidence intervals for AUC, PR-AUC, F1, and Brier score on the student-level holdout (XGBoost); (ii) calibration metrics including Brier score, expected calibration error (ECE), and logistic calibration slope/intercept; and (iii) screening-relevant operating characteristics—sensitivity, specificity, positive predictive value (PPV), and negative predictive value (NPV)—across probability thresholds 0.3–0.7. Sensitivity corresponds to recall at the chosen threshold; PPV to precision. As a transportability check, we additionally report institution-held-out (GroupKFold by campus) AUC mean±SD for CatBoost and XGBoost.

---

## 8. Paste-ready Results

On the student-level holdout, XGBoost achieved AUC 0.853 (95% CI 0.844–0.862), PR-AUC 0.823 (0.808–0.836), and Brier 0.156 (0.151–0.161). Competing ML/DL models yielded similar discrimination (AUC 0.847–0.853). Calibration showed ECE 0.033 with slope 0.202, indicating useful ranking but imperfect absolute probability calibration. At the default threshold 0.5, sensitivity was 0.769, specificity 0.776, PPV 0.737, and NPV 0.805; lowering the threshold to 0.3 increased sensitivity to 0.904 and NPV to 0.879 while reducing specificity to 0.569. Institution-held-out AUCs remained comparable to student-level performance (XGBoost 0.856±0.005 vs 0.853).

---

## 9. Paste-ready Limitations

> Bootstrap CIs and full screening tables are reported primarily for the XGBoost primary model. Calibration slope indicates residual miscalibration; local recalibration may be required before operational use. Formal pairwise DeLong tests across all architectures were not the focus of this revision. NPV depends on cohort prevalence (~0.45 in the evaluated split) and will differ in populations with different base rates.

---

## 10. Paste-ready reviewer response

Thank you for this important suggestion. We strengthened predictive-model evaluation beyond aggregate metrics by reporting: (1) 95% bootstrap confidence intervals for AUC, PR-AUC, F1, and Brier; (2) calibration analysis (Brier, ECE, slope/intercept); and (3) screening operating characteristics including sensitivity, specificity, PPV, and NPV across multiple thresholds. We also retain institution-held-out AUC as a transportability check. Results are summarized in Tables A–E of the revised manuscript / Supplementary Material.

---

## 11. File index

| File | Contents |
|---|---|
| `xai_rag_depression/outputs/model_performance.csv` | Multi-model point estimates |
| `xai_rag_depression/outputs/strengthened/table_b_bootstrap_ci.csv` | Bootstrap CIs |
| `xai_rag_depression/outputs/strengthened/table_c_calibration.csv` | Calibration |
| `xai_rag_depression/outputs/strengthened/threshold_screening_metrics.csv` | Sens/spec/PPV/F1 by threshold (XGBoost) |
| `xai_rag_depression/outputs/strengthened/threshold_screening_metrics_with_npv.csv` | Same + NPV (XGBoost) |
| `xai_rag_depression/outputs/strengthened/screening_metrics_all_models_thr0.5.csv` | Sens/spec/PPV/NPV all models @ 0.5 |
| `xai_rag_depression/outputs/strengthened/table_institution_vs_student_level.csv` | Held-out vs student AUC |
| `xai_rag_depression/outputs/strengthened/institution_heldout_reviewer_report.md` | Held-out narrative |
