# Institution-Held-Out Validation Report
## Response to Reviewer — Cross-Campus Transportability

**Study:** Agentic XAI–RAG for HMS depression/anxiety screening  
**Analysis date:** 2026-09-21  
**Cohort:** HMS 2024–2025 public release with institutional characteristics  
**Analytic N:** 63,581 students | **Campuses (`schoolnum`):** 135  
**Outcome:** `dep_or_anx` (elevated depression/anxiety screening signal; prevalence ≈ 44.9%)  
**Notebook / script:** `xai_rag_depression/10_institution_heldout_validation.ipynb`, `run_institution_heldout.py`  
**Outputs:** `outputs/strengthened/institution_heldout_*.csv`, `table_institution_vs_student_level.csv`

---

## 1. Reviewer concern (what we addressed)

The reviewer requested evidence that model performance is not an artifact of **students from the same institutions appearing in both training and test sets**. With only a **student-level** stratified split, campuses can be shared across train/test, which can overstate **cross-institution** generalization.

We therefore added an **institution-held-out (leave-campus-out)** sensitivity analysis. This does **not** replace the primary student-level evaluation; it answers the transportability question.

---

## 2. Relation to the primary analysis (what stayed unchanged)

| Item | Status |
|------|--------|
| Primary split | Student-level stratified 80/10/10 (seed 42) — **unchanged** |
| Primary performance table | `outputs/model_performance.csv` — **kept** |
| Institution-held-out | **New** sensitivity / transportability check |

**Reporting stance for the manuscript**
- Primary claim: discrimination under student-level holdout  
- Secondary claim: discrimination remains comparable when **entire campuses** are held out of training  

---

## 3. Methods (detailed, reviewer-ready)

### 3.1 Grouping unit
- Group key: **`schoolnum`** (campus ID in HMS)
- **`schoolnum` was never used as a predictor**
- Institutional covariates (`inst_type`, `inst_size`, `inst_public`) remain features only in the E1 setting (see §3.4)

### 3.2 Validation scheme
- Algorithm: **`GroupKFold(n_splits=5)`**
- Constraint: for each fold, **no campus appears in both training and test**
- Observed fold balance (E1 run): ~25–28 held-out schools per fold; ~12.7k students per test fold

### 3.3 Preprocessing and training (per fold)
All of the following were fit **only on training campuses** within each fold:
1. Median imputation (`SimpleImputer`)
2. Standardization (`StandardScaler`)
3. Classifier fitting with HMS non-response weights (`nrweight`)

Test campuses were transformed with the train-fitted preprocessor only.

### 3.4 Models and feature sets
**Models (paper primary ML models):**
- CatBoost  
- XGBoost  

**Feature experiments:**

| Code | Description | # features |
|------|-------------|------------|
| **E1_full** | Full analytic feature set (same 25 predictors as main paper) | 25 |
| **E6_no_institution** | Drop `inst_type`, `inst_size`, `inst_public` (stronger check that performance does not depend on institutional characteristics) | 22 |

Training budget matched the published pipeline (`XAI_FAST=1`, same tree count as primary student-level results) for comparability.

### 3.5 Metrics
Per fold, then summarized as **mean ± SD across 5 folds**:
- AUC (primary discrimination metric)
- PR-AUC
- Brier score
- Recall at probability threshold 0.5

Comparison metric:
- **ΔAUC = student-level AUC − institution-held-out mean AUC**  
  (negative Δ means held-out AUC was slightly higher)

### 3.6 Cohort construction (aligned with main paper)
Same inclusion rules as the primary pipeline:
- `Finished == 1`
- Non-missing `dep_or_anx`
- ≥60% feature completeness on the 25 predictors
- Resulting N = 63,581 students across 135 campuses

---

## 4. Results

### 4.1 Main comparison table (E1 — full features)

| Model | Student-level AUC (primary) | Institution-held-out AUC (mean ± SD) | ΔAUC (student − held-out) |
|-------|----------------------------:|-------------------------------------:|--------------------------:|
| CatBoost | 0.852 | 0.855 ± 0.004 | −0.003 |
| XGBoost | 0.853 | 0.856 ± 0.005 | −0.002 |

**Interpretation:** Under leave-campus-out evaluation, discrimination did **not** degrade relative to the student-level holdout. Point estimates were essentially unchanged (within ~0.003 AUC).

### 4.2 Stronger check without institution covariates (E6)

| Model | Student-level AUC (ref.) | Institution-held-out AUC (mean ± SD) | ΔAUC |
|-------|-------------------------:|-------------------------------------:|-----:|
| CatBoost | 0.852 | 0.854 ± 0.004 | −0.002 |
| XGBoost | 0.853 | 0.855 ± 0.004 | −0.002 |

**Interpretation:** Removing institutional characteristic features does not materially change leave-campus-out performance, supporting that discrimination is driven primarily by **student-level** predictors rather than campus metadata.

### 4.3 Additional metrics (E1, mean across folds)

| Model | Held-out PR-AUC | Held-out Brier |
|-------|----------------:|---------------:|
| CatBoost | 0.829 ± 0.005 | 0.155 ± 0.003 |
| XGBoost | 0.830 ± 0.005 | 0.155 ± 0.003 |

These remain close to the corresponding student-level PR-AUC / Brier values in `model_performance.csv`.

### 4.4 Full numeric table (from `table_institution_vs_student_level.csv`)

| experiment | model | student_level_auc | institution_heldout_auc_mean | institution_heldout_auc_std | delta_auc | student_level_pr_auc | heldout_pr_auc_mean | heldout_pr_auc_std | student_level_brier | heldout_brier_mean | heldout_brier_std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1_full | CatBoost | 0.852041 | 0.854679 | 0.004474 | −0.002638 | 0.824839 | 0.828510 | 0.005026 | 0.156911 | 0.155257 | 0.002660 |
| E1_full | XGBoost | 0.853226 | 0.855512 | 0.004600 | −0.002286 | 0.823070 | 0.829648 | 0.005065 | 0.156096 | 0.154766 | 0.002939 |
| E6_no_institution | CatBoost | 0.852041 | 0.854480 | 0.004065 | −0.002439 | 0.824839 | 0.828067 | 0.005484 | 0.156911 | 0.155400 | 0.002439 |
| E6_no_institution | XGBoost | 0.853226 | 0.855288 | 0.004137 | −0.002062 | 0.823070 | 0.829422 | 0.005207 | 0.156096 | 0.155006 | 0.002696 |

### 4.5 Per-fold AUC stability (E1)

Held-out AUCs across folds were tight (~0.850–0.863), indicating the result is not driven by a single easy/hard campus fold.

**E1 CatBoost (per fold AUC):** 0.8618, 0.8557, 0.8504, 0.8537, 0.8518  
**E1 XGBoost (per fold AUC):** 0.8632, 0.8564, 0.8520, 0.8536, 0.8524  

Fold composition: ~25–28 test schools and ~12,709–12,730 test students per fold. Full per-fold rows (including PR-AUC, Brier, recall, prevalence) are in `institution_heldout_per_fold.csv`.

---

## 5. What this does and does not show

### Supports
- Cross-campus **transportability of discrimination** within the HMS 2024–2025 multi-institution sample
- That shared-campus leakage in the student-level split was **not** the main driver of the reported AUCs
- That institutional covariates are **not required** for held-out campus performance (E6)

### Does not claim
- Clinical diagnostic validity
- Guaranteed performance at a future campus outside HMS participation / sampling frame
- That calibration or threshold operating points need no local adaptation at deployment
- That fairness across demographic subgroups is solved (that remains a separate analysis)

---

## 6. Suggested manuscript placement

1. **Keep** the original student-level model comparison table as the primary Results table.  
2. **Add** a new short table (main text or Supplement): “Student-level vs institution-held-out AUC.”  
3. **Methods:** add one paragraph describing GroupKFold by `schoolnum` (see §7).  
4. **Results:** 2–3 sentences with CatBoost/XGBoost mean±SD and Δ.  
5. **Limitations:** replace “we did not evaluate unseen institutions” with a statement that institution-held-out was performed and showed no AUC degradation, while noting external HMS-naïve campuses still warrant local validation.

---

## 7. Paste-ready text

### Methods
In addition to the primary student-level stratified 80/10/10 split, we assessed cross-institution generalization using 5-fold GroupKFold grouped by campus identifier (`schoolnum`; N=135 institutions). Within each fold, preprocessing and model fitting were performed only on training campuses, and evaluation used entirely held-out campuses. Campus ID was used solely as a grouping key and was not included as a predictor. We report mean ± SD of AUC, PR-AUC, and Brier across folds for CatBoost and XGBoost under the full feature set (E1) and a no-institution-covariate sensitivity set (E6).

### Results
Institution-held-out discrimination remained comparable to student-level performance. For the full feature set, CatBoost AUC was 0.855±0.004 (vs student-level 0.852) and XGBoost AUC was 0.856±0.005 (vs 0.853). Mean AUC difference (student-level minus institution-held-out) was approximately −0.002, indicating no degradation under leave-campus-out evaluation. Results were essentially unchanged after removing institutional covariates (E6).

### Reviewer rebuttal
We thank the reviewer for requesting an institution-held-out evaluation. Our primary evaluation remains the student-level stratified holdout for comparability with the original manuscript. Because HMS is multi-institutional, we additionally performed leave-institution-out validation via GroupKFold on `schoolnum` (K=5; 135 campuses). Institution-held-out discrimination remained close to student-level performance (CatBoost 0.855±0.004 vs 0.852; XGBoost 0.856±0.005 vs 0.853), with no AUC degradation relative to the student-level holdout. We report this as a transportability sensitivity analysis and have updated Methods/Results/Limitations accordingly. We also report an E6 (no institution covariates) GroupKFold setting as a stronger check that performance does not rely on institutional characteristic features.

### Limitations (updated wording)
Although the primary evaluation used student-level splitting, we additionally evaluated leave-institution-out generalization with GroupKFold on campus identifiers. Discrimination did not degrade for held-out campuses within the HMS 2024–2025 sample. Nonetheless, performance at institutions outside this survey wave or sampling frame may still require local calibration before operational use.

---

## 8. File checklist for the response package

| File | Contents |
|------|----------|
| `table_institution_vs_student_level.csv` | Side-by-side student vs held-out metrics |
| `institution_heldout_summary.csv` | Mean±SD by experiment/model |
| `institution_heldout_per_fold.csv` | Fold-level transparency |
| `institution_heldout_manuscript_snippets.md` | Short paste snippets |
| `10_institution_heldout_validation.ipynb` | Reproducible analysis notebook |
| `run_institution_heldout.py` | CLI runner for the same analysis |

---

## 9. One-sentence bottom line for the reviewer

**After holding out entire campuses from training (GroupKFold on `schoolnum`, K=5, 135 institutions), CatBoost/XGBoost AUCs remained ~0.85 and did not fall below the primary student-level estimates, including when institutional covariates were removed.**
