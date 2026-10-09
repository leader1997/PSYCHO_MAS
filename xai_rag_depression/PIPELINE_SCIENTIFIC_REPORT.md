# Explainability-Augmented RAG for Depression Risk Classification Using the Healthy Minds Study Dataset

**Technical report for scientific publication**  
**Pipeline version:** 2024–2025 HMS wave | **Random seed:** 42 | **Execution mode:** FAST (`XAI_FAST=1`)

---

## Abstract

We present an end-to-end pipeline for binary classification of elevated depression and/or anxiety (`dep_or_anx`) among U.S. college students using the Healthy Minds Study (HMS) 2024–2025 wave. Nine predictive models—six classical machine learning (ML) and three deep learning (DL) architectures—are trained on 25 survey features, explained with model-specific explainable AI (XAI) methods, and aggregated into weighted global and local explanations. Explanations are combined with a retrieval-augmented generation (RAG) knowledge base comprising project artifacts and 27 vectorized literature PDFs on student mental health. A local large language model (LLM; Ollama, Llama 3) generates grounded, non-clinical screening narratives. We compare **four RAG systems** in an ablation study: baseline (dataset + prediction), RAG + local XAI, RAG + aggregated XAI, and the full system (local + aggregated XAI + literature). Results show strong cross-model agreement on academic impairment, loneliness, flourishing, and financial stress as primary drivers (AUC ≈ 0.85 across models), high explanation stability for tree-based ML models (Spearman/Kendall/Jaccard-based scores ≥ 0.93), and XAI-augmented systems achieving higher answer relevancy (0.90 vs 0.77 baseline) with comparable faithfulness (0.73–0.90).

---

## 1. Introduction

### 1.1 Motivation

College student mental health is a major public health concern. Survey-based screening models can identify students at elevated risk, but black-box predictions are insufficient for trustworthy communication with students, counselors, or policymakers. This pipeline addresses three gaps:

1. **Predictive performance** across diverse ML and DL model families on harmonized HMS data.
2. **Explainability** via SHAP, coefficients, and Integrated Gradients, with stability testing and cross-model aggregation.
3. **Grounded narrative generation** via RAG + LLM, combining model-specific XAI evidence with peer-reviewed and grey literature on student depression.

### 1.2 Research questions

- Which survey variables most strongly predict elevated depression/anxiety risk?
- Do tree-based ML and neural tabular models agree on feature importance rankings?
- Are explanations stable across random seeds, cross-validation folds, and input perturbations?
- Does augmenting RAG with local XAI, aggregated XAI, or both improve LLM answer quality compared to prediction-only RAG?

### 1.3 System title

**Explainability-Augmented RAG for Depression Risk Classification Using the Healthy Minds Study Dataset**

---

## 2. Data

### 2.1 Source

| Item | Value |
|------|-------|
| Dataset | Healthy Minds Study (HMS) 2024–2025 public release with institutional characteristics |
| File | `HMS_2024-2025_PUBLIC_instchars.csv` |
| Raw respondents | 84,735 |
| Target variable | `dep_or_anx` (binary: 1 = elevated depression and/or anxiety; 0 = not elevated) |

### 2.2 Inclusion criteria

1. **Survey completion:** `Finished == 1`
2. **Valid target:** non-missing `dep_or_anx`
3. **Feature completeness:** at least 60% of the 25 input features present (≥ 15 of 25 features non-missing per row)

### 2.3 Cleaned sample

| Statistic | Value |
|-----------|-------|
| Cleaned rows | **63,581** |
| Positive class rate | **44.9%** |
| Train / validation / test | **50,864 / 6,358 / 6,359** (80% / 10% / 10%) |
| Split method | Stratified by `dep_or_anx`, `random_state=42` |

### 2.4 Survey weights

Training uses the HMS non-response weight `nrweight` as `sample_weight` for ML models and weighted loss for DL models.

---

## 3. Features (n = 25)

Gender non-binary and transgender indicators (`gender_nonbin`, `gender_trans`) were **excluded** to align with the deep learning notebook configuration.

| Group | Features |
|-------|----------|
| **Demographics** | `age`, `gender_male`, `gender_female`, `international` |
| **Financial stress** | `fincur`, `finpast`, `food_worry` |
| **Sleep and lifestyle** | `sleep_wknight`, `sleep_wkend`, `exerc` |
| **Social support and belonging** | `belong1`, `belong2`, `lonely` |
| **Academic impact** | `aca_impa` |
| **Positive mental health** | `flourish`, `positiveMH` |
| **Substance use** | `sub_any`, `binge1pl`, `smok_cig` |
| **Discrimination** | `discrim_race`, `discrim_gender`, `discrim_sexual` |
| **Institution** | `inst_type`, `inst_size`, `inst_public` |

### 3.1 Feature definitions (selected)

| Feature | Description |
|---------|-------------|
| `aca_impa` | Academic impairment due to emotional/mental health (days academic performance affected) |
| `lonely` | Self-reported loneliness |
| `flourish` | Flourishing / positive psychological functioning |
| `fincur` | Current financial situation/stress (higher = more strain) |
| `food_worry` | Food insecurity / worry about running out of food |
| `positiveMH` | Positive mental health indicator |
| `dep_or_anx` | **Target:** elevated depression and/or anxiety screening flag |

### 3.2 Preprocessing

- Numeric coercion for all columns
- **Median imputation** + **StandardScaler** (fit on training set only)
- Same scaled feature space shared by all 9 models for comparable attributions

---

## 4. Predictive models (n = 9)

### 4.1 Model inventory and XAI methods

| Model | Family | XAI method |
|-------|--------|------------|
| Logistic Regression | ML | Standardized coefficients + odds ratios + SHAP LinearExplainer |
| Random Forest | ML | TreeSHAP + permutation importance (cross-check) |
| Extra Trees | ML | TreeSHAP + permutation importance (cross-check) |
| XGBoost | ML | TreeSHAP (native `pred_contribs` for XGBoost) |
| LightGBM | ML | TreeSHAP |
| CatBoost | ML | TreeSHAP |
| FT-Transformer | DL | Integrated Gradients (Captum) |
| TabTransformer | DL | Integrated Gradients (Captum) |
| DCNv2 | DL | Integrated Gradients (Captum) |

### 4.2 Hyperparameters (FAST mode, used for reported results)

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Tree ensembles (`n_estimators` / `iterations`) | 60 |
| Tree depth | 6 (boosting); `min_samples_leaf=5` (RF, Extra Trees) |
| Learning rate (boosting) | 0.05 |
| Class imbalance | `class_weight="balanced"` or `scale_pos_weight` |
| DL epochs | 4 (early stopping, patience=2) |
| DL optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| DL batch size | 512 |
| SHAP subsample (test set) | 300 rows |
| IG subsample (test set) | 150 rows |

*Full-quality mode (`XAI_FAST=0`): 300 trees, 35 DL epochs, 1,500 SHAP rows, 600 IG rows.*

### 4.3 Deep learning architectures

- **FT-Transformer:** per-feature linear tokenizer → CLS token + 2-layer Transformer encoder (d=64, 4 heads) → linear head
- **TabTransformer:** per-feature embedding → 2-layer Transformer (d=32) → flatten + MLP head
- **DCNv2:** CrossNetV2 (2 cross layers) + deep MLP (128 hidden) → concatenated head

---

## 5. Predictive performance (test set, n = 6,359)

| Model | Family | AUC | Accuracy | F1 | Macro F1 | Recall | PR-AUC | Brier |
|-------|--------|-----|----------|-----|----------|--------|--------|-------|
| **XGBoost** | ML | **0.853** | 0.773 | 0.753 | 0.772 | 0.769 | 0.823 | 0.156 |
| LightGBM | ML | 0.852 | 0.770 | 0.750 | 0.769 | 0.769 | 0.823 | 0.157 |
| CatBoost | ML | 0.852 | 0.772 | 0.752 | 0.770 | 0.772 | 0.825 | 0.157 |
| FT-Transformer | DL | 0.852 | 0.768 | 0.753 | 0.767 | 0.789 | 0.824 | 0.158 |
| TabTransformer | DL | 0.851 | 0.769 | 0.725 | 0.763 | 0.678 | 0.824 | 0.157 |
| DCNv2 | DL | 0.850 | 0.771 | 0.745 | 0.769 | 0.745 | 0.823 | 0.157 |
| Random Forest | ML | 0.849 | 0.769 | 0.747 | 0.767 | 0.759 | 0.820 | 0.158 |
| Extra Trees | ML | 0.848 | 0.768 | 0.747 | 0.766 | 0.764 | 0.819 | 0.158 |
| Logistic Regression | ML | 0.847 | 0.766 | 0.745 | 0.764 | 0.759 | 0.818 | 0.160 |

**Key findings:**
- All nine models achieve **AUC ≈ 0.847–0.853**, indicating strong but similar discriminative performance.
- Best ML model by AUC: **XGBoost (0.853)**; best DL: **FT-Transformer (0.852)**.
- TabTransformer has highest accuracy among DL models but lower recall (0.678) and F1 (0.725).
- Brier scores (0.156–0.160) indicate reasonable probabilistic calibration.

---

## 6. Explainable AI (XAI)

### 6.1 Normalized importance

For each model *m* and feature *j*:

$$\text{normalized\_importance}_{m,j} = \frac{|\text{importance}_{m,j}|}{\sum_k |\text{importance}_{m,k}|}$$

Importance sources: mean |SHAP| (ML), mean |Integrated Gradients| (DL), or |coefficient| (logistic regression).

### 6.2 Global feature rankings (mean across 9 models)

Top features by mean normalized importance:

| Rank | Feature | Mean importance (approx.) | In top-10 for all 9 models? |
|------|---------|---------------------------|----------------------------|
| 1 | `aca_impa` | 0.30 | Yes (9/9) |
| 2 | `lonely` | 0.17 | Yes (9/9) |
| 3 | `flourish` | 0.14 | Yes (9/9) |
| 4 | `fincur` | 0.08 | Yes (9/9) |
| 5 | `positiveMH` | 0.05 | Yes (9/9) |
| 6 | `age` | 0.03 | 7/9 |
| 7 | `food_worry` | 0.03 | Yes (9/9) |
| 8 | `sleep_wknight` | 0.03 | Yes (9/9) |

**Cross-model consensus:** Academic impairment (`aca_impa`), loneliness, flourishing, and current financial stress (`fincur`) are the four most consistently important features across all model families.

**Note on `aca_impa`:** This variable measures academic impairment *due to* emotional/mental health and is partially tautological with the depression/anxiety target. It should be interpreted as a strong correlate/marker rather than an independent upstream cause.

### 6.3 Per-model top-5 features

| Model | Top 5 features |
|-------|----------------|
| Logistic Regression | aca_impa, lonely, flourish, fincur, age |
| Random Forest | aca_impa, lonely, flourish, fincur, positiveMH |
| XGBoost | aca_impa, lonely, flourish, fincur, age |
| LightGBM | aca_impa, lonely, flourish, fincur, age |
| CatBoost | aca_impa, lonely, flourish, fincur, positiveMH |
| Extra Trees | aca_impa, lonely, positiveMH, flourish, fincur |
| FT-Transformer | aca_impa, lonely, flourish, fincur, sleep_wknight |
| TabTransformer | aca_impa, lonely, flourish, fincur, sleep_wknight |
| DCNv2 | aca_impa, flourish, lonely, fincur, positiveMH |

### 6.4 Semantic feature groups (aggregated)

Weighted aggregation uses model weights (Section 7). Group-level importances:

| Group | Aggregated importance | Cross-model agreement |
|-------|----------------------|------------------------|
| Academic impact | 0.300 | High |
| Social support and belonging | 0.194 | Low |
| Positive mental health | 0.188 | High |
| Financial stress | 0.132 | High |
| Sleep and lifestyle | 0.056 | Medium |
| Substance use | 0.041 | Low |
| Demographics | 0.037 | Low |
| Institution | 0.028 | Low |
| Discrimination | 0.023 | Low |

**Aggregated insight:** *"The aggregated explanation suggests that academic impact, social support and belonging, and positive mental health are the strongest contributors to depression-risk classification."*

### 6.5 Tree vs neural agreement

For top-10 features, tree models (RF, XGBoost, LightGBM, CatBoost, Extra Trees) and neural models (FT-Transformer, TabTransformer, DCNv2) show **high agreement** on core drivers (`aca_impa`, `lonely`, `flourish`, `fincur`, `food_worry`, `sleep_wknight`) but **diverge** on institution-type features and some discrimination variables.

---

## 7. Explanation stability and model weighting

### 7.1 Stability protocol

**ML models (full protocol, 7 runs each):**
- 3 random seeds (re-split + retrain)
- 2 stratified CV folds
- 2 Gaussian input perturbations (σ = 0.05 on scaled features)

**DL models (reduced protocol, 4 runs each):**
- 2 seed retrainings at reduced epochs
- 2 IG perturbation runs on same trained model

**Metrics (compared to baseline importances from Notebook 1):**
- Spearman rank correlation (mapped to [0,1])
- Kendall tau (mapped to [0,1])
- Jaccard similarity @ top-10 features

$$\text{stability\_score} = \text{mean}(\text{Spearman}_{01}, \text{Kendall}_{01}, \text{Jaccard@10})$$

### 7.2 Stability results

| Model | Stability score | Std | n runs | Calibration (1 − Brier) | Model weight |
|-------|-----------------|-----|--------|-------------------------|--------------|
| Random Forest | **0.972** | 0.031 | 7 | 0.842 | 0.628 |
| XGBoost | 0.969 | 0.023 | 7 | 0.844 | **0.631** |
| LightGBM | 0.968 | 0.031 | 7 | 0.843 | 0.627 |
| CatBoost | 0.965 | 0.033 | 7 | 0.843 | 0.627 |
| Logistic Regression | 0.963 | 0.035 | 7 | 0.840 | 0.619 |
| Extra Trees | 0.931 | 0.033 | 7 | 0.842 | 0.601 |
| FT-Transformer | 0.898 | 0.007 | 4 | 0.842 | 0.580 |
| TabTransformer | 0.846 | 0.002 | 4 | 0.843 | 0.544 |
| DCNv2 | 0.724 | 0.037 | 4 | 0.843 | 0.469 |

### 7.3 Model weight formula

$$\text{model\_weight}_m = \text{macro\_F1}_m \times \text{stability\_score}_m \times \text{calibration\_score}_m$$

where `calibration_score = 1 − Brier`.

### 7.4 Final aggregated feature importance

$$\text{final\_importance}_j = \frac{\sum_m \text{model\_weight}_m \cdot \text{normalized\_importance}_{m,j}}{\sum_m \text{model\_weight}_m}$$

Full rankings: see `outputs/aggregated_importance.csv`.

---

## 8. Explanation faithfulness (deletion test)

Top-5 features per model are masked (set to training median); test AUC drop measures whether highlighted features genuinely drive predictions.

| Model | Base AUC | Masked AUC (top-5) | AUC drop |
|-------|----------|---------------------|----------|
| DCNv2 | 0.850 | 0.644 | **0.206** |
| Random Forest | 0.849 | 0.652 | 0.196 |
| Extra Trees | 0.848 | 0.660 | 0.188 |
| LightGBM | 0.852 | 0.674 | 0.178 |
| XGBoost | 0.853 | 0.683 | 0.170 |
| CatBoost | 0.852 | 0.691 | 0.162 |
| TabTransformer | 0.851 | 0.718 | 0.133 |
| Logistic Regression | 0.847 | 0.721 | 0.126 |
| FT-Transformer | 0.852 | 0.726 | 0.125 |

Large AUC drops (0.16–0.21) for tree models confirm that top-ranked features are faithful drivers. DL models show smaller drops, suggesting IG attributions may be less tightly coupled to model decisions or that DL models rely on more distributed representations.

---

## 9. Local (instance-level) explanations

Three de-identified test cases selected by CatBoost predicted probability:

### 9.1 High-risk case (p = 0.979)

| Feature | Level | Direction | SHAP value |
|---------|-------|-----------|------------|
| aca_impa | high | increases risk | +1.258 |
| flourish | low | increases risk | +0.728 |
| fincur | low | increases risk | +0.415 |
| lonely | medium | increases risk | +0.345 |
| food_worry | high | increases risk | +0.324 |

### 9.2 Low-risk case (p = 0.05)

| Feature | Level | Direction | SHAP value |
|---------|-------|-----------|------------|
| aca_impa | low | decreases risk | −0.960 |
| lonely | low | decreases risk | −0.425 |
| flourish | high | decreases risk | −0.420 |
| age | high | decreases risk | −0.397 |
| fincur | high | decreases risk | −0.300 |

### 9.3 Borderline case (p = 0.50)

| Feature | Level | Direction | SHAP value |
|---------|-------|-----------|------------|
| fincur | low | increases risk | +0.456 |
| lonely | medium | increases risk | +0.431 |
| flourish | high | decreases risk | −0.390 |
| aca_impa | low | decreases risk | −0.249 |
| positiveMH | medium | decreases risk | −0.178 |

Feature levels are bucketed into low / medium / high using training-set tertiles. No row indices or quasi-identifiers are stored.

---

## 10. RAG knowledge base

### 10.1 Document types

| Type | Count | Description |
|------|-------|-------------|
| **Literature (PDF chunks)** | 2,278 | Chunked text from 27 student mental health PDFs |
| Feature dictionary | 25 | Plain-language definitions of input variables |
| Model performance cards | 9 | Per-model metrics + stability + calibration |
| Local explanation cards | 3 | De-identified case-level SHAP explanations |
| Global / aggregated cards | 2 | Cross-model importance and semantic groups |
| Dataset overview | 1 | HMS 2024–2025 description |
| Domain knowledge | 1 | Known correlates of student depression |
| Ethics limitations | 1 | Safe-use guidance |
| **Total documents** | **2,321** | |

### 10.2 Embedding and retrieval

| Parameter | Value |
|-----------|-------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Similarity | Cosine (normalized embeddings) |
| PDF chunk size | 900 characters, 150 overlap |
| Storage | `outputs/kb/embeddings.npy`, `documents.json` |

### 10.3 Literature corpus (27 PDFs)

| PDF | Chunks | Topic area |
|-----|--------|------------|
| HMS-2022-Executive-Summary.pdf | 30 | Healthy Minds Study findings |
| Mental_disorders_among_college_students... (WHO).pdf | 159 | Global college mental health epidemiology |
| 2024-State-of-Mental-Health-in-America-Report.pdf | 170 | National mental health statistics |
| Student Loneliness at Ohio State (2025).pdf | 31 | Loneliness on campus |
| A-Student-Primer-on-College-Mental-Health.pdf | 55 | Student/family guide |
| counseling-center-mental-health-guide...pdf | 50 | Help-seeking, campus counseling |
| Stepped-Care-Guide-V10.pdf | 65 | Stepped care models |
| Creating-Environments-for-Flourishing.pdf | 69 | Positive mental health / flourishing |
| UNCF_MentalHealthReport_Flourishing_FINAL.pdf | 73 | Flourishing in minority-serving institutions |
| Eisenberg2012.pdf | 74 | Help-seeking barriers |
| collegedrinkingFactSheet2021.pdf | 18 | Substance use |
| JISS 2023 Burnout in College.pdf | 44 | Academic burnout |
| *(+ 15 additional reports)* | — | Trends, discrimination, wellbeing, NSDUH, etc. |

Full inventory: `outputs/kb/pdf_inventory.csv`.

### 10.4 Retrieval quality (sanity checks)

Example queries after PDF integration:

| Query | Top retrieved document type |
|-------|------------------------------|
| "Why is this student at high depression risk?" | Literature (Mental-Health_Report, Student Primer) |
| "What helps with loneliness and depression?" | Literature (Ohio State Loneliness report) |
| "When should a student seek campus counseling?" | Literature (MentalHealth interactive guide, Student Primer) |
| "What are the most important feature groups?" | XAI aggregated importance summary |
| "Is this a clinical diagnosis?" | Literature (WHO college mental health) + ethics card |

---

## 11. LLM generation and evaluation

### 11.1 Systems compared (ablation design)

| ID | System | Local XAI | Aggregated XAI | Literature | Context provided to LLM |
|----|--------|-----------|----------------|------------|-------------------------|
| **S0** | `baseline` | ✗ | ✗ | ✗ | Dataset description + predicted class and probability |
| **S1** | `rag_xai` | ✓ | ✗ | ✓ | S0 + CatBoost SHAP top-5 factors + 6 retrieved literature chunks |
| **S2** | `rag_agg_xai` | ✗ | ✓ | ✓ | S0 + aggregated semantic groups + agreement scores + 6 literature chunks |
| **S3** | `full` | ✓ | ✓ | ✓ | S0 + local XAI + aggregated XAI + 6 literature chunks |

Literature retrieval is held constant across S1–S3 so the ablation isolates the contribution of local vs aggregated XAI.

### 11.2 LLM configuration

| Parameter | Value |
|-----------|-------|
| Backend | Ollama (local) |
| Model | Llama 3 (`llama3:latest`) |
| Temperature | 0.2 (generation), 0.0 (judge) |
| System prompt | Baseline: prediction-only sections; XAI systems: strict grounding, no clinical diagnosis, uncertainty, model agreement, literature-grounded recommendation |

### 11.3 LLM-as-judge metrics

Evaluated on 3 de-identified cases (high-risk, low-risk, borderline). Full results: `outputs/rag_ablation_evaluation.csv`.

| Case | System | Faithfulness | Answer relevancy | Hallucination rate | Context precision | Context recall |
|------|--------|--------------|------------------|--------------------|-------------------|----------------|
| high_risk | baseline | 0.90 | 0.50 | 0.50 | — | — |
| high_risk | rag_xai | 0.90 | 0.90 | 0.50 | 0.20 | 0.70 |
| high_risk | rag_agg_xai | 0.80 | 0.90 | 0.50 | 0.20 | 0.70 |
| high_risk | **full** | 0.80 | 0.90 | **0.20** | 0.20 | 0.70 |
| low_risk | baseline | 0.90 | 0.90 | 0.50 | — | — |
| low_risk | rag_xai | 0.80 | 0.90 | 0.50 | 0.20 | 0.70 |
| low_risk | rag_agg_xai | 0.70 | 0.90 | **0.20** | 0.20 | 0.70 |
| low_risk | full | 0.70 | 0.90 | 0.50 | 0.20 | 0.70 |
| borderline | baseline | 0.90 | 0.90 | 0.50 | — | — |
| borderline | rag_xai | 0.70 | 0.90 | 0.50 | 0.20 | 0.70 |
| borderline | **rag_agg_xai** | **0.90** | 0.90 | 0.50 | 0.20 | 0.40 |
| borderline | full | 0.70 | 0.90 | 0.50 | 0.20 | 0.70 |

**System-level means:**

| Metric | S0 baseline | S1 rag_xai | S2 rag_agg_xai | S3 full |
|--------|-------------|------------|----------------|---------|
| Faithfulness | **0.90** | 0.80 | 0.80 | 0.73 |
| Answer relevancy | 0.77 | **0.90** | **0.90** | **0.90** |
| Hallucination rate | 0.50 | 0.50 | **0.40** | **0.40** |
| Context precision | — | 0.20 | 0.20 | 0.20 |
| Context recall | — | **0.70** | 0.60 | **0.70** |

### 11.4 Ablation interpretation

1. **XAI improves relevancy.** All three XAI-augmented systems (S1–S3) achieve mean answer relevancy of 0.90 vs 0.77 for baseline. Without XAI evidence, the LLM produces generic answers that score lower on relevancy (notably high-risk case: 0.50 vs 0.90).

2. **Local vs aggregated XAI trade-offs.** S1 (local SHAP) and S2 (aggregated groups) tie on faithfulness (0.80) and relevancy (0.90). S2 achieves lower mean hallucination (0.40 vs 0.50), suggesting consensus-based semantic groups reduce unsupported claims. S1 achieves higher context recall (0.70 vs 0.60), reflecting tighter grounding in case-specific factors.

3. **Case-dependent winners.** Aggregated XAI (S2) wins faithfulness on the borderline case (0.90 vs 0.70 for S1/S3), where cross-model consensus is most informative. Local XAI (S1) wins faithfulness on the high-risk case (0.90). The full system (S3) achieves the lowest hallucination on high-risk (0.20) by combining both evidence types.

4. **Full system is not uniformly best.** S3 does not dominate all metrics — faithfulness drops to 0.73 (mean) because richer context increases the chance of minor unsupported elaboration. For deployment, S2 (aggregated only) or S1 (local only) may be preferable depending on whether case-specific or consensus-based narratives are needed.

### 11.5 Example generated answer (high-risk case, full system)

**Prediction:** Elevated depression/anxiety risk (probability 0.979).

**Main contributing factors:** High academic impairment (`aca_impa`), low flourishing, low financial stress indicator (`fincur`), medium loneliness, high food worry.

**Model agreement:** Academic impact (importance 0.30, high agreement), financial stress (0.13, high), positive mental health (0.19, high).

**Uncertainty:** High model confidence but complex factor interactions; observational associations, not causal claims.

**Recommendation (non-clinical):** Consider campus counseling services, stress-reduction strategies, and activities supporting positive mental health. Grounded in retrieved literature on college student mental health prevalence and help-seeking.

### 11.6 RAGAS benchmark evaluation (100 questions)

A dedicated labeled benchmark (`outputs/rag_eval_benchmark.csv`, 10 categories × 10 questions) is evaluated with **six RAGAS metrics** following Batal et al. (2026, *Sustainability* 18(2):875, §6.3) and the RAGAS framework (Es et al., 2023):

| Metric | RAGAS implementation |
|--------|----------------------|
| Context precision | `context_precision` |
| Faithfulness | `faithfulness` |
| Answer relevancy | `answer_relevancy` |
| Context recall | `context_recall` |
| Answer correctness | `answer_correctness` |
| Semantic similarity | `answer_similarity` → `semantic_similarity` |

**Configuration:** Ollama generation (default `llama3`); RAGAS evaluator LLM `gpt-4o` (`XAI_RAGAS_MODEL`); embeddings `text-embedding-3-small`.

**Citable artifact:** `07_ragas_benchmark_evaluation.ipynb` | **Runner:** `python run_nb7.py`

**Outputs:** `rag_eval_ragas_results.csv`, `rag_eval_ragas_summary.csv`, `rag_eval_ragas_by_category.csv`

**Supplementary (domain-specific, not in Batal et al.):** chunk-level P@k/R@k vs `gold_relevant_chunk_ids`; `safety_score` and `refusal_appropriateness` for mental-health guardrails.

---

## 12. Pipeline architecture

```
HMS 2024-2025 CSV
       │
       ▼
┌──────────────────────────────────────┐
│  Notebook 1: Models + XAI            │
│  • 6 ML + 3 DL models                │
│  • SHAP / IG / coefficients          │
│  • Global + local explanations       │
│  • Faithfulness (deletion test)      │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  Notebook 2: Stability + Aggregation   │
│  • Seed / CV / perturbation stability  │
│  • Weighted aggregation                │
│  • Semantic groups + JSON cards        │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  Notebook 3: RAG Knowledge Base        │
│  • Project cards + 27 PDFs (2,278 chunks)│
│  • Embed (MiniLM-L6-v2, dim=384)      │
│  • Cosine retrieval                    │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  Notebook 4: LLM Generation + Eval     │
│  • 4-system ablation (baseline,        │
│    local XAI, aggregated XAI, full)    │
│  • Ollama (Llama 3)                   │
│  • LLM-as-judge metrics                │
└──────────────────────────────────────┘
```

---

## 13. Ethical limitations

The following must be stated in any publication or deployment:

1. **Not a diagnostic tool.** Predictions are screening signals, not clinical diagnoses.
2. **Does not replace professionals.** The system must not substitute for licensed mental-health providers.
3. **Bias risk.** Models trained on observational survey data may reflect historical biases, especially for sensitive features (gender, race, discrimination).
4. **Privacy.** Local explanations use bucketed feature levels; no individual identifiers are stored in explanation cards.
5. **Association ≠ causation.** Feature importances reflect model reliance, not proven causal effects.
6. **`aca_impa` circularity.** Academic impairment is partially defined by mental health status; its dominance in explanations should be interpreted cautiously.
7. **LLM grounding.** Answers must be restricted to retrieved evidence; hallucination rate remains non-zero (≈ 0.40 proposed system mean).
8. **Literature retrieval precision.** Context precision (≈ 0.20) indicates room for improved chunking, reranking, or hybrid retrieval.

---

## 14. Limitations and future work

| Limitation | Future direction |
|------------|------------------|
| Single wave (2024–2025) | Multi-wave longitudinal validation (2007–2025 HMS series) |
| FAST training mode (60 trees, 4 DL epochs) | Full-quality retraining (`XAI_FAST=0`) |
| CPU-only DL training | GPU acceleration for larger architectures |
| Single LLM (Llama 3 8B) | Compare multiple LLMs; human expert evaluation |
| Low RAG context precision | Hybrid retrieval (BM25 + dense), cross-encoder reranking |
| No causal inference | Propensity-score or instrumental-variable sensitivity analyses |
| `aca_impa` tautology | Ablation study excluding partially outcome-defined features |

---

## 15. Reproducibility

### 15.1 Repository structure

```
xai_rag_depression/
├── 01_models_and_explainability.ipynb
├── 02_stability_aggregation_cards.ipynb
├── 03_rag_knowledge_base.ipynb
├── 04_llm_generation_evaluation.ipynb
├── 07_ragas_benchmark_evaluation.ipynb      # RAGAS 100-Q eval (Batal et al. 2026)
├── run_nb1.py / run_nb2.py / run_nb4.py / run_nb7.py
├── literature/pdfs/                        # 27 PDF corpus
└── outputs/
    ├── model_performance.csv
    ├── global_importance_per_model.csv
    ├── aggregated_importance.csv
    ├── stability_scores.csv
    ├── faithfulness.csv
    ├── local_explanations.json
    ├── rag_evaluation.csv
    ├── cards/                              # JSON explanation cards
    └── kb/                                 # embeddings.npy, documents.json
```

### 15.2 Dependencies

Python 3.12; scikit-learn, xgboost, lightgbm, catboost, torch, captum, shap, sentence-transformers, pypdf, pandas, numpy, scipy, requests, ragas, datasets, langchain-openai; Ollama for LLM generation; OpenAI API for RAGAS evaluation.

### 15.3 Execution

```powershell
cd xai_rag_depression
$env:XAI_FAST="1"          # or "0" for full quality
python run_nb1.py
python run_nb2.py
jupyter nbconvert --execute 03_rag_knowledge_base.ipynb
$env:XAI_OLLAMA_MODEL="llama3:latest"
python run_nb4.py
```

---

## 16. Summary of key results for paper tables

**Table 1.** Test-set predictive performance (9 models, n=6,359): AUC range 0.847–0.853.

**Table 2.** Top aggregated features: aca_impa (0.30), lonely (0.17), flourish (0.14), fincur (0.08) — all with high cross-model agreement (≥ 7/9 models in top-10).

**Table 3.** Explanation stability: tree ML models ≥ 0.93; DL models 0.72–0.90.

**Table 4.** Faithfulness (AUC drop when masking top-5): 0.13–0.21 across models.

**Table 5.** RAG ablation (3 cases, LLM-as-judge): XAI systems (S1–S3) achieve answer relevancy 0.90 vs 0.77 baseline; aggregated XAI (S2) lowest hallucination (0.40); full system (S3) best context recall (0.70).

**Table 6.** Knowledge base: 2,321 documents (2,278 literature chunks from 27 PDFs), 384-d embeddings.

---

## References (literature corpus)

The RAG knowledge base includes peer-reviewed articles, government reports, and institutional guides. Key sources ingested:

- Healthy Minds Study Executive Summary (2022)
- WHO World Mental Health Survey — college students
- Mental Health America — State of Mental Health in America (2024)
- Eisenberg, D. et al. — help-seeking among college students
- Ohio State University — Student Loneliness Report (2025)
- JED Foundation / Stepped Care Guide
- UNCF — Mental Health and Flourishing Report
- NIH / NCBI Bookshelf (NBK595227)
- Multiple peer-reviewed journal articles (Scientific Reports, BMC Psychiatry, etc.)

Full bibliography with file names: `outputs/kb/pdf_inventory.csv`.

---

*Report generated from pipeline outputs in `xai_rag_depression/outputs/`. All numeric values reflect the FAST-mode run (seed=42) executed on 2026-06-13.*
