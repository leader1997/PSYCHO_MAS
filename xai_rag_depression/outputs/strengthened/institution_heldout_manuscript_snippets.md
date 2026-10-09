# Institution-held-out — manuscript / rebuttal snippets

## Methods

In addition to the primary student-level stratified 80/10/10 split, we assessed cross-institution generalization using 5-fold GroupKFold grouped by campus identifier (`schoolnum`; N=135 institutions). Within each fold, preprocessing and model fitting were performed only on training campuses, and evaluation used entirely held-out campuses. Campus ID was used solely as a grouping key and was not included as a predictor.

## Results

Institution-held-out discrimination remained close to student-level performance. CatBoost: student-level AUC=0.852; institution-held-out AUC=0.855±0.004 (delta=-0.003). XGBoost: student-level AUC=0.853; institution-held-out AUC=0.856±0.005 (delta=-0.002). Mean AUC difference (student-level minus institution-held-out) was -0.002, indicating no degradation under leave-campus-out evaluation. Results were essentially unchanged when institution covariates were removed (E6).

## Reviewer rebuttal

We thank the reviewer for requesting an institution-held-out evaluation. Our primary evaluation remains the student-level stratified holdout for comparability with the original manuscript. Because HMS is multi-institutional, we additionally performed leave-institution-out validation via GroupKFold on `schoolnum` (K=5; 135 campuses). Institution-held-out discrimination remained close to student-level performance (CatBoost 0.855±0.004 vs 0.852; XGBoost 0.856±0.005 vs 0.853), with no AUC degradation relative to the student-level holdout. We now report this as a transportability sensitivity analysis (Table: institution vs student-level) and have updated Methods/Results/Limitations accordingly. We also report the E6 (no institution covariates) GroupKFold setting as a stronger check that performance does not rely on institutional characteristic features.
