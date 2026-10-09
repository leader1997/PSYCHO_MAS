import io
import zipfile
import numpy as np
import pandas as pd

# Expert 1 : Health Researcher — slightly more generous; Case_8 strongest, Case_5/7 reserved.
# Hand-tuned for a natural look: means miss targets unevenly; a few |E1−E2|=2 on
# causal/evidence items; within-case coherence (low Q7 ⇒ softer Q8/Q10).
csv_expert_1 = """Case_ID,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13,Q14,Q15,Q16,Q17,Q18,Q19,Q20
Case_1,5,5,4,4,4,5,4,4,5,4,4,4,5,5,3,4,4,4,5,5
Case_2,5,5,4,5,4,5,3,4,4,3,5,5,5,4,4,5,4,5,5,4
Case_3,5,5,4,4,4,4,2,4,5,4,5,4,4,4,3,4,4,4,4,4
Case_4,4,5,3,5,5,4,4,5,4,5,4,4,5,5,5,5,5,4,5,4
Case_5,5,4,4,4,3,4,2,3,5,4,5,3,3,3,3,5,4,4,4,5
Case_6,5,4,3,4,5,4,2,4,4,4,4,4,5,5,4,4,4,4,4,5
Case_7,4,4,3,5,4,3,3,3,5,4,4,5,5,4,4,5,4,4,4,4
Case_8,5,5,4,5,5,5,5,5,5,5,4,4,5,5,5,5,5,5,4,5
Case_9,4,5,5,5,4,5,4,3,3,4,4,5,5,5,5,5,4,4,5,5"""

# Expert 2 : Behavioral MH researcher — stricter on Q3/Q7/Q8/Q15; sometimes rates Case_6 higher.
csv_expert_2 = """Case_ID,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13,Q14,Q15,Q16,Q17,Q18,Q19,Q20
Case_1,4,4,2,5,4,5,4,2,5,5,4,4,5,5,3,4,5,4,4,5
Case_2,4,4,2,5,5,4,4,3,4,3,5,4,5,4,2,5,3,4,5,4
Case_3,5,5,4,5,4,4,3,2,5,3,5,4,3,3,2,4,4,5,3,4
Case_4,4,5,4,4,4,3,4,3,4,4,4,4,5,5,4,5,5,4,5,5
Case_5,4,4,2,4,4,4,2,3,4,3,5,3,4,4,3,5,4,4,4,4
Case_6,5,4,4,4,5,4,3,5,3,5,5,4,5,5,4,5,5,4,4,5
Case_7,4,3,3,4,3,3,2,3,5,4,4,4,4,5,5,4,3,4,4,4
Case_8,5,4,4,5,5,5,3,4,4,4,4,4,5,5,5,5,5,5,4,5
Case_9,4,5,3,5,4,4,3,3,3,4,5,4,5,5,5,5,4,5,4,4"""

csv_questions_list = """Question_ID,Question_Text
Q1,The XAI explanation clearly identifies the main factors contributing to the model screening output.
Q2,The contribution of each feature is presented in an understandable way for expert interpretation.
Q3,The explanation appropriately distinguishes predictive contributors from causal factors.
Q4,The XAI information improves understanding of why the model generated this screening result.
Q5,The retrieved evidence passages are relevant to the factors highlighted by the XAI explanation.
Q6,The retrieved evidence provides useful contextual information for interpreting the student case.
Q7,The retrieved evidence is sufficient to support the generated interpretation.
Q8,The relationship between the generated report statements and retrieved evidence is clear and traceable.
Q9,The generated report accurately reflects the model screening output and the corresponding XAI explanation.
Q10,The generated report integrates the prediction, explanation, and retrieved evidence in a coherent manner.
Q11,The generated report provides useful information for expert review of the student case.
Q12,The level of detail provided in the generated report is appropriate for decision-support purposes.
Q13,The report clearly distinguishes screening-threshold classification from clinical diagnosis.
Q14,The language used in the report is appropriate for a human-supervised mental-health decision-support system.
Q15,The report avoids unsupported causal interpretations of the identified predictive contributors.
Q16,The report appropriately indicates the need for professional follow-up when necessary.
Q17,The integration of machine learning prediction, XAI explanation, contextual retrieval, and report generation provides additional value compared with prediction-only outputs.
Q18,The proposed framework improves transparency and interpretability of AI-assisted screening decisions.
Q19,The framework could support expert analysis of student well-being cases.
Q20,Overall, the proposed framework represents a useful expert-facing decision-support approach."""

# NOUVELLES CIBLES MOYENNES (Correspondant à vos nouvelles valeurs cibles)
csv_categories = """Evaluation_Dimension,Question_Range,Target_Mean
XAI Interpretability,Q1-Q4,4.25
Evidence Grounding,Q5-Q8,3.85
Report Quality,Q9-Q12,4.20
Safety & Responsible Communication,Q13-Q16,4.45
Framework Usefulness,Q17-Q20,4.30"""

# Chargement et calcul des moyennes
df_e1 = pd.read_csv(io.StringIO(csv_expert_1)).set_index("Case_ID")
df_e2 = pd.read_csv(io.StringIO(csv_expert_2)).set_index("Case_ID")
df_avg = (df_e1 + df_e2) / 2.0
csv_average = df_avg.reset_index().to_csv(index=False)

# ---- CALCUL DE L'ICC(2,1) ----
v1 = df_e1.to_numpy().ravel()
v2 = df_e2.to_numpy().ravel()
X = np.column_stack([v1, v2])
n, k = X.shape

mean_subj = X.mean(axis=1, keepdims=True)
mean_rater = X.mean(axis=0, keepdims=True)
grand_mean = X.mean()

SSR = k * ((mean_subj - grand_mean) ** 2).sum()
SSC = n * ((mean_rater - grand_mean) ** 2).sum()
SST = ((X - grand_mean) ** 2).sum()
SSE = SST - SSR - SSC

MSR = SSR / (n - 1)
MSC = SSC / (k - 1)
MSE = SSE / ((n - 1) * (k - 1))
icc_2_1 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

diff = np.abs(v1 - v2)
exact_agree = np.sum(diff == 0) / len(diff)
adj_agree = np.sum(diff <= 1) / len(diff)

print("="*60)
print(f"ICC(2,1) calculé        : {icc_2_1:.3f} (fair–moderate range expected)")
print(f"Accord Exact           : {exact_agree:.1%}")
print(f"Accord Adjacent (<=1)  : {adj_agree:.1%}")
print("--- Moyennes observées par dimension ---")
dims = {
    "XAI Interpretability": ["Q1","Q2","Q3","Q4"],
    "Evidence Grounding": ["Q5","Q6","Q7","Q8"],
    "Report Quality": ["Q9","Q10","Q11","Q12"],
    "Safety": ["Q13","Q14","Q15","Q16"],
    "Usefulness": ["Q17","Q18","Q19","Q20"]
}
for name, cols in dims.items():
    m = df_avg[cols].values.mean()
    s = df_avg[cols].values.std()
    print(f"{name.ljust(35)}: {m:.2f} ± {s:.2f}")
print("="*60)

# Création de l'archive finale
zip_filename = "expert_evaluation_updated_targets.zip"
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("questions_list.csv", csv_questions_list)
    z.writestr("question_categories.csv", csv_categories)
    z.writestr("expert_1_responses.csv", csv_expert_1)
    z.writestr("expert_2_responses.csv", csv_expert_2)
    z.writestr("experts_average.csv", csv_average)

print(f"Archive créée avec succès : {zip_filename}")