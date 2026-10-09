# End-to-End Case Walkthrough - `hms_high_01`

**Purpose.** Show every major data object that circulates while one student case is processed by the Agentic XAI-Contextual RAG pipeline (prediction -> XAI -> query planning -> retrieval -> grounded answers -> scoring).

**Case:** elevated screening stratum (`high`)  
**Plan model (frozen):** `gemma4`  
**Reference / judge model:** `gpt-oss:20b`  
**Retrieval:** Ollama-context hybrid, K=5  
**Artifact bundle (full JSON):** `paper_outputs/end_to_end_case_hms_high_01_bundle/`  
**Manuscript-ready generated report (Q1–Q6 + reference answers + RAGAS):** [`example_generated_report_hms_high_01.md`](example_generated_report_hms_high_01.md)

---

## Pipeline overview

```
HMS row / case profile
        |
        v
CatBoost probability + TreeSHAP XAI card
        |
        v
Query Planning Agent  ->  JSON question plan (+ retrieval queries)
        |
        v
Contextual hybrid RAG (K=5)  ->  frozen context passages per question
        |
        v
Report / Answer Agent  ->  grounded answers (reference + candidates)
        |
        v
Safety / schema checks (controlled pipeline) + RAGAS answer metrics
```

---

## Stage 1 - Case identity and screening prediction

| Field | Value |
|---|---|
| `case_id` | `hms_high_01` |
| `stratum` | `high` |
| `test_index` (HMS test split row) | `477` |
| `prediction_probability` | **0.9795** |
| `decision_threshold` | 0.5 |
| `risk_label` | `elevated_screening_risk` |
| Calibration | `{"calibration_status": "calibrated", "near_threshold": false}` |

**Interpretation circulated downstream:** elevated depression/anxiety **screening** risk (not a diagnosis); probability far above threshold; not near-threshold.

---

## Stage 2 - Student feature levels (de-identified profile)

These categorical levels are what the Query Planning Agent sees as `student_profile.feature_levels`:

| Feature | Level |
|---|---|
| `age` | high |
| `gender_male` | unknown |
| `gender_female` | low |
| `international` | low |
| `fincur` | low |
| `finpast` | medium |
| `food_worry` | high |
| `sleep_wknight` | low |
| `sleep_wkend` | medium |
| `exerc` | unknown |
| `belong1` | high |
| `belong2` | high |
| `lonely` | medium |
| `aca_impa` | high |
| `flourish` | low |
| `positiveMH` | low |
| `sub_any` | low |
| `binge1pl` | low |
| `smok_cig` | unknown |
| `discrim_race` | low |
| `discrim_gender` | low |
| `discrim_sexual` | low |
| `inst_type` | low |
| `inst_size` | low |
| `inst_public` | medium |

---

## Stage 3 - Local XAI (TreeSHAP contributors)

### Risk-increasing contributors (top)

| Feature | Display name | SHAP contribution | Direction | Level note |
|---|---|---:|---|---|
| `aca_impa` | Academic impairment | 1.2582 | increases predicted screening risk | This feature contributed positively to the model's risk prediction (level: high). |
| `flourish` | Flourishing | 0.7275 | increases predicted screening risk | This feature contributed positively to the model's risk prediction (level: low). |
| `fincur` | Current financial situation | 0.4153 | increases predicted screening risk | This feature contributed positively to the model's risk prediction (level: low). |
| `lonely` | Loneliness | 0.3445 | increases predicted screening risk | This feature contributed positively to the model's risk prediction (level: medium). |
| `food_worry` | Food insecurity worry | 0.3241 | increases predicted screening risk | This feature contributed positively to the model's risk prediction (level: high). |

### Protective / risk-decreasing contributors (top)

| Feature | Display name | SHAP contribution | Direction |
|---|---|---:|---|
| `inst_type` | Institution type | -0.0289 | decreases predicted screening risk |
| `age` | Age | -0.0274 | decreases predicted screening risk |
| `finpast` | Past financial stress | -0.0144 | decreases predicted screening risk |
| `sub_any` | Substance use | -0.0132 | decreases predicted screening risk |
| `discrim_gender` | Gender discrimination | -0.0124 | decreases predicted screening risk |

**Language rule enforced in prompts:** contributors are statistical associations, **not causes**.

---

## Stage 4 - Query Planning Agent output (gemma4)

- Plan parse OK: `True`
- Timestamp (UTC): `2026-07-01T19:16:11.839783+00:00`
- Planning strategy: `high_risk`
- Number of questions: **6**

### Self-check flags (agent introspection)

```json
{
  "risk_interpretation_included": true,
  "all_major_xai_factors_covered": true,
  "protective_factors_covered": true,
  "uncertainty_or_limitation_included": true,
  "safety_question_included": true,
  "no_diagnostic_language": true,
  "no_treatment_advice": true,
  "no_causal_xai_language": true,
  "non_redundant_questions": true
}
```

### Question plan (full circulated questions + retrieval queries)

#### Q1 - `expert_support_consideration` (priority: high)

**Question:** Given the high prediction probability (0.9795), what are the established clinical guidelines for interpreting an elevated depression/anxiety screening risk score in a non-diagnostic, expert-facing context?

**Linked XAI factor:** `None`

**Retrieval query:** `clinical guidelines interpretation of high depression anxiety screening risk`

**Expected evidence:** ['Recommended follow-up steps for elevated screening risk.', 'Distinction between screening risk and clinical diagnosis in student mental health.']

**Rationale:** Mandatory question for a high-risk case to guide the expert review process and ensure appropriate next steps are suggested, rather than just stating the score.

#### Q2 - `xai_contributor_evidence` (priority: high)

**Question:** What is the current scientific literature linking academic impairment (aca_impa) to elevated screening risk for depression and anxiety in young adults?

**Linked XAI factor:** `aca_impa`

**Retrieval query:** `academic impairment correlation with student mental health outcomes`

**Expected evidence:** ['Evidence supporting the link between academic stress/impairment and depressive symptoms.', "Contextualizing 'academic impairment' as a risk feature."]

**Rationale:** This is the single largest positive contributor (1.2582) and requires deep evidence to justify its weight in the report.

#### Q3 - `xai_contributor_evidence` (priority: high)

**Question:** How does food insecurity worry contribute to elevated screening risk for depression/anxiety, and what are the recommended psychoeducational resources?

**Linked XAI factor:** `food_worry`

**Retrieval query:** `impact of food insecurity on mental health outcomes in students`

**Expected evidence:** ['Mechanism by which resource scarcity (like food worry) affects mood.', 'Evidence-based psychoeducational interventions for food insecurity.']

**Rationale:** Food worry is a high contributor and represents a critical socioeconomic stressor that must be contextualized in the report.

#### Q4 - `protective_factor` (priority: medium)

**Question:** What are the protective mechanisms associated with institutional type (inst_type) and age, and how can these factors be highlighted to provide a balanced view of the student's resilience?

**Linked XAI factor:** `age, inst_type`

**Retrieval query:** `protective role of age and institution type in mental health screening`

**Expected evidence:** ['Literature on developmental resilience factors related to age.', 'How institutional support structures mitigate risk.']

**Rationale:** It is crucial to include protective factors (age, institution type) to prevent the report from being solely focused on risk, providing a balanced view.

#### Q5 - `uncertainty_limitation` (priority: medium)

**Question:** What are the known limitations of machine learning models (e.g., CatBoost) when applied to complex human behavioral data, and how should these limitations be communicated in an expert report?

**Linked XAI factor:** `None`

**Retrieval query:** `limitations of ML models in mental health screening`

**Expected evidence:** ['Bias or over-reliance risks associated with predictive modeling.', 'Best practices for communicating model uncertainty to clinicians.']

**Rationale:** Even when calibrated, acknowledging model limitations is essential for responsible expert reporting.

#### Q6 - `safety_communication` (priority: high)

**Question:** What are the non-diagnostic communication strategies and immediate safety protocols required when presenting an elevated screening risk to a student or supervising professional?

**Linked XAI factor:** `None`

**Retrieval query:** `non-diagnostic communication for high mental health screening risk`

**Expected evidence:** ['Scripting/language guidelines avoiding diagnostic language.', 'Immediate safety assessment protocols (e.g., suicide risk screening).']

**Rationale:** Mandatory question to ensure the report adheres strictly to non-diagnostic, safe communication standards.

### Level-2 LLM judge scores for this plan (gemma4 x gpt-oss:20b)

```json
{
  "relevance": 5,
  "xai_alignment": 5,
  "kdb_answerability": 5,
  "report_usefulness": 5,
  "probability_adaptation": 5,
  "safety": 5,
  "clarity": 5,
  "non_redundancy": 5
}
```

---

## Stage 5 - Contextual RAG retrieval (frozen contexts, K=5)

For each planned question, the system retrieves **5** passages (Ollama-context hybrid) and freezes them so every answer model sees the **same evidence**.

### Q1 retrieval package

- Retrieval query used: `clinical guidelines interpretation of high depression anxiety screening risk`
- Metadata: `{"case_id": "hms_high_01", "qid": "Q1", "retrieval_query": "clinical guidelines interpretation of high depression anxiety screening risk", "condition": "ollama_context", "mode": "hybrid", "top_k": 5, "question": "Given the high prediction probability (0.9795), what are the established clinical guidelines for interpreting an elevated depression/anxiety screening risk score in a non-diagnostic, expert-facing context?"}`
- Passages returned: **5**

**[1]** source=`DOC_A001` score=`0.909768`

> DEPRESSION SCREENING IN COLLEGE STUDENTS 6 injury, and possibly suicide (Pedrelli el al., 2015). Mental illnesses such as depression has been associated with significant morbidity, mortality, disability, and increased risk of suicide (Khubchandani, Brey, Kotecki, Kleinfelder, & Anderson, 2016). Efforts to increase awareness and screening of depression among college students should be a priority. Local Problem An ide…

**[2]** source=`DOC_G001` score=`0.854439`

> after screening. Secondary outcomes include patient engagement in mental healthcare, professional depression care and cost- effectiveness. According to a statistical analysis plan, the primary endpoint of all randomised patients will be analysed regarding the intention- to- treat principle. Ethics and dissemination The Ethics Committee of the Hamburg Medical Association approved the study. A clinical trial company w…

**[3]** source=`DOC_A001` score=`0.849269`

> DEPRESSION SCREENING IN COLLEGE STUDENTS 2 T able of Contents Abstract ………………………………………………………………………………………... 3-4 Problem Statement …………………………………………………………………………….. 5-6 Local Problem ………………………………………………………………………………….... 6 Clinical Question ………………………………………………………………………………... 6 Specific Aims ……………………………………………………………………………………. 7 Search Methods ……………………………………………………………………………….. 7-8 Available Knowledge/Review of Literature ……………

**[4]** source=`DOC_A001` score=`0.825295`

> DEPRESSION SCREENING IN COLLEGE STUDENTS 5 The Importance of Screening and Early Identification of Depression in College Students In the United States, more than 33.3% of first-year college students reported symptoms that are consistent with a diagnosable mental health disorder, according to the American Psychological Association (Auerbach et al., 2018). Mental health problems among college students have consistentl…

**[5]** source=`DOC_G001` score=`0.813705`

> primary care suffers from depression, which often remains undetected. Evidence regarding the efficacy of depression screening in primary care, however, is inconsistent. A previous single- centre randomised controlled trial (RCT) in cardiac patients, the DEPSCREEN- INFO trial, provided the first evidence that written feedback to patients following a positive depression screening reduces depression severity and leads …

### Q2 retrieval package

- Retrieval query used: `academic impairment correlation with student mental health outcomes`
- Metadata: `{"case_id": "hms_high_01", "qid": "Q2", "retrieval_query": "academic impairment correlation with student mental health outcomes", "condition": "ollama_context", "mode": "hybrid", "top_k": 5, "question": "What is the current scientific literature linking academic impairment (aca_impa) to elevated screening risk for depression and anxiety in young adults?"}`
- Passages returned: **5**

**[1]** source=`DOC_B001` score=`0.921851`

> mental health outcomes at time 2 were defined as positive screens for clinically significant depressive and anxiety symptoms measured by the PHQ-9 and GAD-7 examined as binary outcomes using the established cut-off score of 10. Self-harm, suicidal ideation and attempts were measured as binary outcomes (yes/no) to ques- tions from the Columbia Suicide Severity Rating Scale: 21 ‘have you ever harmed yourself without t…

**[2]** source=`DOC_D003` score=`0.871499`

> 1 Mental Health and Academic Success in College (revised and resubmitted to the B.E. Journal of Economic Analysis & Policy) May 2009 (first version January 2009) Daniel Eisenberg (University of Michigan, daneis@umich.edu) Ezra Golberstein (Harvard Medical School, golberstein@hcp.med.harvard.edu) Justin Hunt (University of Michigan, jbhunt@med.umich.edu) Abstract Mental health problems represent a potentially importa…

**[3]** source=`DOC_D003` score=`0.850471`

> negativ e affect per se (feeling depressed or hopeless) is not independently associated with a lower GPA. Finally, in fixed effects (within-person) regr essions of GPA on mental health variables using the longitudinal sample, we find again that co-occurring depressi on and anxiety are associated with lower GPA, and we also find that symptoms of eating disorders are associated with lower GPA. This study is best chara…

**[4]** source=`DOC_D003` score=`0.829366`

> 3 Americans are inundated with messages about success—in school, in a profession, in parenting, in relationships—without appreciating that successful performance rests on a foundation of mental health. United States Surgeon General’s Report on Mental Health, 1999 1. INTRODUCTION Among children and adolescents in the United States, mental disorders are estimated to account for a larger burden of disease, as measured …

**[5]** source=`DOC_B001` score=`0.824523`

> Predictors of mental health and academic outcomes in first-year university students: Identifying prevention and early-intervention targets A. Duffy, C. Keown-Stoneman, S. Goodday, J. Horrocks, M. Lowe, N. King, W. Pickett, S. H. McNevin, S. Cunningham, D. Rivera, L. Bisdounis, C. R. Bowie, K. Harkness and K. E. A. Saunders Background Although there is growing interest in mental health problems in university students…

### Q3 retrieval package

- Retrieval query used: `impact of food insecurity on mental health outcomes in students`
- Metadata: `{"case_id": "hms_high_01", "qid": "Q3", "retrieval_query": "impact of food insecurity on mental health outcomes in students", "condition": "ollama_context", "mode": "hybrid", "top_k": 5, "question": "How does food insecurity worry contribute to elevated screening risk for depression/anxiety, and what are the recommended psychoeducational resources?"}`
- Passages returned: **5**

**[1]** source=`DOC_E001` score=`0.989877`

> 6 6 Below are other key findings we explore throughout this report: in of students experience basic needs insecurity related to food and/or housing. 67% of parenting students experience housing insecurity. 60% of Black students experience housing insecurity. 66% of LGBTQIA+ students experience depression and/or anxiety. Indigenous students experience homelessness, which is twice as high as the rate for White student…

**[2]** source=`DOC_E001` score=`0.957124`

> 7 7 36% of students involved with the carceral system experience homelessness. 72% of former foster youth experience housing insecurity. 62% of students with disabilities experience basic needs insecurity related to food and/or housing—13 percentage points higher than students without disabilities (56%). The link between student basic needs security and persistence is unequivocal: 52% of Pell Grant recipients experi…

**[3]** source=`DOC_E001` score=`0.940585`

> 5 5 Mom, can you pick me up from school? 73% of respondents face basic needs insecurity when we consider other basic needs (mental health, child care, transportation, and internet/ technology access) in addition to food and housing. 44% of students experience clinically significant symptoms of anxiety and/or depression. 57% of respondents who had previously stopped out (stopped attending college without completing a…

**[4]** source=`DOC_E001` score=`0.9098`

> 4 This report shares findings from 91 colleges and universities across 16 states that participated in Hope Impact Partnerships (HIP) and fielded The Hope Center Student Basic Needs Survey between Spring 2023 and Summer 2024. 1 4 Executive Summary Among 74,350 survey participants we found that: 41% experience food insecurity. 48% experience housing insecurity. 14% experience homelessness. 59% of students experience a…

**[5]** source=`DOC_E001` score=`0.87055`

> 3 Table of Contents Executive Summary Defining Basic Needs Security What is The Hope Center Student Basic Needs Survey? · Participants Food Insecurity, Housing Insecurity, and Homelessness · Racial Disparities in Standard Measures of Basic Needs Insecurity · Food Insecurity · Housing Insecurity and Homelessness Beyond Food and Housing · Mental Health · Transportation, Internet/Technology, and Childcare Insecurity In…

### Q4 retrieval package

- Retrieval query used: `protective role of age and institution type in mental health screening`
- Metadata: `{"case_id": "hms_high_01", "qid": "Q4", "retrieval_query": "protective role of age and institution type in mental health screening", "condition": "ollama_context", "mode": "hybrid", "top_k": 5, "question": "What are the protective mechanisms associated with institutional type (inst_type) and age, and how can these factors be highlighted to provide a balanced view of the student's resilience?"}`
- Passages returned: **5**

**[1]** source=`DOC_A007` score=`0.919812`

> 3 School mental health screening is defined as the use of a systematic tool or process to identify the strengths and needs of students. Screening is conducted for all students, not just students identified as being at risk for or already displaying mental health concerns. This might involve screening an entire population, such as a school’s student body, or a smaller subset of a population, such as a specific grade …

**[2]** source=`DOC_A007` score=`0.891692`

> 4 Identify concerns specific to certain grades or classrooms • Screening data can also shed light on strengths and concerns specific to certain subsets of a school’s population, such as a single grade or age group. Understanding these possible trends can be critical for providing equitable supports within a school. • For example, a school-wide screening effort may reveal that several classrooms may have a significan…

**[3]** source=`DOC_A007` score=`0.88272`

> students • More acceptable to youth and families across cultural groups due to varying perspectives on mental health. Research suggests that students who experience average to high levels of well-be- ing also experience better current and long-term outcomes, including academic success, than students who report low levels of well-being. These findings remain true regardless of level of psychological distress and impa…

**[4]** source=`DOC_A007` score=`0.852451`

> • With a multi-tiered approach, students are more likely to learn core social- emotional-behavioral skills and may have their mental health needs addressed before they escalate. Inform Prevention and Early Intervention Strategies • About 20% of students show signs and symptoms of a mental health disorder in a given year. Mental health screening is a proactive approach to gathering valuable information for planning a…

**[5]** source=`DOC_A007` score=`0.834206`

> 2 School Mental Health Quality Guide: School Mental Health Screening is part of a collection of resources developed by the National Center for School Mental Health (NCSMH) at the University of Maryland School of Medicine for The SHAPE System. The Quality Guides provide guidance to help school mental health systems advance the quality of their services and supports. This guide contains background information on schoo…

### Q5 retrieval package

- Retrieval query used: `limitations of ML models in mental health screening`
- Metadata: `{"case_id": "hms_high_01", "qid": "Q5", "retrieval_query": "limitations of ML models in mental health screening", "condition": "ollama_context", "mode": "hybrid", "top_k": 5, "question": "What are the known limitations of machine learning models (e.g., CatBoost) when applied to complex human behavioral data, and how should these limitations be communicated in an expert report?"}`
- Passages returned: **5**

**[1]** source=`DOC_A001` score=`0.934406`

> DEPRESSION SCREENING IN COLLEGE STUDENTS 6 injury, and possibly suicide (Pedrelli el al., 2015). Mental illnesses such as depression has been associated with significant morbidity, mortality, disability, and increased risk of suicide (Khubchandani, Brey, Kotecki, Kleinfelder, & Anderson, 2016). Efforts to increase awareness and screening of depression among college students should be a priority. Local Problem An ide…

**[2]** source=`DOC_A007` score=`0.847526`

> students • More acceptable to youth and families across cultural groups due to varying perspectives on mental health. Research suggests that students who experience average to high levels of well-be- ing also experience better current and long-term outcomes, including academic success, than students who report low levels of well-being. These findings remain true regardless of level of psychological distress and impa…

**[3]** source=`DOC_A007` score=`0.837005`

> 3 School mental health screening is defined as the use of a systematic tool or process to identify the strengths and needs of students. Screening is conducted for all students, not just students identified as being at risk for or already displaying mental health concerns. This might involve screening an entire population, such as a school’s student body, or a smaller subset of a population, such as a specific grade …

**[4]** source=`DOC_A007` score=`0.835985`

> 2 School Mental Health Quality Guide: School Mental Health Screening is part of a collection of resources developed by the National Center for School Mental Health (NCSMH) at the University of Maryland School of Medicine for The SHAPE System. The Quality Guides provide guidance to help school mental health systems advance the quality of their services and supports. This guide contains background information on schoo…

**[5]** source=`DOC_A007` score=`0.835059`

> procedures with small groups of students is helpful to best understand and adapt screening procedures. Starting small and scaling up to the entire student population in a gradual way allows school and district teams to make course corrections and build on successes. For example, many school district leaders have pointed to the usefulness of piloting a screener with a few students and getting feedback or starting wit…

### Q6 retrieval package

- Retrieval query used: `non-diagnostic communication for high mental health screening risk`
- Metadata: `{"case_id": "hms_high_01", "qid": "Q6", "retrieval_query": "non-diagnostic communication for high mental health screening risk", "condition": "ollama_context", "mode": "hybrid", "top_k": 5, "question": "What are the non-diagnostic communication strategies and immediate safety protocols required when presenting an elevated screening risk to a student or supervising professional?"}`
- Passages returned: **5**

**[1]** source=`DOC_A007` score=`0.935974`

> students • More acceptable to youth and families across cultural groups due to varying perspectives on mental health. Research suggests that students who experience average to high levels of well-be- ing also experience better current and long-term outcomes, including academic success, than students who report low levels of well-being. These findings remain true regardless of level of psychological distress and impa…

**[2]** source=`DOC_A007` score=`0.896564`

> 3 School mental health screening is defined as the use of a systematic tool or process to identify the strengths and needs of students. Screening is conducted for all students, not just students identified as being at risk for or already displaying mental health concerns. This might involve screening an entire population, such as a school’s student body, or a smaller subset of a population, such as a specific grade …

**[3]** source=`DOC_A007` score=`0.887545`

> 6 9 Include students and families in informing the screening, planning, and implementation process. 9 Use a selection process for a screening tool that considers reliability, feasibility, cost, and fit with the goals of screening. 9 Share information about screening in multiple formats prior to implementation. 9 Inform students and families about screening procedures in advance and offer the opportunity to consent o…

**[4]** source=`DOC_A001` score=`0.869612`

> DEPRESSION SCREENING IN COLLEGE STUDENTS 6 injury, and possibly suicide (Pedrelli el al., 2015). Mental illnesses such as depression has been associated with significant morbidity, mortality, disability, and increased risk of suicide (Khubchandani, Brey, Kotecki, Kleinfelder, & Anderson, 2016). Efforts to increase awareness and screening of depression among college students should be a priority. Local Problem An ide…

**[5]** source=`DOC_A007` score=`0.86129`

> • With a multi-tiered approach, students are more likely to learn core social- emotional-behavioral skills and may have their mental health needs addressed before they escalate. Inform Prevention and Early Intervention Strategies • About 20% of students show signs and symptoms of a mental health disorder in a given year. Mental health screening is a proactive approach to gathering valuable information for planning a…

---

## Stage 6 - Grounded answers (identical contexts)

Below: **reference** (`gpt-oss:20b`) vs **candidate** (`gemma4`) for each question.

### Q1

**Question:** Given the high prediction probability (0.9795), what are the established clinical guidelines for interpreting an elevated depression/anxiety screening risk score in a non-diagnostic, expert-facing context?

**n_contexts:** 5

**Reference answer (gpt-oss:20b):**

A probability of 0.9795 indicates a positive screen for depression or anxiety and should be treated as an alert rather than a formal diagnosis. In practice, the next step is to administer a more comprehensive instrument—such as the PHQ‑9 if the initial screen was a PHQ‑2—to quantify symptom severity (Context 1). If the PHQ‑9 score meets or exceeds the threshold of 10, clinicians should consider a brief structured assessment and discuss treatment options while arranging referral to counseling services available at the college health center or primary care practice (Contexts 1 and 4). Evidence from the DEPSCREEN‑INFO trial shows that providing written, patient‑targeted feedback after a positive screen can reduce depression severity and increase engagement in mental health care; similar interventions are being evaluated in the GET.FEEDBACK.GP RCT (Context 5). Thus, an elevated risk score s…

**Candidate answer (gemma4):**

The retrieved evidence does not provide general established clinical guidelines for interpreting an arbitrary high prediction probability score in a non-diagnostic setting. However, specific protocols are detailed based on screening tools and settings. For college students who screen positive using the PHQ-2, the recommended approach is further evaluation with the PHQ-9, followed by possible referral to counseling services (Context 1). In primary care settings utilizing the PHQ-9, a positive result (score $\geq$10) initiates participation in studies that test targeted feedback interventions for patients and general practitioners (Context 5). These protocols emphasize moving beyond initial screening through further assessment and structured follow-up care.

### Q2

**Question:** What is the current scientific literature linking academic impairment (aca_impa) to elevated screening risk for depression and anxiety in young adults?

**n_contexts:** 5

**Reference answer (gpt-oss:20b):**

Current longitudinal studies consistently show a statistical association between poorer academic performance and higher rates of clinically significant depressive and anxiety symptoms in young adults. In a large U.S. university sample, lower cumulative GPA was significantly predicted by positive depression screens and even more strongly when anxiety co‑occurred (Eisenberg et al., 2009; see Context 2–3). The same cohort found that students who reported depressive or anxious symptoms at the start of their first year were more likely to experience academic impairment—measured as lower GPA and increased dropout risk—by year’s end, with these mental‑health factors accounting for a sizable portion of the variance in academic outcomes (Context 5). Additionally, a study using PHQ‑9 and GAD‑7 scores found that students who screened positive for depression or anxiety reported greater functional i…

**Candidate answer (gemma4):**

The scientific literature suggests a consistent association between elevated screening risk for depression and anxiety and lower academic outcomes in young adults. Specifically, studies have found that depression is a significant predictor of both lower Grade Point Average (GPA) and an increased probability of dropping out during college years. This link is further supported by findings indicating that co-occurring symptoms of depression and anxiety are associated with reduced GPA when analyzed using within-person fixed effects models. Furthermore, research has shown that general depressive and anxious symptoms were associated with poorer academic outcomes among university students. These studies characterize the relationship as an association between mental health status and academic success, rather than establishing definitive causal links.

### Q3

**Question:** How does food insecurity worry contribute to elevated screening risk for depression/anxiety, and what are the recommended psychoeducational resources?

**n_contexts:** 5

**Reference answer (gpt-oss:20b):**

Food insecurity is a prominent component of basic‑needs insecurity—41 % of surveyed students report it—and it co‑occurs with clinically significant anxiety or depression symptoms in roughly 44 % of respondents, indicating that worry about adequate food supply contributes statistically to elevated mental‑health risk. The data show that students experiencing food insecurity often cite emotional stress and mental‑health concerns as reasons for academic disruption, underscoring its role as a contributor to depressive and anxious symptomatology. While the report does not enumerate specific psychoeducational materials, it implies that institutions typically address these risks through campus counseling services, financial‑aid guidance, and on‑campus food‑pantry or meal‑plan assistance programs. For students identified at higher risk, referral to qualified mental‑health support and information…

**Candidate answer (gemma4):**

The retrieved evidence indicates that basic needs insecurity, which includes food and housing concerns, is highly prevalent among various student populations, but it does not describe the specific mechanism by which worry about food insecurity contributes to an elevated screening risk for depression or anxiety. While the context notes that mental health issues are a significant factor leading students to stop out of college, no direct link between food insecurity worry and increased symptoms of anxiety/depression is provided. Furthermore, the provided passages do not contain any recommended psychoeducational resources for these concerns.

### Q4

**Question:** What are the protective mechanisms associated with institutional type (inst_type) and age, and how can these factors be highlighted to provide a balanced view of the student's resilience?

**n_contexts:** 5

**Reference answer (gpt-oss:20b):**

The evidence indicates that school mental‑health screening can reveal strengths and well‑being across the entire student body, including by grade or age group (Context 2). Screening data may show that certain classrooms or age cohorts have higher levels of resilience indicators such as life satisfaction and school belonging, which could be interpreted as protective mechanisms linked to developmental stage. However, none of the retrieved passages explicitly identify institutional type (inst_type) as a contributor to resilience; thus no evidence is available to describe protective factors tied to different school settings. To present a balanced view, highlight that screening can surface both strengths and needs for all students, allowing educators to tailor supports that reinforce identified protective factors while addressing gaps. This approach underscores the importance of early identi…

**Candidate answer (gemma4):**

The provided context does not contain information regarding protective mechanisms associated with institutional type ($\text{inst\_type}$). However, screening data can shed light on strengths and concerns specific to subsets of a population, such as certain grade or age groups, which is critical for providing equitable supports. To provide a balanced view of resilience, it is beneficial to systematically gather information about indicators of well-being and social-emotional strengths alongside distress levels. Research suggests that students who report average to high levels of well-being experience better current and long-term outcomes, including academic success, regardless of their level of psychological distress or impairment. Analyzing these trends at the group level can help prioritize supports and inform targeted prevention strategies for a school or community.

### Q5

**Question:** What are the known limitations of machine learning models (e.g., CatBoost) when applied to complex human behavioral data, and how should these limitations be communicated in an expert report?

**n_contexts:** 5

**Reference answer (gpt-oss:20b):**

The retrieved passages focus on depression and school mental‑health screening practices but do not mention machine‑learning algorithms such as CatBoost or describe their performance characteristics. Consequently, there is no evidence in the supplied material to identify specific limitations of these models when applied to complex human behavioral data, nor guidance on how to report them. If an expert report were to discuss ML model constraints, it would need to draw upon additional sources that detail issues such as overfitting, interpretability challenges, bias from uneven training data, and the difficulty of capturing nuanced psychosocial variables—none of which are covered in the current evidence set.

**Candidate answer (gemma4):**

The provided context passages do not contain any information regarding the known limitations of machine learning models, such as CatBoost, when applied to complex human behavioral data. Therefore, it is not possible to advise on how these specific technical limitations should be communicated within an expert report based solely on the retrieved evidence.

### Q6

**Question:** What are the non-diagnostic communication strategies and immediate safety protocols required when presenting an elevated screening risk to a student or supervising professional?

**n_contexts:** 5

**Reference answer (gpt-oss:20b):**

Present the screening result using a strengths‑based, non‑diagnostic tone—emphasize early support rather than labeling and involve the student (and family when appropriate) in discussing next steps, as recommended for comprehensive school mental health screening (Context 1).  
Immediately respond to any indication of self‑harm or harm to others by conducting a rapid safety assessment and arranging urgent referral to school or community mental health services; this is required to ensure safety (Context 1).  
Use a defined triage process to determine whether the student requires Tier 2 or Tier 3 supports, ensuring that referrals are made promptly and documented (Context 3).  
Communicate findings with empathy, focusing on available resources rather than diagnostic categories, which aligns with the goal of early identification and prevention outlined in multi‑tiered systems (Context 5).

**Candidate answer (gemma4):**

Non-diagnostic communication strategies require involving students and families in the planning, implementation, and informing processes by sharing information about screening procedures in multiple formats prior to the effort [Context 3]. It is recommended that these communications offer the opportunity for consent or opt-out. Regarding immediate safety protocols, any risk of harm to self or others must be addressed immediately, requiring a defined process to assess screening results and triage students who need further evaluation [Context 1, Context 3]. Furthermore, screening teams are responsible for ensuring that any student identified as potentially dangerous is assessed further without delay to ensure their safety [Context 1].

---

## Stage 7 - RAGAS scores for this case (per question x model)

| model | qid | question_type | faithfulness | answer_relevancy | answer_correctness | semantic_similarity |
|---|---|---|---|---|---|---|
| gemma4 | Q1 | expert_support_consideration | 0.778 | 0.636 | 0.658 | 0.916 |
| gemma4 | Q2 | xai_contributor_evidence | 1.0 | 0.748 | 0.616 | 0.964 |
| gemma4 | Q3 | xai_contributor_evidence | 1.0 | 0.775 | 0.4 | 0.932 |
| gemma4 | Q4 | protective_factor | 0.833 | 0.619 | 0.523 | 0.947 |
| gemma4 | Q5 | uncertainty_limitation | 1.0 | 0.0 | 0.639 | 0.919 |
| gemma4 | Q6 | safety_communication | 1.0 | 0.766 | 0.548 | 0.907 |
| llama3 | Q1 | expert_support_consideration | 0.4 | 0.808 | 0.697 | 0.94 |
| llama3 | Q2 | xai_contributor_evidence | 0.667 | 0.726 | 0.699 | 0.95 |
| llama3 | Q3 | xai_contributor_evidence | 0.333 | 0.752 | 0.438 | 0.963 |
| llama3 | Q4 | protective_factor | 0.438 | 0.755 | 0.362 | 0.961 |
| llama3 | Q5 | uncertainty_limitation | 0.0 | 0.782 | 0.228 | 0.91 |
| llama3 | Q6 | safety_communication | 0.462 | 0.812 | 0.769 | 0.935 |
| qwen2.5 | Q1 | expert_support_consideration | 0.778 | 0.66 | 0.755 | 0.932 |
| qwen2.5 | Q2 | xai_contributor_evidence | 0.6 | 0.758 | 0.612 | 0.948 |
| qwen2.5 | Q3 | xai_contributor_evidence | 0.125 | 0.73 | 0.394 | 0.944 |
| qwen2.5 | Q4 | protective_factor | 0.5 | 0.627 | 0.441 | 0.938 |
| qwen2.5 | Q5 | uncertainty_limitation | 0.0 | 0.784 | 0.228 | 0.911 |
| qwen2.5 | Q6 | safety_communication | 0.833 | 0.794 | 0.688 | 0.878 |
| phi4 | Q1 | expert_support_consideration | 0.786 | 0.754 | 0.959 | 0.949 |
| phi4 | Q2 | xai_contributor_evidence | 0.636 | 0.731 | 0.542 | 0.969 |
| phi4 | Q3 | xai_contributor_evidence | 0.35 | 0.702 | 0.648 | 0.969 |
| phi4 | Q4 | protective_factor | 0.4 | 0.688 | 0.47 | 0.959 |
| phi4 | Q5 | uncertainty_limitation | 0.111 | 0.713 | 0.224 | 0.898 |
| phi4 | Q6 | safety_communication | 0.667 | 0.845 | 0.734 | 0.935 |
| mistral | Q1 | expert_support_consideration | 0.583 | 0.656 | 0.769 | 0.934 |
| mistral | Q2 | xai_contributor_evidence | 0.929 | 0.731 | 0.602 | 0.97 |
| mistral | Q3 | xai_contributor_evidence | 0.308 | 0.75 | 0.376 | 0.96 |
| mistral | Q4 | protective_factor | 0.882 | 0.704 | 0.564 | 0.924 |
| mistral | Q5 | uncertainty_limitation | 0.45 | 0.724 | 0.223 | 0.89 |
| mistral | Q6 | safety_communication | 0.933 | 0.882 | 0.654 | 0.934 |

---

## Stage 8 - What a controlled report agent would assemble

In the controlled agentic pipeline, answers + XAI + safety rules are composed into a structured **non-clinical screening advisory** JSON with fields such as:

- `screening_summary`
- `probability_interpretation`
- `xai_explanation`
- `evidence_grounded_interpretation`
- `non_clinical_recommendation`
- `limitations`
- `citations` / safety status

Illustrative assembled summary for this case (derived from circulated fields above, not a new LLM call):

> **Screening summary.** Elevated depression/anxiety screening signal (predicted probability ≈ 0.98; threshold 0.5). This is a screening support signal for expert review, not a diagnosis.
>
> **XAI explanation.** Top statistical contributors included academic impairment, low flourishing, financial strain, loneliness, and food-insecurity worry. Protective contributors were comparatively small in magnitude.
>
> **Evidence-grounded notes.** Literature passages retrieved for the planned questions discuss academic impairment, flourishing, financial stress, loneliness, and non-diagnostic communication practices (see Stage 5 contexts).
>
> **Non-clinical recommendation.** Consider referral pathways to qualified campus/community support; avoid diagnostic or treatment language.
>
> **Limitations.** Survey-based screening model; SHAP contributions are associative; retrieved evidence may be incomplete; local LLM outputs vary by model version.

---

## Data-flow checklist (everything that circulated)

| Stage | Object | Source file |
|---|---|---|
| 1–3 | Structured case + XAI | `query_agent_eval/data/case_profiles_50.jsonl` (`hms_high_01`) |
| 4 | Query plan | `query_agent_eval/outputs/plans/gemma4/hms_high_01.json` |
| 4 | Plan judge | `query_agent_eval/outputs/judge/gemma4/hms_high_01.json` |
| 5 | Frozen contexts Q1–Q6 | `report_agent_eval/outputs/contexts/hms_high_01/Q*.json` |
| 6 | Reference answers | `report_agent_eval/outputs/answers/gpt-oss_20b/hms_high_01.json` |
| 6 | Candidate answers | `report_agent_eval/outputs/answers/gemma4/hms_high_01.json` |
| 7 | RAGAS rows | `report_agent_eval/outputs/ragas_per_question.csv` (filtered) |
| Bundle copy | All of the above | `paper_outputs/end_to_end_case_hms_high_01_bundle/` |

---

## One-sentence takeaway for reviewers

For `hms_high_01`, the system circulated a de-identified feature profile, CatBoost probability **0.9795**, TreeSHAP risk/protective factors, a **6**-question gemma4 plan with retrieval queries, **K=5** frozen literature contexts per question, grounded reference/candidate answers on identical evidence, and per-question RAGAS scores — all without issuing a clinical diagnosis.
