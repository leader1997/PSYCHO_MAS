# Generated expert-facing report — elevated case `hms_high_01`

This file is the **complete generated report** for one elevated HMS screening case. In this pipeline the Report Agent does not emit a single narrative PDF; the report **is** the set of grounded answers to the Query Planning Agent’s six questions, each written from frozen retrieved evidence (K = 5).

**Case:** `hms_high_01` (elevated screening stratum)  
**Prediction:** probability **0.9795**, threshold **0.5**, label `elevated_screening_risk` (not near threshold)  
**Plan model:** `gemma4` (temperature 0.1), frozen plan  
**Candidate report (shown first):** `gemma4` (temperature 0.1)  
**Reference answers (comparison):** `gpt-oss:20b` (same questions, same frozen contexts)  
**Retrieval:** Ollama-context hybrid, K = 5 per question  
**Sources:** `query_agent_eval/outputs/plans/gemma4/hms_high_01.json`, `report_agent_eval/outputs/contexts/hms_high_01/`, `report_agent_eval/outputs/answers/gemma4/hms_high_01.json`, `report_agent_eval/outputs/answers/gpt-oss_20b/hms_high_01.json`, `report_agent_eval/outputs/ragas_per_question.csv`

**Language rule.** This is a **non-diagnostic screening support report** for expert review. It is not a diagnosis, treatment plan, or clinical chart note.

---

## 1. Case/XAI Card that conditioned the report

| Field | Value |
|---|---|
| `case_id` | `hms_high_01` |
| HMS test-split row | 477 |
| Screening probability | 0.9795 |
| Decision threshold | 0.5 |
| Risk label | elevated screening risk |
| Calibration | calibrated; `near_threshold = false` |

**Top TreeSHAP risk contributors**

| Feature | Display name | SHAP | Level |
|---|---|---:|---|
| `aca_impa` | Academic impairment | +1.2582 | high |
| `flourish` | Flourishing | +0.7275 | low |
| `fincur` | Current financial situation | +0.4153 | low |
| `lonely` | Loneliness | +0.3445 | medium |
| `food_worry` | Food insecurity worry | +0.3241 | high |

**Top protective contributors**

| Feature | Display name | SHAP |
|---|---|---:|
| `inst_type` | Institution type | −0.0289 |
| `age` | Age | −0.0274 |

---

## 2. Planned questions (Query Planning Agent)

Planning strategy: `high_risk`. Six questions, each with a retrieval query used to freeze K = 5 passages.

| ID | Type | Linked XAI | Question |
|---|---|---|---|
| Q1 | expert_support_consideration | — | Given the high prediction probability (0.9795), what are the established clinical guidelines for interpreting an elevated depression/anxiety screening risk score in a non-diagnostic, expert-facing context? |
| Q2 | xai_contributor_evidence | `aca_impa` | What is the current scientific literature linking academic impairment (aca_impa) to elevated screening risk for depression and anxiety in young adults? |
| Q3 | xai_contributor_evidence | `food_worry` | How does food insecurity worry contribute to elevated screening risk for depression/anxiety, and what are the recommended psychoeducational resources? |
| Q4 | protective_factor | `age`, `inst_type` | What are the protective mechanisms associated with institutional type (inst_type) and age, and how can these factors be highlighted to provide a balanced view of the student's resilience? |
| Q5 | uncertainty_limitation | — | What are the known limitations of machine learning models (e.g., CatBoost) when applied to complex human behavioral data, and how should these limitations be communicated in an expert report? |
| Q6 | safety_communication | — | What are the non-diagnostic communication strategies and immediate safety protocols required when presenting an elevated screening risk to a student or supervising professional? |

Retrieval queries: Q1 `clinical guidelines interpretation of high depression anxiety screening risk`; Q2 `academic impairment correlation with student mental health outcomes`; Q3 `impact of food insecurity on mental health outcomes in students`; Q4 `protective role of age and institution type in mental health screening`; Q5 `limitations of ML models in mental health screening`; Q6 `non-diagnostic communication for high mental health screening risk`.

---

## 3. Evidence sources (frozen retrieved passages)

Hybrid Ollama-context retrieval returned **K = 5** passages per question. Document IDs, chunk IDs, titles, and retrieval scores are archived in `report_agent_eval/outputs/contexts/hms_high_01/Q*.json`. The Report Agent received these passages as unlabeled `[Context 1]…[Context 5]` text blocks and was **not** required to emit inline citations; this table is the complementary evidence-source listing for the worked example.

### 3.1 Summary by question (unique documents)

| Question | Retrieval query | Unique `doc_id`s | Primary source titles |
|---|---|---|---|
| Q1 | clinical guidelines interpretation of high depression anxiety screening risk | `DOC_A001`, `DOC_G001` | Depression screening in college students; Depression screening using patient-targeted feedback |
| Q2 | academic impairment correlation with student mental health outcomes | `DOC_B001`, `DOC_D003` | Predictors of mental health and academic outcomes in first-year students; Mental Health and Academic Success in College (Eisenberg et al.) |
| Q3 | impact of food insecurity on mental health outcomes in students | `DOC_E001` | The Hope Center 2023–2024 Student Basic Needs Survey Report |
| Q4 | protective role of age and institution type in mental health screening | `DOC_A007` | School Mental Health Quality Guide: Screening |
| Q5 | limitations of ML models in mental health screening | `DOC_A001`, `DOC_A007` | Depression screening in college students; School Mental Health Quality Guide: Screening |
| Q6 | non-diagnostic communication for high mental health screening risk | `DOC_A007`, `DOC_A001` | School Mental Health Quality Guide: Screening; Depression screening in college students |

### 3.2 Full hit list (rank, doc, chunk, score)

| Q | Rank (= Context *i*) | `doc_id` | `chunk_id` | Score | Title |
|---|---:|---|---|---:|---|
| Q1 | 1 | `DOC_A001` | `DOC_A001_p6_c1` | 0.910 | Depression screening in college students |
| Q1 | 2 | `DOC_G001` | `DOC_G001_p1_c3` | 0.854 | Depression screening using patient-targeted feedback |
| Q1 | 3 | `DOC_A001` | `DOC_A001_p2_c1` | 0.849 | Depression screening in college students |
| Q1 | 4 | `DOC_A001` | `DOC_A001_p5_c1` | 0.825 | Depression screening in college students |
| Q1 | 5 | `DOC_G001` | `DOC_G001_p1_c2` | 0.814 | Depression screening using patient-targeted feedback |
| Q2 | 1 | `DOC_B001` | `DOC_B001_p2_c3` | 0.922 | Predictors of mental health and academic outcomes in first-year students |
| Q2 | 2 | `DOC_D003` | `DOC_D003_p1_c1` | 0.871 | Mental Health and Academic Success in College |
| Q2 | 3 | `DOC_D003` | `DOC_D003_p4_c2` | 0.850 | Mental Health and Academic Success in College |
| Q2 | 4 | `DOC_D003` | `DOC_D003_p3_c1` | 0.829 | Mental Health and Academic Success in College |
| Q2 | 5 | `DOC_B001` | `DOC_B001_p1_c1` | 0.825 | Predictors of mental health and academic outcomes in first-year students |
| Q3 | 1 | `DOC_E001` | `DOC_E001_p6_c1` | 0.990 | Hope Center 2023–2024 Student Basic Needs Survey Report |
| Q3 | 2 | `DOC_E001` | `DOC_E001_p7_c1` | 0.957 | Hope Center 2023–2024 Student Basic Needs Survey Report |
| Q3 | 3 | `DOC_E001` | `DOC_E001_p5_c1` | 0.941 | Hope Center 2023–2024 Student Basic Needs Survey Report |
| Q3 | 4 | `DOC_E001` | `DOC_E001_p4_c1` | 0.910 | Hope Center 2023–2024 Student Basic Needs Survey Report |
| Q3 | 5 | `DOC_E001` | `DOC_E001_p3_c1` | 0.871 | Hope Center 2023–2024 Student Basic Needs Survey Report |
| Q4 | 1 | `DOC_A007` | `DOC_A007_p3_c1` | 0.920 | School Mental Health Quality Guide: Screening |
| Q4 | 2 | `DOC_A007` | `DOC_A007_p4_c1` | 0.892 | School Mental Health Quality Guide: Screening |
| Q4 | 3 | `DOC_A007` | `DOC_A007_p4_c2` | 0.883 | School Mental Health Quality Guide: Screening |
| Q4 | 4 | `DOC_A007` | `DOC_A007_p3_c2` | 0.852 | School Mental Health Quality Guide: Screening |
| Q4 | 5 | `DOC_A007` | `DOC_A007_p2_c1` | 0.834 | School Mental Health Quality Guide: Screening |
| Q5 | 1 | `DOC_A001` | `DOC_A001_p6_c1` | 0.934 | Depression screening in college students |
| Q5 | 2 | `DOC_A007` | `DOC_A007_p4_c2` | 0.848 | School Mental Health Quality Guide: Screening |
| Q5 | 3 | `DOC_A007` | `DOC_A007_p3_c1` | 0.837 | School Mental Health Quality Guide: Screening |
| Q5 | 4 | `DOC_A007` | `DOC_A007_p2_c1` | 0.836 | School Mental Health Quality Guide: Screening |
| Q5 | 5 | `DOC_A007` | `DOC_A007_p6_c2` | 0.835 | School Mental Health Quality Guide: Screening |
| Q6 | 1 | `DOC_A007` | `DOC_A007_p4_c2` | 0.936 | School Mental Health Quality Guide: Screening |
| Q6 | 2 | `DOC_A007` | `DOC_A007_p3_c1` | 0.897 | School Mental Health Quality Guide: Screening |
| Q6 | 3 | `DOC_A007` | `DOC_A007_p6_c1` | 0.888 | School Mental Health Quality Guide: Screening |
| Q6 | 4 | `DOC_A001` | `DOC_A001_p6_c1` | 0.870 | Depression screening in college students |
| Q6 | 5 | `DOC_A007` | `DOC_A007_p3_c2` | 0.861 | School Mental Health Quality Guide: Screening |

**Note.** Rank *i* corresponds to `[Context i]` in the Report Agent user message. When an answer mentions “Context 1,” it refers to the rank-1 row for that question, not to a bibliographic citation ID.

---

## 4. Generated report (`gemma4` candidate) — full answers

Each block is the **verbatim** Report Agent answer. All six answers used **n_contexts = 5** and recorded `error = null`.

### Q1. Interpreting the elevated screening probability

**Type:** expert_support_consideration  
**Timestamp (UTC):** 2026-07-14T17:02:51

The retrieved evidence does not provide general established clinical guidelines for interpreting an arbitrary high prediction probability score in a non-diagnostic setting. However, specific protocols are detailed based on screening tools and settings. For college students who screen positive using the PHQ-2, the recommended approach is further evaluation with the PHQ-9, followed by possible referral to counseling services (Context 1). In primary care settings utilizing the PHQ-9, a positive result (score ≥10) initiates participation in studies that test targeted feedback interventions for patients and general practitioners (Context 5). These protocols emphasize moving beyond initial screening through further assessment and structured follow-up care.

### Q2. Academic impairment as a statistical contributor

**Type:** xai_contributor_evidence (`aca_impa`)  
**Timestamp (UTC):** 2026-07-14T17:03:11

The scientific literature suggests a consistent association between elevated screening risk for depression and anxiety and lower academic outcomes in young adults. Specifically, studies have found that depression is a significant predictor of both lower Grade Point Average (GPA) and an increased probability of dropping out during college years. This link is further supported by findings indicating that co-occurring symptoms of depression and anxiety are associated with reduced GPA when analyzed using within-person fixed effects models. Furthermore, research has shown that general depressive and anxious symptoms were associated with poorer academic outcomes among university students. These studies characterize the relationship as an association between mental health status and academic success, rather than establishing definitive causal links.

### Q3. Food insecurity worry

**Type:** xai_contributor_evidence (`food_worry`)  
**Timestamp (UTC):** 2026-07-14T17:03:33

The retrieved evidence indicates that basic needs insecurity, which includes food and housing concerns, is highly prevalent among various student populations, but it does not describe the specific mechanism by which worry about food insecurity contributes to an elevated screening risk for depression or anxiety. While the context notes that mental health issues are a significant factor leading students to stop out of college, no direct link between food insecurity worry and increased symptoms of anxiety/depression is provided. Furthermore, the provided passages do not contain any recommended psychoeducational resources for these concerns.

### Q4. Protective factors (institution type and age)

**Type:** protective_factor  
**Timestamp (UTC):** 2026-07-14T17:03:56

The provided context does not contain information regarding protective mechanisms associated with institutional type (inst_type). However, screening data can shed light on strengths and concerns specific to subsets of a population, such as certain grade or age groups, which is critical for providing equitable supports. To provide a balanced view of resilience, it is beneficial to systematically gather information about indicators of well-being and social-emotional strengths alongside distress levels. Research suggests that students who report average to high levels of well-being experience better current and long-term outcomes, including academic success, regardless of their level of psychological distress or impairment. Analyzing these trends at the group level can help prioritize supports and inform targeted prevention strategies for a school or community.

### Q5. Model limitations

**Type:** uncertainty_limitation  
**Timestamp (UTC):** 2026-07-14T17:04:11

The provided context passages do not contain any information regarding the known limitations of machine learning models, such as CatBoost, when applied to complex human behavioral data. Therefore, it is not possible to advise on how these specific technical limitations should be communicated within an expert report based solely on the retrieved evidence.

### Q6. Non-diagnostic communication and safety

**Type:** safety_communication  
**Timestamp (UTC):** 2026-07-14T17:04:37

Non-diagnostic communication strategies require involving students and families in the planning, implementation, and informing processes by sharing information about screening procedures in multiple formats prior to the effort [Context 3]. It is recommended that these communications offer the opportunity for consent or opt-out. Regarding immediate safety protocols, any risk of harm to self or others must be addressed immediately, requiring a defined process to assess screening results and triage students who need further evaluation [Context 1, Context 3]. Furthermore, screening teams are responsible for ensuring that any student identified as potentially dangerous is assessed further without delay to ensure their safety [Context 1].

---

## 5. Reference answers (`gpt-oss:20b`) — same questions, same contexts

Included so the manuscript can show candidate vs reference on **identical evidence**. These answers are **not** the deployed Report Agent for this case; they are the RAGAS ground-truth / reference stack.

### Q1 (reference)

A probability of 0.9795 indicates a positive screen for depression or anxiety and should be treated as an alert rather than a formal diagnosis. In practice, the next step is to administer a more comprehensive instrument—such as the PHQ-9 if the initial screen was a PHQ-2—to quantify symptom severity (Context 1). If the PHQ-9 score meets or exceeds the threshold of 10, clinicians should consider a brief structured assessment and discuss treatment options while arranging referral to counseling services available at the college health center or primary care practice (Contexts 1 and 4). Evidence from the DEPSCREEN-INFO trial shows that providing written, patient-targeted feedback after a positive screen can reduce depression severity and increase engagement in mental health care; similar interventions are being evaluated in the GET.FEEDBACK.GP RCT (Context 5). Thus, an elevated risk score should prompt immediate follow-up assessment, appropriate referral, and consideration of tailored feedback to enhance patient engagement, while explicitly noting that screening is a non-diagnostic tool.

### Q2 (reference)

Current longitudinal studies consistently show a statistical association between poorer academic performance and higher rates of clinically significant depressive and anxiety symptoms in young adults. In a large U.S. university sample, lower cumulative GPA was significantly predicted by positive depression screens and even more strongly when anxiety co-occurred (Eisenberg et al., 2009; see Context 2–3). The same cohort found that students who reported depressive or anxious symptoms at the start of their first year were more likely to experience academic impairment—measured as lower GPA and increased dropout risk—by year’s end, with these mental-health factors accounting for a sizable portion of the variance in academic outcomes (Context 5). Additionally, a study using PHQ-9 and GAD-7 scores found that students who screened positive for depression or anxiety reported greater functional impairment in school tasks, reinforcing the link between mental-health risk and academic difficulty (Context 1). Together, these findings underscore that academic impairment serves as an important statistical contributor to elevated screening risk for depression and anxiety among young adults.

### Q3 (reference)

Food insecurity is a prominent component of basic-needs insecurity—41% of surveyed students report it—and it co-occurs with clinically significant anxiety or depression symptoms in roughly 44% of respondents, indicating that worry about adequate food supply contributes statistically to elevated mental-health risk. The data show that students experiencing food insecurity often cite emotional stress and mental-health concerns as reasons for academic disruption, underscoring its role as a contributor to depressive and anxious symptomatology. While the report does not enumerate specific psychoeducational materials, it implies that institutions typically address these risks through campus counseling services, financial-aid guidance, and on-campus food-pantry or meal-plan assistance programs. For students identified at higher risk, referral to qualified mental-health support and informational resources about basic-needs assistance is recommended.

### Q4 (reference)

The evidence indicates that school mental-health screening can reveal strengths and well-being across the entire student body, including by grade or age group (Context 2). Screening data may show that certain classrooms or age cohorts have higher levels of resilience indicators such as life satisfaction and school belonging, which could be interpreted as protective mechanisms linked to developmental stage. However, none of the retrieved passages explicitly identify institutional type (inst_type) as a contributor to resilience; thus no evidence is available to describe protective factors tied to different school settings. To present a balanced view, highlight that screening can surface both strengths and needs for all students, allowing educators to tailor supports that reinforce identified protective factors while addressing gaps. This approach underscores the importance of early identification and equitable support rather than attributing resilience to specific institutional or age categories without empirical backing.

### Q5 (reference)

The retrieved passages focus on depression and school mental-health screening practices but do not mention machine-learning algorithms such as CatBoost or describe their performance characteristics. Consequently, there is no evidence in the supplied material to identify specific limitations of these models when applied to complex human behavioral data, nor guidance on how to report them. If an expert report were to discuss ML model constraints, it would need to draw upon additional sources that detail issues such as overfitting, interpretability challenges, bias from uneven training data, and the difficulty of capturing nuanced psychosocial variables—none of which are covered in the current evidence set.

### Q6 (reference)

Present the screening result using a strengths-based, non-diagnostic tone—emphasize early support rather than labeling and involve the student (and family when appropriate) in discussing next steps, as recommended for comprehensive school mental health screening (Context 1). Immediately respond to any indication of self-harm or harm to others by conducting a rapid safety assessment and arranging urgent referral to school or community mental health services; this is required to ensure safety (Context 1). Use a defined triage process to determine whether the student requires Tier 2 or Tier 3 supports, ensuring that referrals are made promptly and documented (Context 3). Communicate findings with empathy, focusing on available resources rather than diagnostic categories, which aligns with the goal of early identification and prevention outlined in multi-tiered systems (Context 5).

---

## 6. RAGAS scores for this case (`gemma4`)

| Question | Faithfulness | Answer relevancy | Answer correctness | Semantic similarity |
|---|---:|---:|---:|---:|
| Q1 | 0.778 | 0.636 | 0.658 | 0.916 |
| Q2 | 1.000 | 0.748 | 0.616 | 0.964 |
| Q3 | 1.000 | 0.775 | 0.400 | 0.932 |
| Q4 | 0.833 | 0.619 | 0.523 | 0.947 |
| Q5 | 1.000 | 0.000 | 0.639 | 0.919 |
| Q6 | 1.000 | 0.766 | 0.548 | 0.907 |

Judge / reference stack: `gpt-oss:20b`; embeddings: `nomic-embed-text`.

**How to read the abstentions.** Q3, Q4, and Q5 (and part of Q1) state that the retrieved passages do not cover the asked topic. That follows Report Agent Rule 2 (do not invent facts). Faithfulness is high on those rows because the answer does not add unsupported claims. Q5 answer relevancy of **0.000** is an embedding artifact of a near-pure refusal, not a contradiction of the question.

---

## 7. What this report is and is not

| It is | It is not |
|---|---|
| Six evidence-grounded answers for expert review | A clinical diagnosis or treatment plan |
| Conditioned on Case/XAI + frozen K=5 passages with archived `doc_id` / `chunk_id` | A citation-checked manuscript with inline source IDs |
| Safe abstention when evidence is missing | Proof that the knowledge base covers every planned topic |
| The deployed candidate report for this walkthrough | The only possible report (reference answers differ in style and coverage claims) |

---

## 8. How to include this in the manuscript

**Main text.** One short exhibit: Case/XAI numbers + evidence-source summary (Section 3.1) + Q2 (best XAI-linked answer) + Q6 (safety) + one abstention (Q5) to show grounding discipline.

**Supplementary.** This full file (or Sections 3–4 if space is tight). Point readers to the end-to-end walkthrough for retrieved passage excerpts: `end_to_end_case_walkthrough_hms_high_01.md`.

**Figure caption (if pairing with the planner figure).**

> **Figure X (continued). Generated report for `hms_high_01`.** The Report Agent (`gemma4`) answers the six planned questions from identical frozen hybrid-retrieval contexts (K=5; documented `doc_id` / `chunk_id` sources in Supplementary Table). Answers remain non-diagnostic, treat XAI factors as statistical contributors, and abstain when retrieved passages are insufficient (Q3–Q5). Full wording and reference (`gpt-oss:20b`) answers are in the Supplementary Material.

**Reviewer sentence.**

> We now include a complete generated report for one elevated case (`hms_high_01`): the Query Planning Agent questions, the frozen evidence sources (`doc_id`, chunk, title, score), the full Report Agent answers, the reference answers written on the same frozen evidence, and per-question RAGAS scores.

---

## 9. File index

| Artifact | Path |
|---|---|
| This exhibit | `paper_outputs/example_generated_report_hms_high_01.md` |
| Candidate answers | `report_agent_eval/outputs/answers/gemma4/hms_high_01.json` |
| Reference answers | `report_agent_eval/outputs/answers/gpt-oss_20b/hms_high_01.json` |
| Frozen plan | `query_agent_eval/outputs/plans/gemma4/hms_high_01.json` |
| Frozen evidence sources (per question) | `report_agent_eval/outputs/contexts/hms_high_01/Q1.json` … `Q6.json` |
| RAGAS long table | `report_agent_eval/outputs/ragas_per_question.csv` |
| Full pipeline walkthrough (incl. passages) | `paper_outputs/end_to_end_case_walkthrough_hms_high_01.md` |
| Prompts | `paper_outputs/appendix_prompts_planner_report_judge.md` |
