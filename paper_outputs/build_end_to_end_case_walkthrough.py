"""Build an end-to-end case walkthrough markdown from stored artifacts (hms_high_01)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MAS = Path(__file__).resolve().parents[1]
CASE_ID = "hms_high_01"
PLAN_MODEL = "gemma4"
OUT = MAS / "paper_outputs" / "end_to_end_case_walkthrough_hms_high_01.md"
BUNDLE = MAS / "paper_outputs" / "end_to_end_case_hms_high_01_bundle"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def trunc(text: str, n: int = 500) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def md_escape_fence(s: str) -> str:
    return s.replace("```", "``\u200b`")


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)

    # --- Stage 0/1: profile ---
    profile = None
    for line in (MAS / "query_agent_eval" / "data" / "case_profiles_50.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        if o.get("case_id") == CASE_ID:
            profile = o
            break
    if profile is None:
        raise SystemExit(f"Missing profile {CASE_ID}")
    (BUNDLE / "01_case_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    sc = profile["structured_case"]

    # raw values if present in xai cards / case profiles
    raw_values = None
    cp_path = MAS / "xai_rag_depression" / "outputs" / "case_profiles.json"
    if cp_path.exists():
        cps = load_json(cp_path)
        if isinstance(cps, list):
            for c in cps:
                if c.get("case_id") in (CASE_ID, "high_risk_case") and c.get("predicted_probability", 0) > 0.9:
                    raw_values = c
                    break
        elif isinstance(cps, dict) and CASE_ID in cps:
            raw_values = cps[CASE_ID]
    if raw_values:
        (BUNDLE / "01b_case_profile_alt.json").write_text(
            json.dumps(raw_values, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # --- Stage 2: gemma4 plan + judge ---
    plan_wrap = load_json(MAS / "query_agent_eval" / "outputs" / "plans" / PLAN_MODEL / f"{CASE_ID}.json")
    judge_wrap = load_json(MAS / "query_agent_eval" / "outputs" / "judge" / PLAN_MODEL / f"{CASE_ID}.json")
    (BUNDLE / "02_query_plan_gemma4.json").write_text(
        json.dumps(plan_wrap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (BUNDLE / "03_query_plan_judge_gemma4.json").write_text(
        json.dumps(judge_wrap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plan = plan_wrap.get("plan") or {}
    qplan = plan.get("question_plan") or []

    # --- Stage 3: contexts ---
    ctx_dir = MAS / "report_agent_eval" / "outputs" / "contexts" / CASE_ID
    contexts = {}
    for q in qplan:
        qid = q.get("question_id") or q.get("qid")
        fp = ctx_dir / f"{qid}.json"
        if fp.exists():
            contexts[qid] = load_json(fp)
            (BUNDLE / f"04_context_{qid}.json").write_text(
                json.dumps(contexts[qid], indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # --- Stage 4: answers ---
    ref = load_json(MAS / "report_agent_eval" / "outputs" / "answers" / "gpt-oss_20b" / f"{CASE_ID}.json")
    cand = load_json(MAS / "report_agent_eval" / "outputs" / "answers" / "gemma4" / f"{CASE_ID}.json")
    (BUNDLE / "05_answers_gpt-oss_reference.json").write_text(
        json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (BUNDLE / "05b_answers_gemma4_candidate.json").write_text(
        json.dumps(cand, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Stage 5: RAGAS rows for this case ---
    ragas = pd.read_csv(MAS / "report_agent_eval" / "outputs" / "ragas_per_question.csv")
    ragas_case = ragas[ragas["case_id"] == CASE_ID].copy()
    ragas_case.to_csv(BUNDLE / "06_ragas_scores_this_case.csv", index=False)

    # Build markdown
    lines: list[str] = []
    a = lines.append

    a(f"# End-to-End Case Walkthrough - `{CASE_ID}`")
    a("")
    a("**Purpose.** Show every major data object that circulates while one student case is processed by the Agentic XAI-Contextual RAG pipeline (prediction -> XAI -> query planning -> retrieval -> grounded answers -> scoring).")
    a("")
    a("**Case:** elevated screening stratum (`high`)  ")
    a(f"**Plan model (frozen):** `{PLAN_MODEL}`  ")
    a("**Reference / judge model:** `gpt-oss:20b`  ")
    a("**Retrieval:** Ollama-context hybrid, K=5  ")
    a(f"**Artifact bundle (full JSON):** `paper_outputs/end_to_end_case_hms_high_01_bundle/`")
    a("")
    a("---")
    a("")
    a("## Pipeline overview")
    a("")
    a("```")
    a("HMS row / case profile")
    a("        |")
    a("        v")
    a("CatBoost probability + TreeSHAP XAI card")
    a("        |")
    a("        v")
    a("Query Planning Agent  ->  JSON question plan (+ retrieval queries)")
    a("        |")
    a("        v")
    a("Contextual hybrid RAG (K=5)  ->  frozen context passages per question")
    a("        |")
    a("        v")
    a("Report / Answer Agent  ->  grounded answers (reference + candidates)")
    a("        |")
    a("        v")
    a("Safety / schema checks (controlled pipeline) + RAGAS answer metrics")
    a("```")
    a("")
    a("---")
    a("")
    a("## Stage 1 - Case identity and screening prediction")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    a(f"| `case_id` | `{CASE_ID}` |")
    a(f"| `stratum` | `{profile.get('stratum')}` |")
    a(f"| `test_index` (HMS test split row) | `{profile.get('test_index')}` |")
    a(f"| `prediction_probability` | **{sc.get('prediction_probability')}** |")
    a(f"| `decision_threshold` | {sc.get('decision_threshold')} |")
    a(f"| `risk_label` | `{sc.get('risk_label')}` |")
    a(f"| Calibration | `{json.dumps(sc.get('calibration'))}` |")
    a("")
    a("**Interpretation circulated downstream:** elevated depression/anxiety **screening** risk (not a diagnosis); probability far above threshold; not near-threshold.")
    a("")
    a("---")
    a("")
    a("## Stage 2 - Student feature levels (de-identified profile)")
    a("")
    a("These categorical levels are what the Query Planning Agent sees as `student_profile.feature_levels`:")
    a("")
    a("| Feature | Level |")
    a("|---|---|")
    for k, v in (sc.get("student_profile") or {}).get("feature_levels", {}).items():
        a(f"| `{k}` | {v} |")
    a("")
    a("---")
    a("")
    a("## Stage 3 - Local XAI (TreeSHAP contributors)")
    a("")
    a("### Risk-increasing contributors (top)")
    a("")
    a("| Feature | Display name | SHAP contribution | Direction | Level note |")
    a("|---|---|---:|---|---|")
    for x in sc.get("xai_contributors") or []:
        a(
            f"| `{x.get('feature')}` | {x.get('display_name')} | {x.get('contribution')} | "
            f"{x.get('direction')} | {x.get('interpretation')} |"
        )
    a("")
    a("### Protective / risk-decreasing contributors (top)")
    a("")
    a("| Feature | Display name | SHAP contribution | Direction |")
    a("|---|---|---:|---|")
    for x in sc.get("protective_contributors") or []:
        a(
            f"| `{x.get('feature')}` | {x.get('display_name')} | {x.get('contribution')} | "
            f"{x.get('direction')} |"
        )
    a("")
    a("**Language rule enforced in prompts:** contributors are statistical associations, **not causes**.")
    a("")
    a("---")
    a("")
    a("## Stage 4 - Query Planning Agent output (gemma4)")
    a("")
    a(f"- Plan parse OK: `{plan_wrap.get('parsed_ok')}`")
    a(f"- Timestamp (UTC): `{plan_wrap.get('timestamp_utc')}`")
    a(f"- Planning strategy: `{(plan.get('risk_interpretation') or {}).get('planning_strategy')}`")
    a(f"- Number of questions: **{len(qplan)}**")
    a("")
    a("### Self-check flags (agent introspection)")
    a("")
    a("```json")
    a(md_escape_fence(json.dumps(plan.get("self_check") or {}, indent=2)))
    a("```")
    a("")
    a("### Question plan (full circulated questions + retrieval queries)")
    a("")
    for q in qplan:
        qid = q.get("question_id")
        a(f"#### {qid} - `{q.get('question_type')}` (priority: {q.get('priority')})")
        a("")
        a(f"**Question:** {q.get('question')}")
        a("")
        a(f"**Linked XAI factor:** `{q.get('linked_xai_factor')}`")
        a("")
        a(f"**Retrieval query:** `{q.get('retrieval_query')}`")
        a("")
        a(f"**Expected evidence:** {q.get('expected_evidence')}")
        a("")
        a(f"**Rationale:** {q.get('rationale')}")
        a("")

    a("### Level-2 LLM judge scores for this plan (gemma4 x gpt-oss:20b)")
    a("")
    scores = judge_wrap.get("scores") or judge_wrap.get("judge") or judge_wrap
    # normalize
    if isinstance(scores, dict) and "scores" in scores:
        scores = scores["scores"]
    a("```json")
    a(md_escape_fence(json.dumps(scores if isinstance(scores, dict) else judge_wrap, indent=2)[:4000]))
    a("```")
    a("")
    a("---")
    a("")
    a("## Stage 5 - Contextual RAG retrieval (frozen contexts, K=5)")
    a("")
    a("For each planned question, the system retrieves **5** passages (Ollama-context hybrid) and freezes them so every answer model sees the **same evidence**.")
    a("")

    for q in qplan:
        qid = q.get("question_id")
        ctx = contexts.get(qid)
        if not ctx:
            a(f"### {qid} - contexts missing")
            a("")
            continue
        passages = ctx.get("contexts") or ctx.get("passages") or ctx.get("hits") or []
        meta = {k: ctx.get(k) for k in ("case_id", "qid", "retrieval_query", "condition", "mode", "top_k", "question") if k in ctx}
        a(f"### {qid} retrieval package")
        a("")
        a(f"- Retrieval query used: `{q.get('retrieval_query')}`")
        if meta:
            a(f"- Metadata: `{json.dumps(meta, ensure_ascii=False)}`")
        a(f"- Passages returned: **{len(passages)}**")
        a("")
        for i, p in enumerate(passages, start=1):
            if isinstance(p, dict):
                text = p.get("text") or p.get("chunk") or p.get("content") or json.dumps(p)[:400]
                src = p.get("doc_id") or p.get("source") or p.get("document") or p.get("category") or ""
                score = p.get("score") or p.get("rrf_score") or p.get("hybrid_score")
                a(f"**[{i}]** source=`{src}` score=`{score}`")
                a("")
                a("> " + trunc(str(text), 420).replace("\n", "\n> "))
                a("")
            else:
                a(f"**[{i}]**")
                a("")
                a("> " + trunc(str(p), 420).replace("\n", "\n> "))
                a("")

    a("---")
    a("")
    a("## Stage 6 - Grounded answers (identical contexts)")
    a("")
    a("Below: **reference** (`gpt-oss:20b`) vs **candidate** (`gemma4`) for each question.")
    a("")

    ref_ans = ref.get("answers") or {}
    cand_ans = cand.get("answers") or {}
    for q in qplan:
        qid = q.get("question_id")
        r = ref_ans.get(qid) or {}
        c = cand_ans.get(qid) or {}
        a(f"### {qid}")
        a("")
        a(f"**Question:** {r.get('question') or q.get('question')}")
        a("")
        a(f"**n_contexts:** {r.get('n_contexts')}")
        a("")
        a("**Reference answer (gpt-oss:20b):**")
        a("")
        a(trunc(str(r.get("answer") or ""), 900))
        a("")
        a("**Candidate answer (gemma4):**")
        a("")
        a(trunc(str(c.get("answer") or ""), 900))
        a("")

    a("---")
    a("")
    a("## Stage 7 - RAGAS scores for this case (per question x model)")
    a("")
    if not ragas_case.empty:
        show = ragas_case[
            [
                "model",
                "qid",
                "question_type",
                "faithfulness",
                "answer_relevancy",
                "answer_correctness",
                "semantic_similarity",
            ]
        ].copy()
        show = show.round(3)
        a("| " + " | ".join(show.columns) + " |")
        a("|" + "|".join(["---"] * len(show.columns)) + "|")
        for _, row in show.iterrows():
            a("| " + " | ".join(str(row[c]) for c in show.columns) + " |")
    else:
        a("_No RAGAS rows found for this case._")
    a("")
    a("---")
    a("")
    a("## Stage 8 - What a controlled report agent would assemble")
    a("")
    a("In the controlled agentic pipeline, answers + XAI + safety rules are composed into a structured **non-clinical screening advisory** JSON with fields such as:")
    a("")
    a("- `screening_summary`")
    a("- `probability_interpretation`")
    a("- `xai_explanation`")
    a("- `evidence_grounded_interpretation`")
    a("- `non_clinical_recommendation`")
    a("- `limitations`")
    a("- `citations` / safety status")
    a("")
    a("Illustrative assembled summary for this case (derived from circulated fields above, not a new LLM call):")
    a("")
    a("> **Screening summary.** Elevated depression/anxiety screening signal (predicted probability ~0.98; threshold 0.5). This is a screening support signal for expert review, not a diagnosis.")
    a(">")
    a("> **XAI explanation.** Top statistical contributors included academic impairment, low flourishing, financial strain, loneliness, and food-insecurity worry. Protective contributors were comparatively small in magnitude.")
    a(">")
    a("> **Evidence-grounded notes.** Literature passages retrieved for the planned questions discuss academic impairment, flourishing, financial stress, loneliness, and non-diagnostic communication practices (see Stage 5 contexts).")
    a(">")
    a("> **Non-clinical recommendation.** Consider referral pathways to qualified campus/community support; avoid diagnostic or treatment language.")
    a(">")
    a("> **Limitations.** Survey-based screening model; SHAP contributions are associative; retrieved evidence may be incomplete; local LLM outputs vary by model version.")
    a("")
    a("---")
    a("")
    a("## Data-flow checklist (everything that circulated)")
    a("")
    a("| Stage | Object | Source file |")
    a("|---|---|---|")
    a(f"| 1–3 | Structured case + XAI | `query_agent_eval/data/case_profiles_50.jsonl` (`{CASE_ID}`) |")
    a(f"| 4 | Query plan | `query_agent_eval/outputs/plans/{PLAN_MODEL}/{CASE_ID}.json` |")
    a(f"| 4 | Plan judge | `query_agent_eval/outputs/judge/{PLAN_MODEL}/{CASE_ID}.json` |")
    a(f"| 5 | Frozen contexts Q1–Q6 | `report_agent_eval/outputs/contexts/{CASE_ID}/Q*.json` |")
    a(f"| 6 | Reference answers | `report_agent_eval/outputs/answers/gpt-oss_20b/{CASE_ID}.json` |")
    a(f"| 6 | Candidate answers | `report_agent_eval/outputs/answers/gemma4/{CASE_ID}.json` |")
    a(f"| 7 | RAGAS rows | `report_agent_eval/outputs/ragas_per_question.csv` (filtered) |")
    a(f"| Bundle copy | All of the above | `paper_outputs/end_to_end_case_hms_high_01_bundle/` |")
    a("")
    a("---")
    a("")
    a("## One-sentence takeaway for reviewers")
    a("")
    a(f"For `{CASE_ID}`, the system circulated a de-identified feature profile, CatBoost probability **{sc.get('prediction_probability')}**, TreeSHAP risk/protective factors, a **{len(qplan)}**-question gemma4 plan with retrieval queries, **K=5** frozen literature contexts per question, grounded reference/candidate answers on identical evidence, and per-question RAGAS scores - all without issuing a clinical diagnosis.")
    a("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)
    print("Bundle", BUNDLE)


if __name__ == "__main__":
    main()
