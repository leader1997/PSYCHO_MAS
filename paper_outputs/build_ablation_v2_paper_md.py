"""Build strengthened ablation paper MD from v2 CSV outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MAS = Path(__file__).resolve().parents[1]
OUT = MAS / "paper_outputs" / "query_strategy_ablation_v2"
MD = MAS / "paper_outputs" / "end_to_end_query_strategy_ablation_v2_for_paper.md"


def main() -> None:
    cases = json.loads((OUT / "selected_cases.json").read_text(encoding="utf-8"))
    n_cases = len(cases)
    strata = {}
    for c in cases:
        strata[c["stratum"]] = strata.get(c["stratum"], 0) + 1

    table = pd.read_csv(OUT / "table_x_end_to_end_ablation.csv") if (OUT / "table_x_end_to_end_ablation.csv").exists() else None
    ret = pd.read_csv(OUT / "retrieval_summary_by_arm.csv") if (OUT / "retrieval_summary_by_arm.csv").exists() else None
    rep = pd.read_csv(OUT / "report_summary_by_arm.csv") if (OUT / "report_summary_by_arm.csv").exists() else None
    deltas = pd.read_csv(OUT / "paired_case_bootstrap_deltas.csv") if (OUT / "paired_case_bootstrap_deltas.csv").exists() else None
    scores = pd.read_csv(OUT / "ragas_scores.csv") if (OUT / "ragas_scores.csv").exists() else None

    n_scored = len(scores) if scores is not None else 0
    n_expected = n_cases * 6 * 3

    lines = []
    a = lines.append
    a("# End-to-End Query-Strategy Ablation (Strengthened v2) — Paper Report")
    a("")
    a("**Status.** Strengthened protocol addressing sample size, fairness, metrics, and case-level statistics.")
    a(f"**Cases.** {n_cases} stratified (" + ", ".join(f"{k}={v}" for k, v in sorted(strata.items())) + ")")
    a(f"**Eval questions.** 6 independent deterministic objectives per case (not Gemma plan text)")
    a(f"**Arms.** direct (1 query) | template (6 queries) | llm_plan (6 queries); final evidence budget B=5")
    a(f"**RAGAS scoring progress.** {n_scored}/{n_expected} question-arm instances")
    a("")
    a("Outputs: `paper_outputs/query_strategy_ablation_v2/`")
    a("")
    a("---")
    a("")
    a("## 1. Design upgrades vs pilot")
    a("")
    a("| Issue in pilot | v2 fix |")
    a("|---|---|")
    a("| 3 cases / 18 questions | 20 stratified cases / 120 questions / 360 report instances |")
    a("| Eval questions from Gemma plan | Independent E1–E6 objectives from Case/XAI card only |")
    a("| Template 7–10 vs LLM 6 queries | Template and LLM fixed at **exactly 6** queries |")
    a("| Proxy faithfulness/relevancy | True **RAGAS** faithfulness + answer_relevancy (`gpt-oss:20b` + `nomic-embed-text`) |")
    a("| No claim metric | **Claim support rate** (sentence-level evidence support) |")
    a("| Question-level bootstrap only | **Case-level** paired bootstrap for arm deltas |")
    a("")
    a("## 2. Held constant")
    a("")
    a("KB, Ollama-context hybrid retriever, `nomic-embed-text`, K=5, final budget B=5, Report Agent `gemma4` (temp 0.1), identical answer prompt, identical independent evaluation questions.")
    a("")
    a("All arms receive the same Case/XAI inputs:")
    a("`{p_hat, risk label, top XAI contributors, protective factors, uncertainty}`.")
    a("Only the transformation into retrieval queries differs.")
    a("")
    a("## 3. Independent evaluation questions (E1–E6)")
    a("")
    a("1. Screening interpretation")
    a("2. Top XAI contributor #1")
    a("3. Top XAI contributor #2")
    a("4. Top XAI contributor #3")
    a("5. Protective factors")
    a("6. Limitations / safety / support")
    a("")
    a("---")
    a("")
    if table is not None:
        a("## 4. Table X. End-to-End Query-Strategy Ablation")
        a("")
        a("| Metric | Direct | Template | LLM planning | Δ LLM–Direct | 95% CI | Δ LLM–Template | 95% CI |")
        a("|---|---|---|---|---:|---|---:|---|")
        for _, r in table.iterrows():
            a(
                f"| {r['metric']} | {r['direct']} | {r['template']} | {r['llm_plan']} | "
                f"{r['delta_llm_direct']} | {r['ci_llm_direct']} | {r['delta_llm_template']} | {r['ci_llm_template']} |"
            )
        a("")
        a("Means are case-level means with case-resampled 95% bootstrap CIs (B=2000).")
        a("")
    else:
        a("## 4. Table X")
        a("")
        a("_Table not ready yet — waiting for RAGAS scoring to finish._")
        a("")

    a("## 5. Methods paragraph (paste-ready)")
    a("")
    a(
        f"**End-to-end query-strategy ablation.** We compared three query-generation strategies on "
        f"**{n_cases}** stratified HMS cases ({strata.get('high',0)} elevated / {strata.get('borderline',0)} borderline / "
        f"{strata.get('low',0)} low): (1) direct retrieval from the Case/XAI card without query decomposition (1 query); "
        f"(2) six deterministic XAI-aware template queries; and (3) six LLM Query Planning Agent retrieval queries "
        f"(frozen `gemma4` plans). All arms used identical Case/XAI inputs; only query formulation differed. "
        f"Downstream components were held constant (knowledge base, hybrid BM25–dense retriever, embedding model, "
        f"K=5, final evidence budget B=5, Report Agent, temperature, prompt). Evaluation used six independent "
        f"deterministic questions per case (screening interpretation, three top XAI contributors, protective factors, "
        f"limitations/safety). Retrieval metrics included Recall@5, nDCG@5, MRR@5, Hit@5, and XAI evidence coverage. "
        f"Reports were scored with RAGAS faithfulness and answer relevancy (`gpt-oss:20b` judge; `nomic-embed-text` "
        f"embeddings) plus claim support rate. Uncertainty used case-level paired bootstrap confidence intervals "
        f"(B=2000) for arm contrasts."
    )
    a("")
    a("## 6. Reviewer response (paste-ready)")
    a("")
    a(
        "Thank you for this important suggestion. We agree that component-wise evaluations do not by themselves "
        "demonstrate the added value of the integrated pipeline. We therefore strengthened our end-to-end "
        "query-strategy ablation to address sample size, fairness, metric consistency, and statistical dependence. "
        f"The revised experiment uses {n_cases} stratified cases, independent (non-planner) evaluation questions, "
        "equalized six-query budgets for template and LLM arms, identical Case/XAI inputs across arms, the same "
        "RAGAS faithfulness/answer-relevancy framework used elsewhere in the manuscript, claim-support rate, and "
        "case-level paired bootstrap confidence intervals for Δ(LLM−Direct) and Δ(LLM−Template). All downstream "
        "components (KB, hybrid retriever, K=5, evidence budget B=5, Report Agent, prompts) remained fixed so that "
        "query strategy is the sole experimental variable. Results are reported in Table X."
    )
    a("")
    a("## 7. Selected cases")
    a("")
    a("| case_id | stratum |")
    a("|---|---|")
    for c in cases:
        a(f"| `{c['case_id']}` | {c['stratum']} |")
    a("")
    a("## 8. Limitations")
    a("")
    a("- Retrieval relevance remains automatic (category/keyword); manual dual annotation on a subset remains desirable.")
    a("- Expert usefulness ratings were not collected in this automated pass.")
    a("- Claim support uses sentence-level lexical grounding against retrieved contexts (reproducible; not multi-rater claim adjudication).")
    a("- RAGAS judge is local `gpt-oss:20b` (same family as elsewhere in the paper).")
    a("")
    if n_scored < n_expected:
        a(f"**Note:** RAGAS scoring incomplete ({n_scored}/{n_expected}). Re-run `python paper_outputs/run_query_strategy_ablation_v2.py score` to resume.")
        a("")

    MD.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", MD)


if __name__ == "__main__":
    main()
