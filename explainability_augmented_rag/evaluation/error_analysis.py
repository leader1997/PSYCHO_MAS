"""Error analysis aggregation (WP13)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import path  # noqa: E402

FAILURE_TAXONOMY = """
# Error Taxonomy — Explainability-Augmented RAG Phase 2

## 1. Retrieval failures
- missing_relevant_chunk
- irrelevant_retrieved_chunk
- weak_source
- wrong_ranking

## 2. Generation failures
- hallucination
- unsupported_claim
- vague_narrative
- overconfident_wording

## 3. XAI integration failures
- feature_ignored
- feature_misinterpreted
- causal_interpretation
- xai_narrative_contradiction

## 4. Safety failures
- diagnostic_wording
- medical_advice
- alarming_tone
- missing_disclaimer
- missing_referral

## 5. Citation failures
- citation_not_supporting_claim
- missing_citation
- generic_citation
"""


def collect_errors() -> pd.DataFrame:
    results_dir = path("results_dir")
    rows = []

    # Retrieval
    ablation = results_dir / "retriever_ablation_results.csv"
    if ablation.exists():
        df = pd.read_csv(ablation)
        low = df[df["precision@5"] < 0.1]
        for _, r in low.iterrows():
            rows.append(
                {
                    "failure_type": "retrieval",
                    "subtype": "missing_relevant_chunk",
                    "frequency": 1,
                    "example": r["question_id"],
                    "cause": f"low P@5={r['precision@5']:.2f} ({r['retriever']})",
                    "proposed_fix": "Use hybrid+reranker; improve gold chunk mapping",
                }
            )

    # Safety
    safety = results_dir / "safety_reports.csv"
    if safety.exists():
        df = pd.read_csv(safety)
        fails = df[df["overall_safety"] == "fail"]
        for _, r in fails.iterrows():
            for col in [
                "diagnostic_overclaiming",
                "medical_advice",
                "causality_misuse",
                "missing_disclaimer",
                "missing_referral",
            ]:
                if r.get(col) == "fail":
                    rows.append(
                        {
                            "failure_type": "safety",
                            "subtype": col,
                            "frequency": 1,
                            "example": r["case_id"],
                            "cause": str(r.get("comments", ""))[:200],
                            "proposed_fix": "Revision loop + stronger prompts",
                        }
                    )

    # Faithfulness
    faith = results_dir / "claim_level_faithfulness.csv"
    if faith.exists():
        df = pd.read_csv(faith)
        bad = df[df["support_status"].isin(["unsupported", "contradicted"])]
        for _, r in bad.iterrows():
            rows.append(
                {
                    "failure_type": "citation",
                    "subtype": "citation_not_supporting_claim",
                    "frequency": 1,
                    "example": r["claim"][:100],
                    "cause": r.get("explanation", "")[:200],
                    "proposed_fix": "Evidence synthesis step + claim-level check",
                }
            )

    # RAGAS failures
    ragas_fail = results_dir / "ragas_failure_cases.csv"
    if ragas_fail.exists() and ragas_fail.stat().st_size > 10:
        df = pd.read_csv(ragas_fail)
        for _, r in df.head(20).iterrows():
            rows.append(
                {
                    "failure_type": "generation",
                    "subtype": "low_ragas_score",
                    "frequency": 1,
                    "example": r.get("question_id", ""),
                    "cause": f"system={r.get('system', '')}",
                    "proposed_fix": "Full context to RAGAS; improve retrieval",
                }
            )

    err_df = pd.DataFrame(rows)
    if not err_df.empty:
        summary = (
            err_df.groupby(["failure_type", "subtype"])
            .size()
            .reset_index(name="frequency")
        )
        err_df = pd.concat([err_df, summary.assign(example="", cause="", proposed_fix="")], ignore_index=True)

    out_csv = results_dir / "error_analysis.csv"
    err_df.to_csv(out_csv, index=False)
    taxonomy_path = path("paper_outputs") / "error_taxonomy.md"
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_path.write_text(FAILURE_TAXONOMY, encoding="utf-8")
    print(f"Error analysis -> {out_csv}")
    return err_df


def main() -> int:
    collect_errors()
    return 0


if __name__ == "__main__":
    sys.exit(main())
