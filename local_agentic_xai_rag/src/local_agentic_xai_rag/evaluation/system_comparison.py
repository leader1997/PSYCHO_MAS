"""System comparison B1/B2/B3 — objective metrics only."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

from local_agentic_xai_rag.agents.normalize import normalize_evidence, normalize_report
from local_agentic_xai_rag.agents.pipeline_agents import run_evidence_agent, run_report_agent
from local_agentic_xai_rag.config_loader import ensure_dirs, path
from local_agentic_xai_rag.llm.ollama_client import ensure_llama3, structured_json
from local_agentic_xai_rag.main import load_case_profiles, load_xai_cards
from local_agentic_xai_rag.prompts import ADVISORY_REPORT_SYSTEM
from local_agentic_xai_rag.tools.pipeline_tools import (
    citation_check,
    filter_unverified_citations,
    format_case,
    hybrid_retrieve,
    safety_rules_check,
)


def run_b1(case_id: str, card: dict, profiles: dict, bm25_only: bool) -> dict:
    """Retrieve → single-shot report (no evidence agent)."""
    fmt = format_case(case_id, card, profiles)
    _, hits = hybrid_retrieve(case_id, card, bm25_only=bm25_only)
    ctx = "\n\n".join(f"[{h.chunk_id}] {h.text[:400]}" for h in hits)
    raw = structured_json(
        ADVISORY_REPORT_SYSTEM,
        f"Case: {fmt['summary']}\nContext:\n{ctx}\nGenerate advisory report JSON for {case_id}",
    )
    report = normalize_report(raw, case_id, {"supported_claims": []})
    return report


def run_b2(case_id: str, card: dict, profiles: dict, bm25_only: bool) -> dict:
    """Evidence agent → report (no safety tools)."""
    fmt = format_case(case_id, card, profiles)
    _, hits = hybrid_retrieve(case_id, card, bm25_only=bm25_only)
    chunk_ids = [h.chunk_id for h in hits]
    ev_raw = run_evidence_agent(case_id, fmt["summary"], card, hits)
    evidence = normalize_evidence(ev_raw, case_id, chunk_ids)
    rep_raw = run_report_agent(case_id, fmt["summary"], card, evidence)
    return normalize_report(rep_raw, case_id, evidence)


def run_b3(case_id: str, card: dict, profiles: dict, bm25_only: bool) -> dict:
    """Full pipeline with safety + citation check."""
    from local_agentic_xai_rag.main import run_case

    return run_case(case_id, bm25_only=bm25_only)["report"]


def score_report(report: dict, elevated: bool) -> dict:
    claims = citation_check(report)
    report = filter_unverified_citations(report, claims)
    claims = citation_check(report)
    safety = safety_rules_check(report, elevated=elevated)
    supported = (claims["support_status"] == "supported").mean() if len(claims) else float("nan")
    return {
        "disclaimer_present": safety.get("missing_disclaimer") == "pass",
        "referral_present": safety.get("missing_referral") == "pass",
        "safety_pass": safety.get("overall_safety") == "pass",
        "n_citations": len(report.get("citations") or []),
        "claim_support_rate": supported,
        "has_recommended_next_step": bool(str(report.get("recommended_next_step", "")).strip()),
    }


def run_comparison(bm25_only: bool = True, benchmark_limit: int = 20) -> pd.DataFrame:
    ensure_dirs()
    ensure_llama3()
    cards = {c["case_id"]: c for c in load_xai_cards()}
    profiles = load_case_profiles()
    rows = []
    for case_id, card in cards.items():
        elevated = float(card.get("prediction_probability", 0.5)) >= 0.5
        for system, fn in (("B1", run_b1), ("B2", run_b2), ("B3", run_b3)):
            print(f"{system} {case_id}...", flush=True)
            try:
                report = fn(case_id, card, profiles, bm25_only)
                metrics = score_report(report, elevated)
                rows.append({"system": system, "case_id": case_id, "source": "case", **metrics})
            except Exception as exc:
                print(f"  skip {system} {case_id}: {exc}", flush=True)
                rows.append(
                    {
                        "system": system,
                        "case_id": case_id,
                        "source": "case",
                        "disclaimer_present": False,
                        "referral_present": False,
                        "safety_pass": False,
                        "n_citations": 0,
                        "claim_support_rate": float("nan"),
                        "has_recommended_next_step": False,
                        "error": str(exc)[:200],
                    }
                )

    out = path("results_dir")
    if rows:
        pd.DataFrame(rows).to_csv(out / "system_comparison_cases.csv", index=False)

    if benchmark_limit > 0:
        from local_agentic_xai_rag.evaluation.retrieval_eval import build_benchmark_query
        from local_agentic_xai_rag.retrieval.hybrid import get_retriever

        bench = pd.read_csv(path("benchmark_gold_csv")).head(benchmark_limit)
        retriever = get_retriever(bm25_only=bm25_only)
        for _, row in bench.iterrows():
            qid = row["question_id"]
            print(f"B1 benchmark {qid}...", flush=True)
            try:
                query = build_benchmark_query(
                    row["question"], row.get("category", ""), row.get("gold_relevant_topics", "")
                )
                hits = retriever.retrieve(query, k=8)
                ctx = "\n\n".join(f"[{h.chunk_id}] {h.text[:400]}" for h in hits)
                raw = structured_json(
                    ADVISORY_REPORT_SYSTEM,
                    f"Question: {row['question']}\nContext:\n{ctx}\nGenerate advisory JSON with case_id={qid}",
                )
                report = normalize_report(raw, qid, {"supported_claims": []})
                metrics = score_report(report, elevated=True)
                gold_topics = str(row.get("gold_relevant_topics", "")).lower().split(",")
                text = json.dumps(report).lower()
                kw_hits = sum(1 for t in gold_topics if t.strip() and t.strip() in text)
                metrics["keyword_recall"] = kw_hits / max(len([t for t in gold_topics if t.strip()]), 1)
                rows.append({"system": "B1", "case_id": qid, "source": "benchmark", **metrics})
            except Exception as exc:
                print(f"  skip {qid}: {exc}", flush=True)
                rows.append(
                    {
                        "system": "B1",
                        "case_id": qid,
                        "source": "benchmark",
                        "disclaimer_present": False,
                        "referral_present": False,
                        "safety_pass": False,
                        "n_citations": 0,
                        "claim_support_rate": float("nan"),
                        "has_recommended_next_step": False,
                        "keyword_recall": float("nan"),
                        "error": str(exc)[:200],
                    }
                )

    df = pd.DataFrame(rows)
    summary = df.groupby("system").mean(numeric_only=True).round(4)
    out = path("results_dir")
    df.to_csv(out / "system_comparison.csv", index=False)
    summary.to_csv(out / "system_comparison_summary.csv")
    print(summary)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-only", action="store_true", default=True)
    parser.add_argument("--benchmark-limit", type=int, default=20)
    args = parser.parse_args()
    run_comparison(bm25_only=args.bm25_only, benchmark_limit=args.benchmark_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
