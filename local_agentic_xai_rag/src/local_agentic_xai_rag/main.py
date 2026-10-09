"""Case pipeline using tools + crew-equivalent agents."""

from __future__ import annotations

import argparse
import json
import shutil
import sys

import pandas as pd

from local_agentic_xai_rag.config_loader import cfg_get, ensure_dirs, path, validate_phase2_paths
from local_agentic_xai_rag.agents.normalize import normalize_query_plan
from local_agentic_xai_rag.agents.pipeline_agents import run_query_planning_agent
from local_agentic_xai_rag.crew import run_crew_pipeline
from local_agentic_xai_rag.llm.ollama_client import MODEL, ensure_llama3
from local_agentic_xai_rag.tools.pipeline_tools import (
    citation_check,
    claims_need_revision,
    filter_unverified_citations,
    format_case,
    hybrid_retrieve,
    safety_rules_check,
)


def load_xai_cards() -> list[dict]:
    p = path("xai_cards_jsonl")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_case_profiles() -> dict:
    p = path("case_profiles")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def run_case(
    case_id: str,
    bm25_only: bool = False,
    use_crewai: bool = False,
    use_query_planning: bool | None = None,
    templates_only: bool = False,
) -> dict:
    cards = {c["case_id"]: c for c in load_xai_cards()}
    profiles = load_case_profiles()
    if case_id not in cards:
        raise KeyError(f"Unknown case_id: {case_id}")
    card = cards[case_id]
    fmt = format_case(case_id, card, profiles)
    elevated = float(card.get("prediction_probability", 0.5)) >= float(
        cfg_get("pipeline", "risk_threshold", default=0.5)
    )

    query_plan = None
    plan_enabled = (
        cfg_get("pipeline", "use_query_planning_agent", default=True)
        if use_query_planning is None
        else use_query_planning
    )
    if plan_enabled and not templates_only:
        raw_plan = run_query_planning_agent(case_id, fmt["summary"], card, profiles)
        query_plan = normalize_query_plan(raw_plan, case_id, card)

    queries, hits = hybrid_retrieve(case_id, card, bm25_only=bm25_only, query_plan=query_plan)

    if use_crewai:
        from local_agentic_xai_rag.crew import try_crewai_kickoff
        out = try_crewai_kickoff(case_id, fmt["summary"], card, hits)
        if out:
            evidence, report = out
        else:
            evidence, report = run_crew_pipeline(case_id, fmt["summary"], card, hits)
    else:
        evidence, report = run_crew_pipeline(case_id, fmt["summary"], card, hits)

    safety = safety_rules_check(report, elevated=elevated)
    claims_df = citation_check(report)

    max_rev = cfg_get("pipeline", "max_revision_attempts", default=1)
    attempts = 0
    cite_rev, cite_comments = claims_need_revision(claims_df)
    while (safety.get("revision_required") or cite_rev) and attempts < max_rev:
        attempts += 1
        comments = list(safety.get("comments", [])) + cite_comments
        _, report = run_crew_pipeline(
            case_id, fmt["summary"], card, hits, evidence=evidence,
            revision_comments=comments,
        )
        safety = safety_rules_check(report, elevated=elevated)
        claims_df = citation_check(report)
        cite_rev, cite_comments = claims_need_revision(claims_df)

    report = filter_unverified_citations(report, claims_df)
    claims_df = citation_check(report)

    return {
        "case_id": case_id,
        "query_plan": query_plan,
        "queries": queries,
        "hits": [h.to_dict() for h in hits],
        "evidence": evidence,
        "report": report,
        "safety": safety,
        "claims": claims_df,
        "model": MODEL,
    }


def save_results(result: dict) -> None:
    out = path("results_dir")
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result["queries"]).to_csv(
        out / "generated_queries.csv",
        mode="a",
        header=not (out / "generated_queries.csv").exists(),
        index=False,
    )

    if result.get("query_plan"):
        with open(out / "query_plans.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result["query_plan"], ensure_ascii=False) + "\n")

    with open(out / "retrieval_results.jsonl", "a", encoding="utf-8") as f:
        for h in result["hits"]:
            f.write(json.dumps({"case_id": result["case_id"], **h}, ensure_ascii=False) + "\n")

    with open(out / "evidence_summaries.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result["evidence"], ensure_ascii=False) + "\n")

    with open(out / "generated_narratives.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result["report"], ensure_ascii=False) + "\n")

    pd.DataFrame([result["safety"]]).to_csv(
        out / "validation_reports.csv",
        mode="a",
        header=not (out / "validation_reports.csv").exists(),
        index=False,
    )

    if len(result["claims"]):
        result["claims"].to_csv(
            out / "claim_level_faithfulness.csv",
            mode="a",
            header=not (out / "claim_level_faithfulness.csv").exists(),
            index=False,
        )

    with open(out / "final_narratives.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result["report"], ensure_ascii=False) + "\n")

    print(f"Case {result['case_id']}: safety={result['safety']['overall_safety']} claims={len(result['claims'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local XAI-RAG pipeline (Ollama llama3 only)")
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--use-crewai", action="store_true")
    parser.add_argument("--templates-only", action="store_true", help="Skip query planning agent; use template queries")
    parser.add_argument("--no-query-planning", action="store_true", help="Disable XAI query planning agent")
    args = parser.parse_args()

    missing = validate_phase2_paths()
    if missing:
        print("ERROR: Missing Phase 2 paths:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    ensure_dirs()
    if args.fresh:
        rd = path("results_dir")
        if rd.exists():
            shutil.rmtree(rd)
        ensure_dirs()

    ensure_llama3()
    print(f"Ollama OK | model={MODEL} only (chat + embeddings)")

    cards = load_xai_cards()
    case_ids = [c["case_id"] for c in cards] if args.all_cases else [args.case or cards[0]["case_id"]]

    for cid in case_ids:
        print(f"Processing {cid}...", flush=True)
        save_results(
            run_case(
                cid,
                bm25_only=args.bm25_only,
                use_crewai=args.use_crewai,
                use_query_planning=not args.no_query_planning,
                templates_only=args.templates_only,
            )
        )

    print(f"Results -> {path('results_dir')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
