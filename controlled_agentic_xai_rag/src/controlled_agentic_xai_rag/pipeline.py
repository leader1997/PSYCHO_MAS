"""End-to-end case pipeline (simplified).

Flow: HMS prediction → case formatter → LLM query planning (questions + retrieval)
      → hybrid RAG (FAISS + BM25) → report agent (answers questions) → final narrative.
No validation gates.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from controlled_agentic_xai_rag.agents.query_planning import run_query_planning_agent
from controlled_agentic_xai_rag.agents.report_generation import run_report_generation_agent
from controlled_agentic_xai_rag.audit.logger import append_jsonl, write_json
from controlled_agentic_xai_rag.config_loader import ensure_dirs
from controlled_agentic_xai_rag.inputs.loader import get_case_profile, get_xai_card
from controlled_agentic_xai_rag.tools.case_formatter import format_case
from controlled_agentic_xai_rag.tools.hybrid_retriever import hybrid_retrieve
from controlled_agentic_xai_rag.llm.ollama_client import use_model


class PipelineOptions:
    """Runtime options: live HMS + LLM query planning + hybrid FAISS RAG."""

    def __init__(
        self,
        ollama_model: str | None = None,
        live_prediction: bool = True,
        hms_test_index: int | None = None,
        embed_model: str | None = None,
        **_: object,
    ):
        self.ollama_model = ollama_model
        self.live_prediction = live_prediction
        self.hms_test_index = hms_test_index
        self.embed_model = embed_model


def _resolve_case_inputs(
    case_id: str,
    options: PipelineOptions,
) -> tuple[dict, dict, dict | None]:
    """Return (xai_card, case_profile, prediction_audit). Live HMS is the default path."""
    if options.live_prediction:
        from controlled_agentic_xai_rag.prediction.hms_predictor import resolve_live_case

        result = resolve_live_case(
            preset=case_id,
            test_index=options.hms_test_index,
        )
        return result.card, result.profile, result.to_audit_dict()
    return get_xai_card(case_id), get_case_profile(case_id), None


def _run_case_inner(case_id: str, options: PipelineOptions) -> dict[str, Any]:
    card, profile, prediction_audit = _resolve_case_inputs(case_id, options)
    formatted = format_case(case_id, card, profile)
    write_json(f"formatted_cases/{case_id}.json", formatted)

    query_plan = run_query_planning_agent(formatted, card)
    append_jsonl("query_plans.jsonl", query_plan)
    write_json(f"query_plans/{case_id}.json", query_plan)

    audit_rows, hits, retrieval_info = hybrid_retrieve(query_plan, bm25_only=False)
    for row in audit_rows:
        row["case_id"] = case_id
        append_jsonl("retrieval_results.jsonl", row)

    report = run_report_generation_agent(formatted, card, hits, query_plan=query_plan)
    append_jsonl("generated_narratives.jsonl", report)

    final_report = json.loads(json.dumps(report))
    final_report["safety_status"] = "draft_unvalidated"
    final_report["ollama_model"] = options.ollama_model
    append_jsonl("final_narratives.jsonl", final_report)

    return {
        "case_id": case_id,
        "ollama_model": options.ollama_model,
        "prediction": prediction_audit,
        "inputs": {
            "xai_card": card,
            "case_profile": profile,
            "source": "live_hms_prediction" if options.live_prediction else "seeded_inputs",
        },
        "formatted_case": formatted,
        "query_plan": query_plan,
        "retrieval_info": retrieval_info,
        "retrieval_audit": audit_rows,
        "hits": [h.to_dict() for h in hits],
        "generated_report": report,
        "final_report": final_report,
        "claims": pd.DataFrame(),
        "validation": [],
        "revision_comments": [],
        "validators_skipped": True,
    }


def run_case(case_id: str, options: PipelineOptions | None = None) -> dict[str, Any]:
    options = options or PipelineOptions()
    ensure_dirs()
    from contextlib import nullcontext

    from controlled_agentic_xai_rag.llm.ollama_client import get_model, use_embed_model

    model = options.ollama_model or get_model()
    options.ollama_model = model
    embed_ctx = use_embed_model(options.embed_model) if options.embed_model else nullcontext()

    with embed_ctx:
        with use_model(model):
            return _run_case_inner(case_id, options)
