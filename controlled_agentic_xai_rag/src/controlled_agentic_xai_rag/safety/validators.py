"""Step 3 validation orchestration."""

from __future__ import annotations

import pandas as pd

from controlled_agentic_xai_rag.config_loader import cfg_get
from controlled_agentic_xai_rag.retrieval.base import RetrievalHit
from controlled_agentic_xai_rag.validators.citation_checker import (
    check_citations,
    claims_need_revision,
    filter_unverified_citations,
)
from controlled_agentic_xai_rag.validators.claim_support import claim_support_summary
from controlled_agentic_xai_rag.validators.safety_rules import check_safety_rules
from controlled_agentic_xai_rag.validators.xai_alignment import check_xai_alignment
from controlled_agentic_xai_rag.agents.report_generation import run_report_generation_agent


def run_validation_gate(
    report: dict,
    card: dict,
    formatted_case: dict,
    hits: list[RetrievalHit],
    max_revision_attempts: int | None = None,
) -> tuple[dict, pd.DataFrame, list[dict], list[str]]:
    elevated = formatted_case["risk_category"] == "elevated_screening_risk"
    validation_rows: list[dict] = []
    revision_comments: list[str] = []

    safety = check_safety_rules(report, elevated=elevated)
    validation_rows.append(safety)
    if not safety["passed"]:
        revision_comments.extend(safety["comments"])

    claims_df = check_citations(report, hits)
    claim_sum = claim_support_summary(claims_df)
    claim_sum["case_id"] = report.get("case_id", "")
    validation_rows.append(claim_sum)
    cite_rev, cite_comments = claims_need_revision(claims_df)
    if cite_rev:
        revision_comments.extend(cite_comments)

    xai = check_xai_alignment(report, card)
    validation_rows.append(xai)
    if not xai["passed"]:
        revision_comments.extend(xai["comments"])

    max_rev = int(max_revision_attempts if max_revision_attempts is not None else cfg_get("pipeline", "max_revision_attempts", default=2))
    attempts = 0
    while revision_comments and attempts < max_rev:
        attempts += 1
        report = run_report_generation_agent(
            formatted_case, card, hits, revision_comments=revision_comments
        )
        revision_comments = []
        safety = check_safety_rules(report, elevated=elevated)
        validation_rows.append({**safety, "revision_attempt": attempts})
        if not safety["passed"]:
            revision_comments.extend(safety["comments"])
        claims_df = check_citations(report, hits)
        claim_sum = claim_support_summary(claims_df)
        validation_rows.append({**claim_sum, "revision_attempt": attempts})
        cite_rev, cite_comments = claims_need_revision(claims_df)
        if cite_rev:
            revision_comments.extend(cite_comments)
        xai = check_xai_alignment(report, card)
        validation_rows.append({**xai, "revision_attempt": attempts})
        if not xai["passed"]:
            revision_comments.extend(xai["comments"])

    report = filter_unverified_citations(report, claims_df)
    claims_df = check_citations(report, hits)

    all_passed = (
        safety["passed"]
        and claim_support_summary(claims_df)["passed"]
        and xai["passed"]
    )
    report["safety_status"] = "accepted" if all_passed else "requires_human_review"
    return report, claims_df, validation_rows, revision_comments
