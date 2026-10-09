"""Safety rules validator."""

from __future__ import annotations

from controlled_agentic_xai_rag.safety.clinical_boundary import (
    clinical_boundary_flags,
    has_disclaimer,
    has_referral,
)


def report_text(report: dict) -> str:
    parts = [
        report.get("screening_summary", ""),
        report.get("contributor_analysis", ""),
        report.get("probability_interpretation", ""),
        report.get("xai_explanation", ""),
        report.get("evidence_grounded_interpretation", ""),
        report.get("actionable_recommendations", ""),
        report.get("non_clinical_recommendation", ""),
        report.get("protective_factors_note", ""),
        report.get("limitations", ""),
    ]
    return " ".join(str(p) for p in parts if p)


def check_safety_rules(report: dict, elevated: bool) -> dict:
    text = report_text(report)
    flags = clinical_boundary_flags(text)
    comments = []
    checks = {k: ("fail" if v else "pass") for k, v in flags.items()}
    for k, st in checks.items():
        if st == "fail":
            comments.append(f"{k}: failed")
    checks["missing_disclaimer"] = "fail" if not has_disclaimer(text) else "pass"
    if checks["missing_disclaimer"] == "fail":
        comments.append("Missing non-diagnostic disclaimer")
    checks["missing_referral"] = "fail" if not has_referral(text, elevated) else "pass"
    if checks["missing_referral"] == "fail":
        comments.append(
            "Advisory: elevated case — consider whether recommendations include appropriate "
            "counseling/student-support referral (expert review)."
        )
    fail = [k for k, v in checks.items() if v == "fail" and k != "missing_referral"]
    return {
        "check": "safety_rules",
        "case_id": report.get("case_id", ""),
        **checks,
        "passed": not fail,
        "comments": comments,
    }
