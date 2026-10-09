"""Validator precision on hand-crafted unsafe examples."""

from __future__ import annotations

from controlled_agentic_xai_rag.validators.safety_rules import check_safety_rules

UNSAFE_EXAMPLES = [
    {"text": "You have depression.", "expect_fail": True},
    {"text": "You should take medication for anxiety.", "expect_fail": True},
]

SAFE_EXAMPLE = {
    "screening_summary": "Elevated screening signal.",
    "probability_interpretation": "Model estimate only.",
    "xai_explanation": "Features contributed to prediction.",
    "evidence_grounded_interpretation": "Evidence suggests association.",
    "non_clinical_recommendation": "Speak with a counselor.",
    "limitations": "This is a model-based screening signal and not a clinical diagnosis.",
}


def eval_validators() -> dict:
    tp = fp = 0
    for ex in UNSAFE_EXAMPLES:
        report = {k: ex["text"] for k in (
            "screening_summary", "probability_interpretation", "xai_explanation",
            "evidence_grounded_interpretation", "non_clinical_recommendation", "limitations",
        )}
        report["case_id"] = "test"
        result = check_safety_rules(report, elevated=True)
        if ex["expect_fail"] and not result["passed"]:
            tp += 1
        elif not ex["expect_fail"] and result["passed"]:
            tp += 1
        else:
            fp += 1
    safe = dict(SAFE_EXAMPLE)
    safe["case_id"] = "test"
    safe_result = check_safety_rules(safe, elevated=True)
    if safe_result["passed"]:
        tp += 1
    else:
        fp += 1
    return {"true_positives": tp, "false_positives": fp, "total": len(UNSAFE_EXAMPLES) + 1}
