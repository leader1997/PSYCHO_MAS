"""XAI alignment checker."""

from __future__ import annotations

import re

from controlled_agentic_xai_rag.safety.clinical_boundary import CAUSALITY_PATTERNS, check_patterns
from controlled_agentic_xai_rag.validators.safety_rules import report_text


def check_xai_alignment(report: dict, card: dict) -> dict:
    text = report_text(report)
    xai_text = " ".join([
        report.get("xai_explanation", ""),
        report.get("contributor_analysis", ""),
    ])
    combined = f"{text} {xai_text}"

    card_features = {f.get("feature", "").lower() for f in card.get("top_positive_factors", [])}
    card_features |= {f.get("display_name", "").lower() for f in card.get("top_positive_factors", [])}
    card_protective = {f.get("feature", "").lower() for f in card.get("top_protective_factors", [])}

    invented = []
    factor_pattern = re.findall(r"\b([a-z_]{3,20})\b", xai_text.lower())
    known = card_features | card_protective | {
        "lonely", "loneliness", "flourish", "aca_impa", "fincur", "food_worry", "age",
    }
    for token in set(factor_pattern):
        if token.endswith("_impa") or token in ("lonely", "flourish", "fincur"):
            if token not in known and token.replace("_", "") not in "".join(known):
                if token not in ("model", "risk", "screening", "student", "feature"):
                    invented.append(token)

    causal_hits = check_patterns(combined, CAUSALITY_PATTERNS)
    comments = []
    if invented:
        comments.append(f"Possible invented XAI contributors: {invented[:5]}")
    if causal_hits:
        comments.append(f"Causal misuse detected: {causal_hits}")

    passed = not invented and not causal_hits
    return {
        "check": "xai_alignment",
        "case_id": report.get("case_id", ""),
        "passed": passed,
        "invented_contributors": invented[:10],
        "causal_misuse": causal_hits,
        "comments": comments,
    }
