"""Rule-based clinical boundary checks (deterministic)."""

from __future__ import annotations

import re

from local_agentic_xai_rag.config_loader import cfg_get

DIAGNOSTIC_PATTERNS = [
    r"\bhas depression\b",
    r"\bhas anxiety\b",
    r"\bdiagnosed with\b",
    r"\byou have depression\b",
    r"\byou have anxiety\b",
]

MEDICAL_ADVICE_PATTERNS = [
    r"\bshould take\b",
    r"\bprescribe\b",
    r"\bmedication for\b",
    r"\bstart therapy\b",
]

CAUSALITY_PATTERNS = [
    r"\bcaused by\b",
    r"\bcauses the risk\b",
    r"\bcaused the risk\b",
]

STIGMA_PATTERNS = [r"\bfault\b", r"\bweak\b", r"\bcrazy\b", r"\bpathetic\b"]


def check_patterns(text: str, patterns: list[str]) -> list[str]:
    lower = text.lower()
    return [p for p in patterns if re.search(p, lower)]


def clinical_boundary_flags(text: str) -> dict[str, list[str]]:
    return {
        "diagnostic_overclaiming": check_patterns(text, DIAGNOSTIC_PATTERNS),
        "medical_advice": check_patterns(text, MEDICAL_ADVICE_PATTERNS),
        "causality_misuse": check_patterns(text, CAUSALITY_PATTERNS),
        "stigmatizing_language": check_patterns(text, STIGMA_PATTERNS),
    }


def has_disclaimer(text: str) -> bool:
    disclaimer = cfg_get("safety", "mandatory_disclaimer", default="not a clinical diagnosis")
    return disclaimer.lower()[:20] in text.lower() or "not a diagnosis" in text.lower() or "not a clinical diagnosis" in text.lower()


def has_referral(text: str, elevated: bool) -> bool:
    if not elevated:
        return True
    keywords = ["counselor", "professional", "support service", "mental-health", "student support"]
    return any(k in text.lower() for k in keywords)
