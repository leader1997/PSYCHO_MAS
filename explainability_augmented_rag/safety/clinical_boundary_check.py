"""Regex-based clinical boundary checks."""

from __future__ import annotations

import re

from generation.config_loader import cfg_get

DIAGNOSTIC_PATTERNS = [
    r"\bhas depression\b",
    r"\bhas anxiety\b",
    r"\bdiagnosed with\b",
    r"\byou have depression\b",
    r"\byou have anxiety\b",
    r"\bclinical diagnosis\b",
]

MEDICAL_ADVICE_PATTERNS = [
    r"\bshould take\b",
    r"\bprescribe\b",
    r"\bmedication for\b",
    r"\bstart therapy\b",
    r"\bdosage\b",
]

CAUSALITY_PATTERNS = [
    r"\bcaused by\b",
    r"\bcauses the risk\b",
    r"\bcaused the risk\b",
    r"\bbecause of your\b.*\brisk\b",
]

STIGMA_PATTERNS = [
    r"\bfault\b",
    r"\bweak\b",
    r"\bcrazy\b",
    r"\bpathetic\b",
]


def check_patterns(text: str, patterns: list[str]) -> list[str]:
    hits = []
    lower = text.lower()
    for p in patterns:
        if re.search(p, lower):
            hits.append(p)
    return hits


def clinical_boundary_flags(text: str) -> dict[str, list[str]]:
    return {
        "diagnostic_overclaiming": check_patterns(text, DIAGNOSTIC_PATTERNS),
        "medical_advice": check_patterns(text, MEDICAL_ADVICE_PATTERNS),
        "causality_misuse": check_patterns(text, CAUSALITY_PATTERNS),
        "stigmatizing_language": check_patterns(text, STIGMA_PATTERNS),
    }


def has_disclaimer(text: str) -> bool:
    disclaimer = cfg_get("safety", "mandatory_disclaimer", default="not a clinical diagnosis")
    return disclaimer.lower()[:20] in text.lower() or "not a diagnosis" in text.lower()


def has_referral(text: str, elevated: bool) -> bool:
    if not elevated:
        return True
    keywords = ["counselor", "professional", "support service", "mental-health", "help"]
    lower = text.lower()
    return any(k in lower for k in keywords)
