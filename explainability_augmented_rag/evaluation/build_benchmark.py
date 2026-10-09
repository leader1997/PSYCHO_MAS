"""Build/enhance 100-Q RAG benchmark (WP9)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Category mapping to master-prompt taxonomy
CATEGORY_MAP = {
    "loneliness_belonging": "student_mental_health",
    "academic_impairment": "student_mental_health",
    "financial_stress": "risk_protective_factors",
    "sleep_lifestyle": "risk_protective_factors",
    "flourishing_positive_mh": "protective_factors",
    "help_seeking_counseling": "safety_referral",
    "ethics_not_diagnosis": "safety_ethics",
    "xai_explanation": "xai_explanation",
    "case_based_explanation": "screening_interpretation",
    "insufficient_context_safety": "safety_ethics",
}

ANSWER_TYPE_MAP = {
    "literature_grounded": "factual",
    "xai_grounded": "xai_safety",
    "safety_refusal": "safety",
    "case_explanation": "explanatory",
    "methodology": "methodology",
}


def main() -> int:
    from scripts.remap_gold_chunks import remap_benchmark
    import pandas as pd
    from generation.config_loader import path

    df = remap_benchmark()
    df["category_master"] = df["category"].map(CATEGORY_MAP).fillna(df["category"])
    df["answer_type_master"] = df["answer_type"].map(ANSWER_TYPE_MAP).fillna(df["answer_type"])
    out = path("benchmark_csv")
    df.to_csv(out, index=False)
    print(f"Enhanced benchmark -> {out} ({len(df)} questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
