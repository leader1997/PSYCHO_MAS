"""Create expert review Excel template (WP12)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import path  # noqa: E402

CRITERIA = [
    "clinical_appropriateness",
    "non_diagnostic_language",
    "faithfulness_evidence",
    "faithfulness_xai",
    "clarity",
    "tone_emotional_safety",
    "usefulness_next_step",
    "absence_overclaiming",
    "citation_relevance",
    "overall_acceptability",
]


def main() -> int:
    template = pd.DataFrame(
        columns=[
            "case_id",
            "system_id",
            "reviewer_id",
            "generated_narrative",
            "prediction_label",
            "xai_factors",
            "cited_evidence",
            *CRITERIA,
            "reviewer_comments",
        ]
    )
    out = path("paper_outputs") / "expert_review_template.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    template.to_excel(out, index=False)
    print(f"Template -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
