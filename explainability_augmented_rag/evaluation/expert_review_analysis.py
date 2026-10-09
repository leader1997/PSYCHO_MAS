"""Analyze completed expert review spreadsheets (WP12)."""

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


def analyze_reviews(review_csv: Path | None = None) -> pd.DataFrame:
    review_csv = review_csv or path("paper_outputs") / "expert_reviews_completed.csv"
    if not review_csv.exists():
        print(f"No completed reviews at {review_csv} — fill expert_review_template.xlsx first.")
        return pd.DataFrame()
    df = pd.read_csv(review_csv)
    summary = df[CRITERIA].agg(["mean", "std"]).T.round(3)
    out = path("results_dir") / "expert_review_summary.csv"
    summary.to_csv(out)
    print(f"Expert review summary -> {out}")
    return summary


if __name__ == "__main__":
    analyze_reviews()
