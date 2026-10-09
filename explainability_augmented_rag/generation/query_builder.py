"""Query construction from case + XAI + benchmark (WP3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import path  # noqa: E402
from xai.build_xai_cards import load_xai_cards  # noqa: E402


def _factor_names(factors: list[dict], n: int = 3) -> str:
    return ", ".join(f["display_name"] for f in factors[:n])


def build_case_queries(case_id: str, card: dict) -> list[dict]:
    pos = card.get("top_positive_factors", [])
    prot = card.get("top_protective_factors", [])
    label = card.get("prediction_label", "elevated_risk")
    pos_names = _factor_names(pos) or "survey features"
    prot_names = _factor_names(prot) or "protective survey features"

    queries = [
        {
            "case_id": case_id,
            "query_type": "risk_communication",
            "query_text": (
                "Retrieve evidence about how elevated depression/anxiety screening risk should be "
                "communicated in a non-diagnostic, non-clinical way to college students."
            ),
        },
        {
            "case_id": case_id,
            "query_type": "feature_specific",
            "query_text": (
                f"Retrieve evidence about how {pos_names} are associated with student "
                f"depression/anxiety screening risk in survey-based studies."
            ),
        },
        {
            "case_id": case_id,
            "query_type": "protective_factor",
            "query_text": (
                f"Retrieve evidence about protective factors such as {prot_names} "
                f"and help-seeking or social support for student mental health."
            ),
        },
        {
            "case_id": case_id,
            "query_type": "safety",
            "query_text": (
                "Retrieve evidence about safe, non-stigmatizing, non-clinical mental-health "
                "communication and ethical limitations of screening tools."
            ),
        },
    ]
    if label == "elevated_risk":
        queries.append(
            {
                "case_id": case_id,
                "query_type": "referral",
                "query_text": (
                    "Retrieve evidence about appropriate next steps and referrals when a student "
                    "shows elevated depression/anxiety screening risk."
                ),
            }
        )
    return queries


def build_benchmark_query(question: str, category: str, topics: str) -> str:
    return f"{question} Context: {category}. Topics: {topics}"


def build_all_case_queries() -> pd.DataFrame:
    rows = []
    for card in load_xai_cards():
        rows.extend(build_case_queries(card["case_id"], card))
    return pd.DataFrame(rows)


def main() -> int:
    df = build_all_case_queries()
    out = path("results_dir") / "generated_queries.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} queries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
