"""Post-generation safety validator with revision loop (WP7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get, path  # noqa: E402
from generation.llm import structured_json  # noqa: E402
from generation.narrative_generator import revise_narrative  # noqa: E402
from generation.prompts import SAFETY_VALIDATOR_SYSTEM  # noqa: E402
from generation.schemas import SAFETY_REPORT_SCHEMA  # noqa: E402
from safety.clinical_boundary_check import (  # noqa: E402
    clinical_boundary_flags,
    has_disclaimer,
    has_referral,
)
from safety.moderation_check import moderation_flagged  # noqa: E402
from xai.build_xai_cards import load_xai_cards  # noqa: E402


def _narrative_text(narrative: dict) -> str:
    parts = [
        narrative.get("screening_summary", ""),
        narrative.get("model_explanation", ""),
        narrative.get("evidence_grounding", ""),
        narrative.get("recommended_next_step", ""),
        narrative.get("limitations", ""),
    ]
    return " ".join(parts)


def validate_narrative(narrative: dict, elevated: bool) -> dict:
    text = _narrative_text(narrative)
    flags = clinical_boundary_flags(text)
    comments = []
    checks = {}

    for key, hits in flags.items():
        status = "fail" if hits else "pass"
        checks[key] = status
        if hits:
            comments.append(f"{key}: matched patterns {hits}")

    checks["missing_disclaimer"] = "fail" if not has_disclaimer(text) else "pass"
    if checks["missing_disclaimer"] == "fail":
        comments.append("Missing non-diagnostic disclaimer")

    checks["missing_referral"] = "fail" if not has_referral(text, elevated) else "pass"
    if checks["missing_referral"] == "fail":
        comments.append("Missing professional referral for elevated risk")

    flagged, mod_msg = moderation_flagged(text)
    checks["privacy_leakage"] = "pass"
    if flagged:
        checks["stigmatizing_language"] = "fail"
        comments.append(mod_msg)

    # LLM judge for unsupported claims
    llm_report = structured_json(
        SAFETY_VALIDATOR_SYSTEM,
        f"Narrative JSON:\n{json.dumps(narrative, indent=2)}\n\nCitations: {narrative.get('citations', [])}",
        SAFETY_REPORT_SCHEMA,
        "safety_report",
    )
    checks["unsupported_claims"] = llm_report.get("unsupported_claims", "pass")
    if llm_report.get("comments"):
        comments.extend(llm_report["comments"])

    fail_keys = [k for k, v in checks.items() if v == "fail"]
    overall = "pass" if not fail_keys else "fail"
    return {
        "case_id": narrative.get("case_id", ""),
        **checks,
        "overall_safety": overall,
        "revision_required": overall == "fail",
        "comments": comments,
    }


def run_validation(max_retries: int | None = None) -> pd.DataFrame:
    max_retries = max_retries or cfg_get("llm", "max_retries", default=2)
    narr_path = path("results_dir") / "generated_narratives.jsonl"
    if not narr_path.exists():
        from generation.narrative_generator import run_generation

        run_generation()

    cards = {c["case_id"]: c for c in load_xai_cards()}
    reports = []
    corrected = []

    with open(narr_path, encoding="utf-8") as f:
        narratives = [json.loads(line) for line in f if line.strip()]

    for narrative in narratives:
        cid = narrative["case_id"]
        elevated = cards.get(cid, {}).get("prediction_label") == "elevated_risk"
        current = narrative
        for attempt in range(max_retries + 1):
            report = validate_narrative(current, elevated)
            report["attempt"] = attempt
            reports.append(report)
            if not report["revision_required"] or attempt >= max_retries:
                corrected.append(current)
                break
            print(f"  Revising {cid} (attempt {attempt + 1})...", flush=True)
            current = revise_narrative(current, report["comments"])

    results_dir = path("results_dir")
    pd.DataFrame(reports).to_csv(results_dir / "safety_reports.csv", index=False)
    with open(results_dir / "corrected_narratives.jsonl", "w", encoding="utf-8") as f:
        for n in corrected:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"Safety reports -> {results_dir / 'safety_reports.csv'}")
    return pd.DataFrame(reports)


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_validation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
