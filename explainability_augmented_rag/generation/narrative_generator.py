"""Safe narrative generation (WP6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get, path  # noqa: E402
from generation.evidence_synthesis import load_case_profiles, case_summary  # noqa: E402
from generation.llm import structured_json  # noqa: E402
from generation.prompts import NARRATIVE_SYSTEM, REVISION_SYSTEM  # noqa: E402
from generation.schemas import NARRATIVE_SCHEMA  # noqa: E402
from xai.build_xai_cards import load_xai_cards  # noqa: E402


def generate_narrative(card: dict, synthesis: dict, profiles: dict) -> dict:
    case_id = card["case_id"]
    disclaimer = cfg_get("safety", "mandatory_disclaimer")
    referral = cfg_get("safety", "elevated_risk_referral")
    user = f"""Case summary: {case_summary(case_id, profiles)}

Prediction: {card['prediction_label']} (p={card['prediction_probability']})

XAI card:
{json.dumps(card, indent=2)}

Evidence synthesis:
{json.dumps({k: v for k, v in synthesis.items() if not k.startswith('_')}, indent=2)}

Mandatory disclaimer to include: {disclaimer}
Referral guidance for elevated risk: {referral}

Generate the structured screening narrative for case_id={case_id}."""
    return structured_json(NARRATIVE_SYSTEM, user, NARRATIVE_SCHEMA, "narrative")


def revise_narrative(narrative: dict, comments: list[str]) -> dict:
    user = f"""Original narrative:
{json.dumps(narrative, indent=2)}

Validator comments to fix:
{chr(10).join('- ' + c for c in comments)}

Return corrected JSON."""
    return structured_json(REVISION_SYSTEM, user, NARRATIVE_SCHEMA, "narrative")


def run_generation(limit: int | None = None) -> list[dict]:
    cards = load_xai_cards()
    profiles = load_case_profiles()
    synth_path = path("results_dir") / "evidence_summaries.jsonl"
    if not synth_path.exists():
        from generation.evidence_synthesis import run_synthesis

        run_synthesis(limit=limit)
    syntheses = {}
    with open(synth_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                syntheses[s["case_id"]] = s

    if limit:
        cards = cards[:limit]
    results = []
    for card in cards:
        cid = card["case_id"]
        print(f"Generating narrative {cid}...", flush=True)
        syn = syntheses.get(cid, {"case_id": cid, "supported_claims": []})
        results.append(generate_narrative(card, syn, profiles))

    out = path("results_dir") / "generated_narratives.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} narratives -> {out}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_generation(limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
