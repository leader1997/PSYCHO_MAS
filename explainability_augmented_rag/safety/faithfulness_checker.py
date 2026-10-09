"""Citation and faithfulness checker (WP8)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import path  # noqa: E402
from generation.llm import structured_json  # noqa: E402
from generation.prompts import FAITHFULNESS_SYSTEM  # noqa: E402
from retrieval.base import load_chunks  # noqa: E402

FAITHFULNESS_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "support_status": {
            "type": "string",
            "enum": ["supported", "partially_supported", "unsupported", "contradicted"],
        },
        "explanation": {"type": "string"},
    },
    "required": ["support_status", "explanation"],
    "additionalProperties": False,
}


def verify_claim(case_id: str, claim: str, chunk_id: str, chunk_text: str) -> dict:
    user = f"Claim: {claim}\n\nCited chunk [{chunk_id}]:\n{chunk_text[:1500]}"
    result = structured_json(
        FAITHFULNESS_SYSTEM, user, FAITHFULNESS_CHECK_SCHEMA, "faithfulness"
    )
    return {
        "case_id": case_id,
        "claim": claim,
        "cited_chunk_id": chunk_id,
        "support_status": result["support_status"],
        "explanation": result["explanation"],
    }


def run_faithfulness_check() -> pd.DataFrame:
    narr_path = path("results_dir") / "corrected_narratives.jsonl"
    if not narr_path.exists():
        narr_path = path("results_dir") / "generated_narratives.jsonl"
    if not narr_path.exists():
        from generation.narrative_generator import run_generation

        run_generation()

    chunks = load_chunks().set_index("chunk_id")["text"].to_dict()
    rows = []
    with open(narr_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            narrative = json.loads(line)
            cid = narrative["case_id"]
            for cite in narrative.get("citations", []):
                chunk_id = cite.get("chunk_id", "")
                claim = cite.get("claim_supported", "")
                text = chunks.get(chunk_id, "")
                rows.append(verify_claim(cid, claim, chunk_id, text))

    df = pd.DataFrame(rows)
    results_dir = path("results_dir")
    df.to_csv(results_dir / "claim_level_faithfulness.csv", index=False)

    if not df.empty:
        summary = {
            "claim_support_rate": (df["support_status"] == "supported").mean(),
            "partial_support_rate": (df["support_status"] == "partially_supported").mean(),
            "unsupported_claim_rate": (df["support_status"] == "unsupported").mean(),
            "hallucination_rate": (df["support_status"].isin(["unsupported", "contradicted"])).mean(),
            "citation_precision": (df["support_status"].isin(["supported", "partially_supported"])).mean(),
            "n_claims": len(df),
        }
        pd.DataFrame([summary]).to_csv(results_dir / "citation_precision_results.csv", index=False)
        print(f"Faithfulness: {summary}")
    print(f"Wrote -> {results_dir / 'claim_level_faithfulness.csv'}")
    return df


def main() -> int:
    run_faithfulness_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
