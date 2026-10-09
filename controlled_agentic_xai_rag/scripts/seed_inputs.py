"""Copy case profiles and XAI cards from Phase 2."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT.parent / "explainability_augmented_rag"
OUT = ROOT / "data" / "inputs"
BENCH = ROOT / "data" / "benchmarks"

COPIES = [
    (PHASE2 / "data" / "case_profiles.json", OUT / "case_profiles.json"),
    (PHASE2 / "xai" / "xai_cards.jsonl", OUT / "xai_cards.jsonl"),
    (PHASE2 / "xai" / "xai_card_schema.json", OUT / "xai_card_schema.json"),
    (PHASE2 / "data" / "rag_benchmark_100_questions.csv", BENCH / "rag_benchmark_100_questions.csv"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    for src, dst in COPIES:
        if not src.exists():
            print(f"SKIP missing: {src}")
            continue
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")


if __name__ == "__main__":
    main()
