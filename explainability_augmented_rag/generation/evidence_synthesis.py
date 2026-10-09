"""Evidence synthesis module (WP5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get, path  # noqa: E402
from generation.llm import structured_json  # noqa: E402
from generation.prompts import EVIDENCE_SYNTHESIS_SYSTEM  # noqa: E402
from generation.query_builder import build_case_queries  # noqa: E402
from generation.schemas import EVIDENCE_SYNTHESIS_SCHEMA  # noqa: E402
from retrieval.reranker import get_retriever  # noqa: E402
from xai.build_xai_cards import load_xai_cards  # noqa: E402


def load_case_profiles() -> dict:
    p = ROOT / "data" / "case_profiles.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def case_summary(case_id: str, profiles: dict) -> str:
    prof = profiles.get(case_id, {})
    levels = prof.get("feature_levels", {})
    parts = [f"{k}: {v}" for k, v in list(levels.items())[:8]]
    return f"De-identified student case ({case_id}). Feature levels: " + "; ".join(parts)


def synthesize_case(card: dict, profiles: dict, retriever_name: str = "hybrid_rerank") -> dict:
    case_id = card["case_id"]
    retriever = get_retriever(retriever_name)
    queries = build_case_queries(case_id, card)
    all_hits = []
    seen = set()
    for q in queries:
        for h in retriever.retrieve(q["query_text"], k=5):
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                all_hits.append(h)
    all_hits = all_hits[:8]

    chunks_block = "\n\n".join(
        f"[{h.chunk_id}] ({h.title}, p.{h.page}): {h.text[:600]}" for h in all_hits
    )
    user = f"""Case summary: {case_summary(case_id, profiles)}

XAI card:
{json.dumps(card, indent=2)}

Retrieved chunks:
{chunks_block}

Synthesize supported claims linked to chunk IDs. Case ID: {case_id}"""
    result = structured_json(
        EVIDENCE_SYNTHESIS_SYSTEM, user, EVIDENCE_SYNTHESIS_SCHEMA, "evidence_synthesis"
    )
    result["_retrieved_chunk_ids"] = [h.chunk_id for h in all_hits]
    result["_retriever"] = retriever_name
    return result


def run_synthesis(limit: int | None = None, retriever_name: str = "hybrid_rerank") -> list[dict]:
    cards = load_xai_cards()
    profiles = load_case_profiles()
    if limit:
        cards = cards[:limit]
    results = []
    for card in cards:
        print(f"Synthesizing {card['case_id']}...", flush=True)
        results.append(synthesize_case(card, profiles, retriever_name))
    out = path("results_dir") / "evidence_summaries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} syntheses -> {out}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retriever", default="hybrid_rerank")
    args = parser.parse_args()
    run_synthesis(limit=args.limit, retriever_name=args.retriever)
    return 0


if __name__ == "__main__":
    sys.exit(main())
