"""Remap phase-1 gold chunk IDs to phase-2 chunk IDs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import path  # noqa: E402
from retrieval.base import load_chunks  # noqa: E402


def remap_gold_id(old_id: str, chunks: pd.DataFrame) -> str | None:
    if pd.isna(old_id) or not str(old_id).strip():
        return None
    old_id = str(old_id).strip()
    # Direct phase1_id match
    match = chunks[chunks["phase1_id"] == old_id]
    if not match.empty:
        return match.iloc[0]["chunk_id"]
    # Doc-level match (feature_*, ethics_*, etc.)
    match = chunks[chunks["phase1_id"].str.startswith(old_id, na=False)]
    if not match.empty:
        return match.iloc[0]["chunk_id"]
    # Literature chunk: lit_STEM_chunk_NNNN
    if old_id.startswith("lit_") and "_chunk_" in old_id:
        stem = old_id.rsplit("_chunk_", 1)[0]
        sub = chunks[chunks["phase1_id"].str.contains(stem.replace("lit_", ""), na=False, regex=False)]
        if not sub.empty:
            return sub.iloc[0]["chunk_id"]
        doc_match = chunks[chunks["doc_id"].str.contains(stem.split("_", 2)[-1][:20], na=False, regex=False)]
        if not doc_match.empty:
            return doc_match.iloc[0]["chunk_id"]
    # Partial doc id
    partial = chunks[chunks["phase1_id"].str.contains(old_id[:20], na=False, regex=False)]
    if not partial.empty:
        return partial.iloc[0]["chunk_id"]
    return None


def remap_benchmark() -> pd.DataFrame:
    src = ROOT / "data" / "rag_eval_benchmark_phase1.csv"
    if not src.exists():
        raise FileNotFoundError(f"Run seed script first: {src}")
    chunks = load_chunks()
    df = pd.read_csv(src)
    new_ids = []
    for _, row in df.iterrows():
        old_list = str(row.get("gold_relevant_chunk_ids", "")).split("|")
        mapped = []
        for oid in old_list:
            oid = oid.strip()
            if not oid:
                continue
            nid = remap_gold_id(oid, chunks)
            if nid:
                mapped.append(nid)
        new_ids.append("|".join(dict.fromkeys(mapped)))
    df["expected_relevant_chunk_ids"] = new_ids
    df = df.rename(columns={"expected_answer_points": "expected_answer"})
    out_csv = path("benchmark_csv")
    out_jsonl = path("benchmark_jsonl")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    hit_rate = (df["expected_relevant_chunk_ids"].str.len() > 0).mean()
    print(f"Remapped benchmark -> {out_csv} (non-empty gold: {hit_rate:.1%})")
    return df


def main() -> int:
    remap_benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
