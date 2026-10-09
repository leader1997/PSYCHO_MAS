#!/usr/bin/env python3
"""One-time seed from xai_rag_depression phase-1 outputs into standalone Phase 2 project."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASE1 = ROOT.parent / "xai_rag_depression"
PDFS_ALT = ROOT.parent.parent / "pdfs_psycho_rec"


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied: {src.name} -> {dst.relative_to(ROOT)}")
    return True


def _copy_tree_glob(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_dir.glob(pattern):
        shutil.copy2(f, dst_dir / f.name)
        n += 1
    if n:
        print(f"  copied {n} files: {src_dir.name}/{pattern} -> {dst_dir.relative_to(ROOT)}")
    return n


def main() -> int:
    print(f"=== Seed Phase 2 from {PHASE1} ===")
    if not PHASE1.exists():
        print(f"ERROR: phase-1 not found at {PHASE1}")
        return 1

    data = ROOT / "data"
    artifacts = data / "project_artifacts"
    raw_pdfs = data / "raw_pdfs"
    xai_dir = ROOT / "xai"

    # PDFs
    n_pdf = _copy_tree_glob(PHASE1 / "literature" / "pdfs", raw_pdfs, "*.pdf")
    if n_pdf == 0:
        n_pdf = _copy_tree_glob(PDFS_ALT, raw_pdfs, "*.pdf")
    if n_pdf == 0:
        print("  WARN: no PDFs found — corpus will use project artifacts + documents.json only")

    # Project artifacts
    for sub in ("cards", "kb"):
        src = PHASE1 / "outputs" / sub
        if sub == "cards":
            _copy_tree_glob(src, artifacts / "cards", "*.json")
        else:
            for name in ("documents.json", "kb_stats.json", "pdf_inventory.csv"):
                _copy_file(src / name, artifacts / name)

    # Other seeds
    seeds = [
        (PHASE1 / "outputs" / "case_profiles.json", data / "case_profiles.json"),
        (PHASE1 / "outputs" / "rag_eval_benchmark.csv", data / "rag_eval_benchmark_phase1.csv"),
        (PHASE1 / "outputs" / "rag_eval_benchmark.json", data / "rag_eval_benchmark_phase1.json"),
        (PHASE1 / "literature_inventory.csv", data / "literature_inventory.csv"),
        (PHASE1 / "reproducibility" / "FEATURE_DICTIONARY.md", data / "FEATURE_DICTIONARY.md"),
        (PHASE1 / "outputs" / "local_explanations.json", artifacts / "local_explanations.json"),
        (PHASE1 / "outputs" / "aggregated_importance.csv", artifacts / "aggregated_importance.csv"),
    ]
    for src, dst in seeds:
        _copy_file(src, dst)

    # Phase-1 chunk ID mapping helper
    docs_path = artifacts / "documents.json"
    if docs_path.exists():
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        mapping = {d["id"]: d.get("title", d["id"]) for d in docs}
        (data / "phase1_doc_id_mapping.json").write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  wrote phase1_doc_id_mapping.json ({len(mapping)} docs)")

    # Placeholder xai schema copy location
    xai_dir.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "paper_outputs").mkdir(exist_ok=True)

    print("\nSeed complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
