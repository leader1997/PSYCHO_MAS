"""Build corpus from knowledge_db PDFs (450/75 chunking)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT.parent / "knowledge_db"
PHASE2 = ROOT.parent / "explainability_augmented_rag"


def load_cfg() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens_to_chars(n: int, cfg: dict) -> int:
    return n * cfg["corpus"]["chars_per_token"]


def chunk_by_tokens(text: str, cfg: dict) -> list[str]:
    size = tokens_to_chars(cfg["corpus"]["chunk_target_tokens"], cfg)
    overlap = tokens_to_chars(cfg["corpus"]["chunk_overlap_tokens"], cfg)
    min_chars = cfg["corpus"]["min_chunk_chars"]
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            break_at = text.rfind(". ", start, end)
            if break_at > start + min_chars:
                end = break_at + 1
        piece = text[start:end].strip()
        if len(piece) >= min_chars:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def infer_year(title: str, meta_year: str) -> str:
    if meta_year:
        return str(meta_year)
    m = re.search(r"(20\d{2})", title)
    return m.group(1) if m else ""


def load_doc_meta() -> dict[str, dict]:
    meta_path = KB / "metadata" / "selected_documents.jsonl"
    out = {}
    if not meta_path.exists():
        return out
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def find_pdfs() -> list[Path]:
    pdfs = []
    for base in (KB / "pdfs", ROOT / "data" / "raw_pdfs"):
        if base.exists():
            pdfs.extend(base.rglob("*.pdf"))
    return sorted(set(pdfs))


def process_pdf(pdf_path: Path, doc_meta: dict) -> tuple[list[dict], dict | None]:
    stem = pdf_path.stem
    doc_id = stem if stem.startswith("DOC_") else f"DOC_{stem}"
    meta = doc_meta.get(doc_id, {})
    title = meta.get("title", stem.replace("_", " "))[:200]
    year = infer_year(title, meta.get("year", ""))
    expertise = meta.get("expertise_area", "")
    source_type = meta.get("source_type", "peer_reviewed")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return [], None

    chunks = []
    chunk_num = 0
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if len(page_text) < 40:
            continue
        section = "body"
        for piece in chunk_by_tokens(page_text, load_cfg()):
            chunk_num += 1
            chunk_id = f"{doc_id}_p{page_num:02d}_c{chunk_num:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "title": title,
                "year": year,
                "expertise_area": expertise,
                "page": page_num,
                "section_heading": section,
                "source_type": source_type,
                "text": piece,
            })
    if not chunks:
        return [], None
    return chunks, {
        "doc_id": doc_id,
        "title": title,
        "year": year,
        "expertise_area": expertise,
        "file_name": pdf_path.name,
        "chunk_count": len(chunks),
    }


def add_artifact_chunks(cfg: dict) -> list[dict]:
    chunks = []
    feat = PHASE2 / "data" / "FEATURE_DICTIONARY.md"
    if feat.exists():
        text = feat.read_text(encoding="utf-8")[:3000]
        for i, piece in enumerate(chunk_by_tokens(text, cfg), start=1):
            chunks.append({
                "chunk_id": f"ART_FEATURE_p01_c{i:03d}",
                "doc_id": "ART_FEATURE",
                "title": "Feature Dictionary",
                "year": "",
                "expertise_area": "XAI / methodology",
                "page": 1,
                "section_heading": "features",
                "source_type": "project_artifact",
                "text": piece,
            })
    safety_text = (
        "Non-clinical screening communication must avoid diagnosis, treatment advice, "
        "and stigmatizing language. Model explanations reflect survey feature contributions, not causes."
    )
    chunks.append({
        "chunk_id": "ART_SAFETY_p01_c001",
        "doc_id": "ART_SAFETY",
        "title": "Safety Communication Guidelines",
        "year": "",
        "expertise_area": "Screening communication",
        "page": 1,
        "section_heading": "safety",
        "source_type": "safety_document",
        "text": safety_text,
    })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback-phase2", action="store_true", help="Copy Phase 2 corpus if no PDFs")
    args = parser.parse_args()
    cfg = load_cfg()
    out_csv = ROOT / cfg["paths"]["processed_chunks_csv"]
    out_jsonl = ROOT / cfg["paths"]["processed_chunks_jsonl"]
    meta_csv = ROOT / cfg["paths"]["corpus_metadata_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    doc_meta = load_doc_meta()
    all_chunks: list[dict] = []
    all_meta: list[dict] = []
    pdfs = find_pdfs()
    if not pdfs and args.fallback_phase2:
        src = PHASE2 / "data" / "processed_chunks.csv"
        if src.exists():
            shutil.copy2(src, out_csv)
            shutil.copy2(PHASE2 / "data" / "processed_chunks.jsonl", out_jsonl)
            print(f"Fallback: copied Phase 2 corpus ({src})")
            return
    for pdf in pdfs:
        chunks, meta = process_pdf(pdf, doc_meta)
        if chunks:
            all_chunks.extend(chunks)
            if meta:
                all_meta.append(meta)
    all_chunks.extend(add_artifact_chunks(cfg))
    if not all_chunks:
        raise RuntimeError("No chunks built. Run download_pdfs.py or use --fallback-phase2")

    df = pd.DataFrame(all_chunks)
    df.to_csv(out_csv, index=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(all_meta).to_csv(meta_csv, index=False)
    print(f"Built {len(all_chunks)} chunks from {len(pdfs)} PDFs -> {out_csv}")

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from controlled_agentic_xai_rag.retrieval.index_sync import record_corpus_built

    fp = record_corpus_built(out_csv)
    print(f"Corpus fingerprint updated; vector indexes invalidated if corpus changed ({fp[:12]}…)")


if __name__ == "__main__":
    main()
