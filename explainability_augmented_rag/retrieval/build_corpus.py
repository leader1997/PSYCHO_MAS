"""Corpus building — section-aware chunking with rich metadata (WP1)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get, ensure_dirs, load_config, path  # noqa: E402

FEATURE_DISPLAY = {
    "age": "Age",
    "gender_male": "Male gender",
    "gender_female": "Female gender",
    "international": "International student status",
    "fincur": "Current financial situation",
    "finpast": "Past financial stress",
    "food_worry": "Food insecurity worry",
    "sleep_wknight": "Weeknight sleep",
    "sleep_wkend": "Weekend sleep",
    "exerc": "Exercise frequency",
    "belong1": "Campus belonging (item 1)",
    "belong2": "Campus belonging (item 2)",
    "lonely": "Loneliness",
    "aca_impa": "Academic impairment",
    "flourish": "Flourishing",
    "positiveMH": "Positive mental health",
    "sub_any": "Substance use",
    "binge1pl": "Binge drinking",
    "smok_cig": "Cigarette smoking",
    "discrim_race": "Racial discrimination",
    "discrim_gender": "Gender discrimination",
    "discrim_sexual": "Sexual-orientation discrimination",
    "inst_type": "Institution type",
    "inst_size": "Institution size",
    "inst_public": "Public institution",
}

DOMAIN_KEYWORDS = {
    "depression": ["depression", "depressive", "mood"],
    "anxiety": ["anxiety", "worry", "panic"],
    "student mental health": ["student", "college", "campus", "university"],
    "screening": ["screening", "screener", "survey"],
    "XAI": ["explainab", "shap", "interpret"],
    "ethics": ["ethic", "fairness", "bias", "responsible"],
    "safety": ["safety", "crisis", "referral", "non-clinical"],
    "RAG evaluation": ["retrieval", "rag", "faithfulness"],
}


def _tokens_to_chars(n: int) -> int:
    cpt = cfg_get("corpus", "chars_per_token", default=4)
    return n * cpt


def clean_text(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_domain_tags(text: str, source_type: str) -> list[str]:
    lower = text.lower()
    tags = []
    for tag, kws in DOMAIN_KEYWORDS.items():
        if any(k in lower for k in kws):
            tags.append(tag)
    if source_type in ("xai_report", "model_report"):
        tags.append("XAI")
    if source_type == "safety_document":
        tags.append("safety")
    return sorted(set(tags)) or ["student mental health"]


def infer_year(stem: str, title: str) -> int | None:
    for s in (stem, title):
        m = re.search(r"(20\d{2})", s)
        if m:
            return int(m.group(1))
    return None


def chunk_by_tokens(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    size = _tokens_to_chars(target_tokens)
    overlap = _tokens_to_chars(overlap_tokens)
    min_chars = cfg_get("corpus", "min_chunk_chars", default=120)
    text = clean_text(text)
    if not text:
        return []
    chunks: list[str] = []
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


def parse_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if len(cleaned) >= 40:
            pages.append((i, cleaned))
    return pages


def make_doc_id(stem: str, idx: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", stem)[:40]
    return f"DOC{idx:03d}_{safe}"


def process_pdfs(pdf_dir: Path, doc_start: int = 1) -> tuple[list[dict], list[dict], int]:
    chunks: list[dict] = []
    metadata: list[dict] = []
    doc_idx = doc_start
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        stem = pdf_path.stem
        doc_id = make_doc_id(stem, doc_idx)
        title = stem.replace("_", " ")
        year = infer_year(stem, title)
        pages = parse_pdf_pages(pdf_path)
        if not pages:
            continue
        full_text = " ".join(t for _, t in pages)
        metadata.append(
            {
                "doc_id": doc_id,
                "title": title,
                "year": year,
                "source_type": "peer_reviewed_literature",
                "file_name": pdf_path.name,
                "page_count": len(pages),
                "char_count": len(full_text),
            }
        )
        section = "body"
        chunk_num = 0
        for page_num, page_text in pages:
            for piece in chunk_by_tokens(
                page_text,
                cfg_get("corpus", "chunk_target_tokens", default=450),
                cfg_get("corpus", "chunk_overlap_tokens", default=75),
            ):
                chunk_num += 1
                chunk_id = f"{doc_id}_p{page_num:02d}_c{chunk_num:03d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "title": title,
                        "year": year,
                        "page": page_num,
                        "section": section,
                        "source_type": "peer_reviewed_literature",
                        "domain_tags": infer_domain_tags(piece, "peer_reviewed_literature"),
                        "text": piece,
                        "phase1_id": f"lit_{stem}_chunk_{chunk_num - 1:04d}",
                    }
                )
        doc_idx += 1
    return chunks, metadata, doc_idx


def _map_source_type(doc_type: str) -> str:
    mapping = {
        "literature": "peer_reviewed_literature",
        "feature": "project_artifact",
        "dataset": "project_artifact",
        "card": "xai_report",
        "ethics": "safety_document",
        "domain": "project_artifact",
        "explanation": "xai_report",
        "model_performance": "model_report",
        "local_explanation": "xai_report",
        "aggregated_explanation": "xai_report",
        "global_feature_importance": "xai_report",
    }
    return mapping.get(doc_type, "project_artifact")


def process_project_artifacts(artifacts_dir: Path, doc_start: int) -> tuple[list[dict], list[dict], int]:
    chunks: list[dict] = []
    metadata: list[dict] = []
    doc_idx = doc_start
    docs_json = artifacts_dir / "documents.json"
    if not docs_json.exists():
        return chunks, metadata, doc_idx

    docs = json.loads(docs_json.read_text(encoding="utf-8"))
    for doc in docs:
        old_id = doc.get("id", f"artifact_{doc_idx}")
        doc_id = f"ART{doc_idx:03d}_{old_id[:30]}"
        title = doc.get("title", old_id)
        text = doc.get("text", "")
        doc_type = doc.get("type", "project_artifact")
        source_type = _map_source_type(doc_type)
        metadata.append(
            {
                "doc_id": doc_id,
                "title": title,
                "year": None,
                "source_type": source_type,
                "file_name": old_id,
                "page_count": 1,
                "char_count": len(text),
            }
        )
        for i, piece in enumerate(
            chunk_by_tokens(
                text,
                cfg_get("corpus", "chunk_target_tokens", default=450),
                cfg_get("corpus", "chunk_overlap_tokens", default=75),
            ),
            start=1,
        ):
            chunk_id = f"{doc_id}_p01_c{i:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "year": None,
                    "page": 1,
                    "section": doc_type,
                    "source_type": source_type,
                    "domain_tags": infer_domain_tags(piece, source_type),
                    "text": piece,
                    "phase1_id": old_id if i == 1 else f"{old_id}_part",
                }
            )
        doc_idx += 1
    return chunks, metadata, doc_idx


def build_corpus() -> pd.DataFrame:
    ensure_dirs()
    pdf_dir = path("raw_pdfs")
    artifacts_dir = path("project_artifacts")

    all_chunks: list[dict] = []
    all_meta: list[dict] = []

    pdf_chunks, pdf_meta, next_idx = process_pdfs(pdf_dir, doc_start=1)
    all_chunks.extend(pdf_chunks)
    all_meta.extend(pdf_meta)

    art_chunks, art_meta, _ = process_project_artifacts(artifacts_dir, doc_start=next_idx)
    all_chunks.extend(art_chunks)
    all_meta.extend(art_meta)

    if not all_chunks:
        raise RuntimeError("No chunks produced — run scripts/seed_from_phase1.py first")

    df = pd.DataFrame(all_chunks)
    if df["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk_id detected")

    meta_df = pd.DataFrame(all_meta)
    out_csv = path("processed_chunks_csv")
    out_jsonl = path("processed_chunks_jsonl")
    meta_csv = path("corpus_metadata_csv")

    df.to_csv(out_csv, index=False)
    meta_df.to_csv(meta_csv, index=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Chunks: {len(df)} | Docs: {len(meta_df)}")
    print(f"  -> {out_csv}")
    print(f"  -> {out_jsonl}")
    print(f"  -> {meta_csv}")
    return df


def main() -> int:
    load_config()
    build_corpus()
    return 0


if __name__ == "__main__":
    sys.exit(main())
