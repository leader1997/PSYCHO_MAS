#!/usr/bin/env python3
"""Standalone runner for notebook 03_rag_knowledge_base.ipynb."""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

OUTPUT_DIR = Path("outputs")
KB_DIR = OUTPUT_DIR / "kb"
PDF_DIR = Path("literature/pdfs")
INVENTORY = Path("literature_inventory.csv")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

FEATURE_DICT = {
    "age": "Respondent age in years.",
    "gender_male": "Indicator for male gender identity.",
    "gender_female": "Indicator for female gender identity.",
    "international": "Indicator for international student status.",
    "fincur": "Current financial situation / stress (higher = more strain).",
    "finpast": "Financial situation while growing up.",
    "food_worry": "Worry about running out of food (food insecurity).",
    "sleep_wknight": "Hours of sleep on a typical weeknight.",
    "sleep_wkend": "Hours of sleep on a typical weekend night.",
    "exerc": "Frequency of physical exercise.",
    "belong1": "Sense of belonging on campus (item 1).",
    "belong2": "Sense of belonging on campus (item 2).",
    "lonely": "Self-reported loneliness.",
    "aca_impa": "Academic impairment due to emotional/mental health.",
    "flourish": "Flourishing / positive psychological functioning scale.",
    "positiveMH": "Positive mental health indicator.",
    "sub_any": "Any substance use indicator.",
    "binge1pl": "Binge drinking (one or more occasions).",
    "smok_cig": "Cigarette smoking indicator.",
    "discrim_race": "Experiences of racial discrimination.",
    "discrim_gender": "Experiences of gender discrimination.",
    "discrim_sexual": "Experiences of sexual-orientation discrimination.",
    "inst_type": "Institution type.",
    "inst_size": "Institution size.",
    "inst_public": "Whether the institution is public.",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if len(piece) >= 80:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_project_docs() -> list[dict]:
    docs: list[dict] = []

    def add(doc_id, doc_type, title, text):
        docs.append({"id": doc_id, "type": doc_type, "title": title, "text": text})

    add(
        "dataset_overview",
        "dataset",
        "Healthy Minds Study 2024-2025 dataset",
        "The Healthy Minds Study (HMS) is a large annual survey of student mental health at US "
        "colleges and universities. This pipeline uses the 2024-2025 wave with institutional "
        "characteristics. The classification target is dep_or_anx: 1 indicates elevated depression "
        "and/or anxiety (screening positive), 0 otherwise. After cleaning (finished surveys, valid "
        "target, at least 60 percent of features present) there are about 63,581 respondents with a "
        "44.9 percent positive rate. Twenty-five features cover demographics, financial stress, sleep "
        "and lifestyle, social support, academics, substance use, discrimination, and institution.",
    )

    for f, desc in FEATURE_DICT.items():
        add(f"feature_{f}", "feature", f"Feature: {f}", f"{f}: {desc}")

    for path in sorted(glob.glob(str(OUTPUT_DIR / "cards" / "*.json"))):
        card = json.load(open(path, encoding="utf-8"))
        add(
            f"card_{Path(path).stem}",
            card.get("card_type", "card"),
            Path(path).stem.replace("_", " "),
            json.dumps(card, indent=2),
        )

    agg_path = OUTPUT_DIR / "cards" / "aggregated_explanation.json"
    if agg_path.exists():
        agg = json.load(open(agg_path, encoding="utf-8"))
        summary = (
            "Across all nine models, the most important feature groups for depression-risk "
            "classification are: "
            + "; ".join(
                f"{g['group']} (importance {g['importance']}, {g['agreement']} agreement)"
                for g in agg["top_feature_groups"][:5]
            )
            + ". "
            + agg["final_insight"]
        )
        add("aggregated_importance_summary", "explanation", "Aggregated feature importance summary", summary)

    add(
        "domain_depression_drivers",
        "domain",
        "Domain knowledge: correlates of student depression",
        "Research consistently links college student depression and anxiety with loneliness and low "
        "social belonging, academic impairment and stress, financial strain and food insecurity, poor "
        "sleep, low positive mental health/flourishing, substance use, and experiences of discrimination. "
        "These are associations in observational data, not proven causes.",
    )

    add(
        "ethics_limitations",
        "ethics",
        "Ethical limitations and safe-use guidance",
        "This system is NOT a diagnostic tool and must not replace mental-health professionals. "
        "Predictions are screening signals that may contain bias, especially for sensitive features "
        "such as gender, race, and sexual-orientation discrimination. Sensitive features must be handled "
        "carefully and reported at group level, never used for individual screening without clinical "
        "validation. The system must preserve privacy and de-identification. Any LLM answer must be "
        "grounded only in retrieved evidence and must state uncertainty and model disagreement.",
    )

    return docs


def ingest_pdfs(docs: list[dict], pdf_dir: Path) -> tuple[int, list[dict]]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir.resolve()}")
        return 0, []

    inventory = {}
    if INVENTORY.exists():
        inv = pd.read_csv(INVENTORY).drop_duplicates(subset=["file_name"], keep="first")
        inventory = {row["file_name"]: row for _, row in inv.iterrows()}

    pdf_stats = []
    n_chunks = 0
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            pages = [clean_text(page.extract_text() or "") for page in reader.pages]
            full_text = clean_text(" ".join(pages))
            chunks = chunk_text(full_text)
            stem = pdf_path.stem
            meta = inventory.get(pdf_path.name, {})
            display_title = meta.get("title", stem) if isinstance(meta, dict) else stem
            for i, chunk in enumerate(chunks):
                docs.append(
                    {
                        "id": f"lit_{stem}_chunk_{i:04d}",
                        "type": "literature",
                        "title": f"{display_title} (part {i + 1}/{len(chunks)})",
                        "text": chunk,
                        "source_file": pdf_path.name,
                    }
                )
            n_chunks += len(chunks)
            pdf_stats.append({"file": pdf_path.name, "title": display_title, "chunks": len(chunks)})
            print(f"  {pdf_path.name}: {len(chunks)} chunks", flush=True)
        except Exception as exc:
            print(f"  SKIP {pdf_path.name}: {exc}", flush=True)

    return n_chunks, pdf_stats


def main() -> int:
    print(f"Working directory: {Path.cwd()}")
    KB_DIR.mkdir(parents=True, exist_ok=True)

    docs = build_project_docs()
    print(f"Built {len(docs)} project KB documents")

    print(f"PDF directory: {PDF_DIR.resolve()}")
    n_lit, pdf_stats = ingest_pdfs(docs, PDF_DIR)
    print(f"Added {n_lit} literature chunks from {len(list(PDF_DIR.glob('*.pdf')))} PDFs")

    texts = [d["text"] for d in docs]
    print("Embedding with all-MiniLM-L6-v2...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    with open(KB_DIR / "documents.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    np.save(KB_DIR / "embeddings.npy", embeddings)

    type_counts = pd.Series([d["type"] for d in docs]).value_counts()
    stats = {
        "total_documents": len(docs),
        "literature_chunks": int(type_counts.get("literature", 0)),
        "embedding_dim": int(embeddings.shape[1]),
        "pdf_count": len(pdf_stats),
        "corpus": "literature_inventory.csv candidates",
    }
    with open(KB_DIR / "kb_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if pdf_stats:
        pd.DataFrame(pdf_stats).sort_values("chunks", ascending=False).to_csv(
            KB_DIR / "pdf_inventory.csv", index=False
        )

    print(f"Saved embeddings {embeddings.shape} and {len(docs)} documents to {KB_DIR}")
    print(type_counts)
    print("DONE run_nb3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
