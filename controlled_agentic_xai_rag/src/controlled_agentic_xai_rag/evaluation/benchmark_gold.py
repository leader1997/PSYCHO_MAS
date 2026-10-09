"""Resolve benchmark gold chunk IDs against the current corpus."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from controlled_agentic_xai_rag.config_loader import path

# Legacy benchmark doc slugs -> current corpus doc_ids (subset of corpus_metadata).
DOC_SLUG_ALIASES: dict[str, list[str]] = {
    "eisenberg": ["DOC_D003"],
    "mental_health_and_academic": ["DOC_D003"],
    "academic_success_in_college": ["DOC_D003"],
    "surgeon": ["DOC_F002", "DOC_C001"],
    "loneliness_and_isolation": ["DOC_F002", "DOC_C001"],
    "our_epidemic_of_loneliness": ["DOC_F002", "DOC_C001"],
    "lumina": ["DOC_C003", "DOC_C001"],
    "invisible_on_campus": ["DOC_C003"],
    "domain_depression_drivers": ["DOC_A001", "DOC_B001"],
    "depression_screening": ["DOC_A001", "DOC_A009"],
    "gad_7": ["DOC_B002", "DOC_B003"],
    "phq_9": ["DOC_B006", "DOC_B007"],
    "treeshap": ["DOC_H001", "DOC_H005"],
    "shap": ["DOC_H001", "DOC_H005"],
    "lime": ["DOC_H003"],
    "hallucination": ["DOC_I003", "DOC_I004"],
    "rag": ["DOC_I001", "DOC_I002"],
    "self_rag": ["DOC_I002"],
    "calibration": ["DOC_H002", "DOC_H007"],
    "interpretability": ["DOC_H004", "DOC_H006"],
    "student_wellbeing": ["DOC_Student_Wellbeing_Guide", "DOC_F004"],
    "belonging": ["DOC_F002", "DOC_F003", "DOC_C001"],
    "loneliness": ["DOC_C001", "DOC_C002", "DOC_C003"],
    "financial": ["DOC_E001", "DOC_E002"],
    "material_hardship": ["DOC_E002", "DOC_E003"],
    "help_seeking": ["DOC_G005", "DOC_A009"],
    "screening": ["DOC_A001", "DOC_A007", "DOC_G001"],
}

CATEGORY_DOC_IDS: dict[str, list[str]] = {
    "loneliness_belonging": ["DOC_C001", "DOC_C002", "DOC_C003", "DOC_F002", "DOC_F003"],
    "academic_impairment": ["DOC_D001", "DOC_D002", "DOC_D003", "DOC_D004"],
    "financial_stress": ["DOC_E001", "DOC_E002", "DOC_E003"],
    "sleep_lifestyle": ["DOC_F001", "DOC_F004", "DOC_F007"],
    "flourishing_positive_mh": ["DOC_F004", "DOC_F007", "DOC_F002"],
    "help_seeking_counseling": ["DOC_G005", "DOC_A009", "DOC_A007"],
    "ethics_not_diagnosis": ["DOC_G005", "DOC_A007", "DOC_I004"],
    "xai_explanation": ["DOC_H001", "DOC_H005", "DOC_H004", "DOC_H003"],
    "case_based_explanation": ["DOC_A001", "DOC_B001", "DOC_G005"],
    "insufficient_context_safety": ["DOC_I004", "DOC_I003", "DOC_G005"],
}

CLAIMS_COLUMNS = ["case_id", "claim", "cited_chunk_id", "support_status", "explanation"]
PRESET_CASE_IDS = ["high_risk_case", "low_risk_case", "borderline_risk_case"]


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).lower()).strip("_")
    return text


def _token_slugs(token: str) -> set[str]:
    raw = _slugify(token.replace("|", " "))
    parts = {p for p in raw.split("_") if len(p) >= 4}
    parts.add(raw)
    for alias in DOC_SLUG_ALIASES:
        if alias in raw:
            parts.add(alias)
    m = re.search(r"lit_\d{4}_(.+?)_chunk", token, re.I)
    if m:
        parts.add(_slugify(m.group(1)))
    m2 = re.match(r"(DOC[A-Z]?_\w+)", token, re.I)
    if m2:
        parts.add(_slugify(m2.group(1)))
    return parts


def _page_from_token(token: str) -> int | None:
    m = re.search(r"_p(\d+)_", token)
    return int(m.group(1)) if m else None


def _doc_ids_for_token(token: str, meta: pd.DataFrame) -> list[str]:
    if token.startswith("feature_"):
        return []

    doc_prefix = token.split("_p")[0] if "_p" in token else ""
    if doc_prefix.startswith("DOC") and doc_prefix in set(meta["doc_id"]):
        return [doc_prefix]

    for slug, doc_ids in DOC_SLUG_ALIASES.items():
        if slug in _slugify(token):
            return doc_ids

    slugs = _token_slugs(token)
    scores: dict[str, int] = {}
    for _, row in meta.iterrows():
        doc_id = str(row["doc_id"])
        hay = _slugify(f"{row.get('title', '')} {row.get('file_name', '')} {row.get('expertise_area', '')}")
        score = sum(1 for s in slugs if s in hay)
        if score:
            scores[doc_id] = score
    if scores:
        best = max(scores.values())
        return [d for d, s in scores.items() if s == best]
    return []


def _chunks_for_doc_ids(
    doc_ids: list[str],
    chunks: pd.DataFrame,
    page: int | None,
    max_per_doc: int,
) -> set[str]:
    out: set[str] = set()
    for doc_id in doc_ids:
        sub = chunks[chunks["doc_id"] == doc_id]
        if sub.empty:
            continue
        if page is not None and "page" in sub.columns:
            page_hits = sub[sub["page"] == page]
            if not page_hits.empty:
                out.update(page_hits["chunk_id"].astype(str).head(max_per_doc))
                continue
        out.update(sub["chunk_id"].astype(str).head(max_per_doc))
    return out


def _gold_id_columns(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("expected_relevant_chunk_ids", "gold_relevant_chunk_ids"):
        val = str(row.get(col, "")).strip()
        if val and val != "nan":
            parts.append(val)
    return "|".join(parts)


def resolve_gold_chunks(
    row: pd.Series,
    chunks: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    *,
    max_chunks_per_doc: int = 5,
) -> set[str]:
    """Map benchmark row to chunk_ids in the current corpus."""
    if meta is None:
        meta = load_corpus_metadata()
    corpus_ids = set(chunks["chunk_id"].astype(str))
    raw = _gold_id_columns(row)
    if not raw:
        raw = str(row.get("gold_chunk_id", row.get("expected_chunk_id", "")))

    resolved: set[str] = set()
    for token in raw.split("|"):
        token = token.strip()
        if not token or token == "nan":
            continue
        if token in corpus_ids:
            resolved.add(token)
            continue
        page = _page_from_token(token)
        doc_ids = _doc_ids_for_token(token, meta)
        if doc_ids:
            resolved |= _chunks_for_doc_ids(doc_ids, chunks, page, max_chunks_per_doc)

    if not resolved:
        category = str(row.get("category", ""))
        fallback_docs = CATEGORY_DOC_IDS.get(category, [])
        if fallback_docs:
            resolved |= _chunks_for_doc_ids(fallback_docs, chunks, None, max_chunks_per_doc)

    return resolved


@lru_cache(maxsize=1)
def load_corpus_chunks() -> pd.DataFrame:
    return pd.read_csv(path("processed_chunks_csv"))


@lru_cache(maxsize=1)
def load_corpus_metadata() -> pd.DataFrame:
    meta_path = path("corpus_metadata_csv")
    if meta_path.exists():
        return pd.read_csv(meta_path)
    return pd.DataFrame(columns=["doc_id", "title", "file_name", "expertise_area"])


def load_claims_csv(claims_path) -> pd.DataFrame:
    df = pd.read_csv(claims_path)
    if "support_status" not in df.columns:
        df = pd.read_csv(claims_path, header=None, names=CLAIMS_COLUMNS)
    return df


def ensure_claims_csv_header(claims_path) -> None:
    if not claims_path.exists():
        return
    text = claims_path.read_text(encoding="utf-8").strip()
    if not text:
        return
    first = text.splitlines()[0]
    if "support_status" in first:
        return
    df = load_claims_csv(claims_path)
    df.to_csv(claims_path, index=False)


def latest_by_case(records: list[dict], key: str = "case_id") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        cid = r.get(key)
        if cid:
            out[cid] = r
    return out
