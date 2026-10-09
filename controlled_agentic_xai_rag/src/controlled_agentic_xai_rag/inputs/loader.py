"""Read-only input loaders with anonymization guard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from controlled_agentic_xai_rag.config_loader import path

ALLOWED_PROFILE_KEYS = {"case_id", "feature_levels", "predicted_probability", "predicted_class", "raw_values"}
FORBIDDEN_KEYS = {"name", "email", "student_id", "ssn", "address", "phone"}


def _inputs_dir() -> Path:
    return path("inputs_dir")


def load_case_profiles() -> dict:
    p = _inputs_dir() / "case_profiles.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_xai_cards() -> list[dict]:
    p = _inputs_dir() / "xai_cards.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_xai_card(case_id: str) -> dict:
    cards = {c["case_id"]: c for c in load_xai_cards()}
    if case_id not in cards:
        raise KeyError(f"Unknown case_id: {case_id}")
    return cards[case_id]


def get_case_profile(case_id: str) -> dict:
    profiles = load_case_profiles()
    if case_id not in profiles:
        raise KeyError(f"Unknown case profile: {case_id}")
    prof = profiles[case_id]
    for key in FORBIDDEN_KEYS:
        if key in prof:
            raise ValueError(f"PII field detected in case profile: {key}")
    return prof


def load_corpus_chunks() -> pd.DataFrame:
    p = path("processed_chunks_csv")
    if not p.exists():
        raise FileNotFoundError(f"Corpus not built: {p}")
    return pd.read_csv(p)
