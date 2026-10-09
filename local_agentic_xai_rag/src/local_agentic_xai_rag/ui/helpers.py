"""Shared helpers for the Streamlit UI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from local_agentic_xai_rag.config_loader import (
    cfg_get,
    load_config,
    path,
    project_root,
    validate_phase2_paths,
)
from local_agentic_xai_rag.llm.ollama_client import MODEL, ping


def load_jsonl(name: str) -> list[dict]:
    p = path("results_dir") / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_results_csv(name: str) -> pd.DataFrame | None:
    p = path("results_dir") / name
    return pd.read_csv(p) if p.exists() else None


def load_gates() -> dict:
    p = path("results_dir") / "acceptance_gates.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    from local_agentic_xai_rag.evaluation.acceptance import run_gates

    return run_gates()


def corpus_stats() -> dict:
    chunks_path = path("processed_chunks_csv")
    if not chunks_path.exists():
        return {"chunks": 0, "documents": 0}
    df = pd.read_csv(chunks_path, usecols=lambda c: c in ("chunk_id", "doc_id"))
    return {
        "chunks": len(df),
        "documents": df["doc_id"].nunique() if "doc_id" in df.columns else 0,
    }


def figure_paths() -> list[Path]:
    fig_dir = path("figures_dir")
    if not fig_dir.exists():
        return []
    return sorted(fig_dir.glob("fig*.png"))


def config_summary() -> dict:
    cfg = load_config()
    return {
        "model": cfg_get("ollama", "model"),
        "retrieval": cfg_get("retrieval", "default"),
        "top_k": cfg_get("retrieval", "top_k"),
        "phase2_root": cfg.get("_phase2_root"),
        "project_root": str(project_root()),
    }
