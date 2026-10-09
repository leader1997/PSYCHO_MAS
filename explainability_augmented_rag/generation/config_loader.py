"""Shared configuration and paths for Phase 2 RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def path(key: str) -> Path:
    cfg = load_config()
    rel = cfg["paths"][key]
    return ROOT / rel


def cfg_get(*keys: str, default: Any = None) -> Any:
    cfg = load_config()
    for k in keys:
        if isinstance(cfg, dict) and k in cfg:
            cfg = cfg[k]
        else:
            return default
    return cfg


def ensure_dirs() -> None:
    for key in (
        "raw_pdfs",
        "project_artifacts",
        "results_dir",
        "paper_outputs",
        "faiss_index",
    ):
        p = path(key)
        p.mkdir(parents=True, exist_ok=True)
    (ROOT / "xai").mkdir(exist_ok=True)
    (ROOT / "notebooks").mkdir(exist_ok=True)
    (path("paper_outputs") / "figures_for_paper").mkdir(parents=True, exist_ok=True)


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv as _load

            _load(env_file, override=True)
        except ImportError:
            pass
