"""Path validation and config helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_INPUTS = (
    "case_profiles",
    "xai_cards_jsonl",
)


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    resolved = {}
    for k, v in cfg.get("paths", {}).items():
        resolved[k] = str(_resolve_path(v))
    cfg["paths"] = resolved
    kb = cfg.get("paths", {}).get("knowledge_db_root")
    if kb:
        cfg["_knowledge_db_root"] = kb
    return cfg


def path(key: str) -> Path:
    return Path(load_config()["paths"][key])


def cfg_get(*keys, default=None):
    cur = load_config()
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_dirs() -> None:
    for k in ("results_dir", "figures_dir", "paper_outputs", "faiss_index", "inputs_dir", "benchmarks_dir"):
        path(k).mkdir(parents=True, exist_ok=True)
    (path("results_dir") / "formatted_cases").mkdir(parents=True, exist_ok=True)
    (path("results_dir") / "query_plans").mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    return ROOT


def validate_inputs() -> list[str]:
    missing = []
    for k in REQUIRED_INPUTS:
        p = path("inputs_dir") / {
            "case_profiles": "case_profiles.json",
            "xai_cards_jsonl": "xai_cards.jsonl",
        }[k]
        if not p.exists():
            missing.append(f"{k}: {p}")
    chunks = path("processed_chunks_csv")
    if not chunks.exists():
        missing.append(f"processed_chunks_csv: {chunks}")
    return missing
