"""Path validation and config helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PATHS = (
    "processed_chunks_csv",
    "xai_cards_jsonl",
    "case_profiles",
    "benchmark_gold_csv",
)


def _resolve_path(raw: str, phase2: Path) -> Path:
    expanded = raw.replace("${phase2_root}", str(phase2))
    p = Path(expanded)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    phase2 = (ROOT / cfg.get("phase2_root", "../explainability_augmented_rag")).resolve()
    cfg["_phase2_root"] = str(phase2)
    resolved = {}
    for k, v in cfg.get("paths", {}).items():
        resolved[k] = str(_resolve_path(v, phase2))
    cfg["paths"] = resolved
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
    for k in ("results_dir", "figures_dir", "paper_outputs", "faiss_index"):
        path(k).mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    return ROOT


def validate_phase2_paths() -> list[str]:
    missing = []
    for k in REQUIRED_PATHS:
        p = path(k)
        if not p.exists():
            missing.append(f"{k}: {p}")
    return missing
