"""Load config.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    raw = load_config()["paths"][key]
    p = Path(raw)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def cfg(*keys, default=None):
    cur = load_config()
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
