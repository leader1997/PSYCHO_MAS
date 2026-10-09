"""Audit logging for pipeline traceability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from controlled_agentic_xai_rag.config_loader import path


def _results() -> Path:
    return path("results_dir")


def append_jsonl(filename: str, record: dict[str, Any]) -> None:
    p = _results() / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(subpath: str, record: dict[str, Any]) -> Path:
    p = _results() / subpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
