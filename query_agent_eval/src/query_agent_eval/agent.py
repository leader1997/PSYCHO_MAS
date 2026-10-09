"""Run the Query Planning Agent on one case with one model; cache outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from query_agent_eval.config import cfg, path
from query_agent_eval.ollama_client import OllamaError, chat_json
from query_agent_eval.prompt import build_messages


def _safe_model_dir(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def plan_path(model: str, case_id: str) -> Path:
    return path("plans_dir") / _safe_model_dir(model) / f"{case_id}.json"


def run_one(model: str, record: dict, *, force: bool = False) -> dict:
    """Generate a question plan for one (model, case). Cached to disk."""
    case_id = record["case_id"]
    out_path = plan_path(model, case_id)

    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    structured_case = record["structured_case"]
    system, user = build_messages(structured_case)

    parsed: dict = {}
    raw = ""
    parsed_ok = False
    error = None
    try:
        parsed, raw, parsed_ok = chat_json(
            model,
            system,
            user,
            temperature=float(cfg("ollama", "temperature", default=0.1)),
        )
    except OllamaError as exc:
        error = str(exc)

    result = {
        "case_id": case_id,
        "model": model,
        "stratum": record.get("stratum"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parsed_ok": parsed_ok,
        "error": error,
        "plan": parsed,
        "raw": raw,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
