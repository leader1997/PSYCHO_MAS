"""Self-contained Ollama chat client with per-call model + robust JSON extraction."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from query_agent_eval.config import cfg


class OllamaError(RuntimeError):
    pass


def _base_url() -> str:
    return str(cfg("ollama", "base_url", default="http://localhost:11434")).rstrip("/")


def chat(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    timeout: int | None = None,
) -> str:
    """Single /api/chat call. Returns raw assistant text."""
    delay = float(cfg("ollama", "inter_call_delay_seconds", default=0.25))
    if delay:
        time.sleep(delay)
    temp = temperature if temperature is not None else float(cfg("ollama", "temperature", default=0.1))
    timeout = timeout or int(cfg("ollama", "request_timeout_seconds", default=900))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temp},
    }
    try:
        r = requests.post(f"{_base_url()}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(f"Ollama chat failed for model={model}: {exc}") from exc
    data = r.json()
    return data.get("message", {}).get("content", "") or data.get("response", "")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first valid top-level JSON object from model output."""
    if not text:
        raise json.JSONDecodeError("empty output", "", 0)
    # Strip common code fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates: list[str] = []
    if fenced:
        candidates.append(fenced.group(1))
    # Greedy first-brace to last-brace fallback.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])
    candidates.append(text.strip())

    last_error: Exception | None = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise last_error if last_error else json.JSONDecodeError("no json", text, 0)


def chat_json(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    timeout: int | None = None,
    retries: int = 2,
) -> tuple[dict[str, Any], str, bool]:
    """Chat and parse JSON.

    Returns (parsed_dict, raw_text, parsed_ok). On repeated parse failure,
    returns ({}, last_raw, False).
    """
    raw = ""
    for attempt in range(retries + 1):
        raw = chat(
            model,
            system,
            user + ("\n\nReturn valid JSON only. No markdown, no commentary." if attempt else ""),
            temperature=0.0 if attempt else temperature,
            timeout=timeout,
        )
        try:
            return _extract_json(raw), raw, True
        except json.JSONDecodeError:
            continue
    return {}, raw, False
