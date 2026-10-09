"""Minimal Ollama chat client (per-call model). Mirrors query_agent_eval."""

from __future__ import annotations

import time

import requests

from report_agent_eval.config import cfg


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
    delay = float(cfg("ollama", "inter_call_delay_seconds", default=0.5))
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
    return (data.get("message", {}) or {}).get("content", "") or data.get("response", "")
