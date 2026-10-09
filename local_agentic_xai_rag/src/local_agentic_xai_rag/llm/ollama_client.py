"""Ollama client — llama3 only for chat and embeddings."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from local_agentic_xai_rag.config_loader import cfg_get, load_config

load_config()

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL") or cfg_get("ollama", "base_url", default="http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL") or cfg_get("ollama", "model", default="llama3")


def ping() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def ensure_llama3() -> None:
    if not ping():
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_URL}")
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
    names = [m["name"].split(":")[0] for m in tags]
    if MODEL.split(":")[0] not in names and f"{MODEL}:latest" not in [m["name"] for m in tags]:
        raise RuntimeError(f"Model {MODEL} not found in Ollama. Run: ollama pull llama3")


def chat(prompt: str, system: str = "", temperature: float = 0.1) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def embed(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    if not texts:
        return []
    batch_size = batch_size or cfg_get("retrieval", "embed_batch_size", default=32)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": MODEL, "input": batch},
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        embs = data.get("embeddings") or [data["embedding"]]
        out.extend(embs)
    return out


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    # trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def structured_json(system: str, user: str, temperature: float = 0.1) -> dict[str, Any]:
    cur = chat(
        user + "\n\nRespond with valid JSON only, no markdown.",
        system=system,
        temperature=temperature,
    )
    for attempt in range(3):
        try:
            return _extract_json(cur)
        except json.JSONDecodeError:
            if attempt >= 2:
                break
            cur = chat(
                f"Fix this to valid JSON only:\n{cur[:4000]}",
                system="Output valid JSON only.",
                temperature=0.0,
            )
    return {}
