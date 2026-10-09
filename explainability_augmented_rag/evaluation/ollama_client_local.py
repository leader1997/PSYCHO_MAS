"""Ollama-only client for benchmark evaluation (no cloud APIs)."""

from __future__ import annotations

import os
import re
from typing import Any

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("OLLAMA_GEN_MODEL", "llama3")
JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "llama3")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def ping() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def chat(prompt: str, system: str = "", model: str | None = None, temperature: float = 0.2) -> str:
    model = model or GEN_MODEL
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
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


def embed(texts: list[str], model: str | None = None, batch_size: int = 64) -> list[list[float]]:
    model = model or EMBED_MODEL
    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": model, "input": batch},
            timeout=600,
        )
        if r.status_code == 404:
            # Fallback for older Ollama builds without /api/embed batching
            for text in batch:
                rr = requests.post(
                    f"{OLLAMA_URL}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=120,
                )
                rr.raise_for_status()
                out.append(rr.json()["embedding"])
            continue
        r.raise_for_status()
        data = r.json()
        embs = data.get("embeddings")
        if embs is None and "embedding" in data:
            embs = [data["embedding"]]
        out.extend(embs)
    return out


def parse_score(text: str) -> float:
    m = re.search(r"([01](?:\.\d+)?)", text)
    return float(m.group(1)) if m else float("nan")


def judge(instruction: str, payload: str, model: str | None = None) -> float:
    model = model or JUDGE_MODEL
    out = chat(
        f"{instruction}\n\n{payload}\n\nRespond with ONLY a number between 0 and 1.",
        system="Strict evaluator. Output only a score between 0 and 1.",
        model=model,
        temperature=0.0,
    )
    return parse_score(out)
