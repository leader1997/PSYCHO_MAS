"""Ollama client — sequential calls with cooldown; per-run model override."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from typing import Any, Iterator

import requests

from controlled_agentic_xai_rag.config_loader import cfg_get, load_config

load_config()

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL") or cfg_get("ollama", "base_url", default="http://localhost:11434")
TIMEOUT = int(cfg_get("ollama", "request_timeout_seconds", default=600))

_active_model: str | None = None
_active_embed_model: str | None = None
_embed_api_available: bool | None = None
_embed_probe_model: str | None = None


class EmbeddingUnavailableError(RuntimeError):
    """Raised when Ollama cannot produce embeddings for the configured embed model."""


def format_for_embedding(text: str, *, model: str | None = None, role: str = "document") -> str:
    """Apply model-specific prefixes (nomic-embed-text expects search_query / search_document)."""
    model_name = (model or get_embed_model()).lower()
    if "nomic-embed" not in model_name:
        return text
    if text.startswith("search_query:") or text.startswith("search_document:"):
        return text
    prefix = "search_query: " if role == "query" else "search_document: "
    return prefix + text


def get_model() -> str:
    if _active_model:
        return _active_model
    return os.environ.get("OLLAMA_MODEL") or cfg_get("ollama", "model", default="llama3")


def get_embed_model() -> str:
    """Embeddings use embed_model (default nomic-embed-text), not the chat model."""
    if _active_embed_model:
        return _active_embed_model
    return (
        os.environ.get("OLLAMA_EMBED_MODEL")
        or cfg_get("ollama", "embed_model", default="nomic-embed-text")
    )


@contextmanager
def use_embed_model(model: str) -> Iterator[None]:
    """Temporarily override the embedding model for one pipeline run."""
    global _active_embed_model, _embed_api_available, _embed_probe_model
    prev = _active_embed_model
    prev_probe = _embed_probe_model
    _active_embed_model = model
    _embed_api_available = None
    _embed_probe_model = None
    try:
        yield
    finally:
        _active_embed_model = prev
        _embed_api_available = None
        _embed_probe_model = None if prev is None else prev_probe


def list_embedding_models() -> list[str]:
    """Installed Ollama models that support /api/embed (excludes chat-only models)."""
    preferred = cfg_get(
        "ollama",
        "embedding_models",
        default=["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed"],
    ) or ["nomic-embed-text"]
    if not ping():
        return list(preferred)
    installed = list_installed_models()
    found: list[str] = []
    for name in installed:
        base = name.split(":")[0]
        if "embed" in base.lower():
            found.append(name)
    for p in preferred:
        bases = {m.split(":")[0] for m in found}
        if p not in bases and p not in found:
            if any(m.split(":")[0] == p for m in installed):
                for m in installed:
                    if m.split(":")[0] == p:
                        found.append(m)
                        break
    current = resolve_embed_model(get_embed_model())
    if current and current not in found:
        found.insert(0, current)
    return found or list(preferred)


@contextmanager
def use_model(model: str) -> Iterator[None]:
    """Temporarily override the chat/verification model for one pipeline run."""
    global _active_model
    prev = _active_model
    _active_model = model
    try:
        yield
    finally:
        _active_model = prev


def _after_call() -> None:
    delay = float(cfg_get("ollama", "inter_call_delay_seconds", default=1))
    if delay > 0:
        time.sleep(delay)


def ping() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def list_installed_models() -> list[str]:
    """Return Ollama model names from /api/tags, or config fallbacks."""
    fallbacks = cfg_get("ollama", "fallback_models", default=["llama3"]) or ["llama3"]
    if not ping():
        return list(fallbacks)
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
        names = [m["name"] for m in tags if m.get("name")]
        return names if names else list(fallbacks)
    except Exception:
        return list(fallbacks)


def ensure_model(model: str | None = None) -> None:
    model = model or get_model()
    if not ping():
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_URL}")
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
    full_names = [m["name"] for m in tags]
    base_names = [m["name"].split(":")[0] for m in tags]
    if model in full_names or model.split(":")[0] in base_names:
        return
    raise RuntimeError(f"Model {model} not found in Ollama. Run: ollama pull {model.split(':')[0]}")


def ensure_llama3() -> None:
    ensure_model(get_model())


def chat(prompt: str, system: str = "", temperature: float = 0.1, model: str | None = None) -> str:
    model = model or get_model()
    stream = bool(cfg_get("ollama", "stream", default=False))
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": stream,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    content = r.json()["message"]["content"].strip()
    _after_call()
    return content


def resolve_embed_model(model: str | None = None) -> str:
    """Map config name (e.g. nomic-embed-text) to installed Ollama tag if present."""
    model = model or get_embed_model()
    if not ping():
        return model
    try:
        installed = list_installed_models()
    except Exception:
        return model
    if model in installed:
        return model
    base = model.split(":")[0]
    for name in installed:
        if name.split(":")[0] == base:
            return name
    return model


def _embed_error_detail(response: requests.Response | None) -> str:
    if response is None:
        return ""
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])[:500]
    except Exception:
        pass
    return (response.text or "")[:500]


def _embed_legacy(text: str, model: str) -> list[float]:
    model = resolve_embed_model(model)
    prompt = text if text and str(text).strip() else " "
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": prompt},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not emb:
        raise EmbeddingUnavailableError(
            f"Ollama /api/embeddings returned no vector for model {model!r}"
        )
    _after_call()
    return emb


def _embed_batch_modern(batch: list[str], model: str) -> list[list[float]]:
    """Call /api/embed; split batch on 400 (Ollama tokenize overload / payload limits)."""
    model = resolve_embed_model(model)
    texts = [t if t and str(t).strip() else " " for t in batch]
    if not texts:
        return []

    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts if len(texts) > 1 else texts[0]},
        timeout=TIMEOUT,
    )
    if r.status_code == 400 and len(texts) > 1:
        mid = max(1, len(texts) // 2)
        return _embed_batch_modern(texts[:mid], model) + _embed_batch_modern(texts[mid:], model)
    if r.status_code == 400 and len(texts) == 1:
        return [_embed_legacy(texts[0], model)]

    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        detail = _embed_error_detail(r)
        raise EmbeddingUnavailableError(
            f"Ollama /api/embed failed ({r.status_code}) for model {model!r}: {detail}"
        ) from exc

    data = r.json()
    embs = data.get("embeddings")
    if embs is None and "embedding" in data:
        embs = [data["embedding"]]
    if not embs:
        raise EmbeddingUnavailableError(
            f"Ollama /api/embed returned no vectors for model {model!r}"
        )
    _after_call()
    return embs


def embeddings_available(model: str | None = None, force_check: bool = False) -> bool:
    """Probe whether Ollama can embed with the configured model."""
    global _embed_api_available, _embed_probe_model
    model = model or get_embed_model()
    if (
        not force_check
        and _embed_api_available is not None
        and _embed_probe_model == model
    ):
        return _embed_api_available
    if not ping():
        _embed_api_available = False
        _embed_probe_model = model
        return False
    model = resolve_embed_model(model)
    try:
        embed([format_for_embedding("embedding probe", model=model, role="query")], model=model)
    except (EmbeddingUnavailableError, requests.RequestException):
        _embed_api_available = False
        _embed_probe_model = model
        return False
    _embed_api_available = True
    _embed_probe_model = model
    return True


def embed(texts: list[str], batch_size: int | None = None, model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    model = resolve_embed_model(model)
    raw_batch = int(batch_size or cfg_get("retrieval", "embed_batch_size", default=16))
    batch_size = max(1, min(raw_batch, 16))
    out: list[list[float]] = []
    modern_ok: bool | None = None
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if modern_ok is not False:
            try:
                out.extend(_embed_batch_modern(batch, model))
                modern_ok = True
                continue
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in (404, 501):
                    modern_ok = False
                else:
                    detail = _embed_error_detail(exc.response)
                    raise EmbeddingUnavailableError(
                        f"Ollama embed API failed ({status}): {detail or exc}"
                    ) from exc
            except EmbeddingUnavailableError as exc:
                msg = str(exc).lower()
                if "does not support embeddings" in msg or "not found" in msg:
                    modern_ok = False
                else:
                    raise
            except requests.RequestException as exc:
                raise EmbeddingUnavailableError(f"Ollama embed API unreachable: {exc}") from exc
        try:
            for text in batch:
                out.append(_embed_legacy(text, model))
        except requests.RequestException as exc:
            raise EmbeddingUnavailableError(
                f"Ollama embeddings unavailable for model {model!r}. "
                f"Use an embedding model (ollama pull nomic-embed-text), not a chat model like llama3. "
                f"Details: {exc}"
            ) from exc
    global _embed_api_available
    _embed_api_available = True
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
