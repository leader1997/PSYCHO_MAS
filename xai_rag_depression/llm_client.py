"""LLM client: Ollama (default) for generation; OpenAI GPT optional for judging."""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def _load_env_file(path: Path, *, override: bool = True) -> None:
    """Minimal .env loader (no external dependency)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def _bootstrap_env() -> None:
    env_paths = [ROOT / ".env", ROOT.parent / ".env", Path.cwd() / ".env"]
    try:
        from dotenv import load_dotenv  # python-dotenv

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path, override=True)
        return
    except ImportError:
        # Wrong `dotenv` package is sometimes installed instead of `python-dotenv`.
        pass
    for env_path in env_paths:
        if env_path.exists():
            _load_env_file(env_path, override=True)


_bootstrap_env()

# Backends: ollama (default) | openai
GENERATION_BACKEND = os.environ.get("XAI_LLM_BACKEND", "ollama").lower()
JUDGE_BACKEND = os.environ.get("XAI_JUDGE_BACKEND", "openai").lower()

OLLAMA_URL = os.environ.get("XAI_OLLAMA_URL", "http://localhost:11434")
# Default candidates are Ollama chat models only (no OpenAI / embed models)
OLLAMA_PREFERRED = [
    m.strip()
    for m in os.environ.get(
        "XAI_OLLAMA_PREFERRED",
        "llama3.1,llama3,gemma2,mistral,phi4,qwen2.5",
    ).split(",")
    if m.strip()
]
OLLAMA_SKIP_PREFIXES = ("nomic-embed", "embed")

OPENAI_GENERATION_MODEL = os.environ.get("XAI_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.environ.get("XAI_OPENAI_JUDGE_MODEL", OPENAI_GENERATION_MODEL)

_openai_client = None
_ollama_generation_model: str | None = None
_ollama_judge_model: str | None = None


def _openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _ollama_list_models() -> list[str] | None:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        print(f"Ollama not reachable: {e}")
        return None


def _is_chat_ollama_model(name: str) -> bool:
    low = name.lower()
    return not any(low.startswith(p) for p in OLLAMA_SKIP_PREFIXES)


def _matches_preferred_ollama(name: str) -> bool:
    if not _is_chat_ollama_model(name):
        return False
    return any(name.startswith(prefix) for prefix in OLLAMA_PREFERRED)


def _select_ollama_model(available: list[str], env_var: str) -> str:
    preferred = [m for m in available if _matches_preferred_ollama(m)]
    env_model = os.environ.get(env_var)
    if env_model:
        # Allow explicit tag, e.g. llama3 -> llama3:latest
        if env_model in available:
            return env_model
        for name in available:
            if name.startswith(env_model):
                return name
    if preferred:
        return preferred[0]
    raise RuntimeError(
        f"No preferred Ollama chat model found. Pull one of: {', '.join(OLLAMA_PREFERRED)}"
    )


def _ollama_model_for(role: str) -> str:
    global _ollama_generation_model, _ollama_judge_model
    available = _ollama_list_models()
    if not available:
        raise RuntimeError(
            f"Ollama is not running at {OLLAMA_URL}. Start with `ollama serve`."
        )
    if role == "judge":
        if _ollama_judge_model is None:
            _ollama_judge_model = _select_ollama_model(
                available, "XAI_OLLAMA_JUDGE_MODEL"
            )
        return _ollama_judge_model
    if _ollama_generation_model is None:
        _ollama_generation_model = _select_ollama_model(available, "XAI_OLLAMA_MODEL")
    return _ollama_generation_model


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def backend_name() -> str:
    return GENERATION_BACKEND


def judge_backend_name() -> str:
    if JUDGE_BACKEND == "openai" and _openai_available():
        return "openai"
    return "ollama"


def is_available() -> bool:
    if GENERATION_BACKEND == "openai":
        return _openai_available()
    return _ollama_list_models() is not None and len(_ollama_list_models() or []) > 0


def is_judge_available() -> bool:
    if judge_backend_name() == "openai":
        return _openai_available()
    return is_available()


def model_name() -> str:
    if GENERATION_BACKEND == "openai":
        return OPENAI_GENERATION_MODEL
    try:
        return _ollama_model_for("generation")
    except RuntimeError:
        return "ollama-unavailable"


def judge_model_name() -> str:
    if judge_backend_name() == "openai":
        return OPENAI_JUDGE_MODEL
    try:
        return _ollama_model_for("judge")
    except RuntimeError:
        return "ollama-unavailable"


def _ollama_chat(model: str, system: str, user: str, temperature: float) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        timeout=600,
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def _openai_chat(model: str, system: str, user: str, temperature: float) -> str:
    from openai import AuthenticationError

    client = _get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except AuthenticationError as e:
        raise RuntimeError(
            "OpenAI authentication failed. Check OPENAI_API_KEY in .env."
        ) from e
    return (response.choices[0].message.content or "").strip()


def _complete(backend: str, model: str, system: str, user: str, temperature: float) -> str:
    if backend == "openai":
        return _openai_chat(model, system, user, temperature)
    return _ollama_chat(model, system, user, temperature)


def chat(system: str, user: str, temperature: float = 0.2) -> str:
    if GENERATION_BACKEND == "openai":
        return _complete("openai", OPENAI_GENERATION_MODEL, system, user, temperature)
    return _complete("ollama", _ollama_model_for("generation"), system, user, temperature)


def parse_score(text: str) -> float:
    m = re.search(r"([01](?:\.\d+)?)", text)
    return float(m.group(1)) if m else float("nan")


def judge(instruction: str, payload: str) -> float:
    """LLM-as-judge (default: GPT via XAI_JUDGE_BACKEND=openai)."""
    backend = judge_backend_name()
    model = judge_model_name()
    try:
        out = _complete(
            backend,
            model,
            "You are a strict evaluator. Respond with ONLY a number between 0 and 1.",
            f"{instruction}\n\n{payload}",
            temperature=0.0,
        )
        return parse_score(out)
    except Exception as e:
        print(f"  judge error: {e}")
        return float("nan")


def list_ollama_models() -> list[str]:
    models = _ollama_list_models() or []
    return [m for m in models if _matches_preferred_ollama(m)]
