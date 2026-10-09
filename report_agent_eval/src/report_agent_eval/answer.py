"""Grounded answer generation from frozen contextual-RAG evidence.

Every model (reference gpt-oss and each candidate) answers the same question
from the identical frozen contexts, using the identical prompt wrapper.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from report_agent_eval.config import cfg, path
from report_agent_eval.ollama_client import OllamaError, chat

ANSWER_SYSTEM = """You are the evidence-grounded answering component of a non-clinical \
psychological screening report system. You answer a single planning question for an \
expert audience (counselors, student-support staff, researchers) using ONLY the \
retrieved evidence provided.

Rules:
1. SCREENING, NOT DIAGNOSIS: never diagnose, never prescribe treatment or medication.
2. GROUNDED: base every factual claim on the provided context passages; do not invent \
facts or citations. If the context is insufficient, say so explicitly.
3. CONTRIBUTORS, NOT CAUSES: describe model/XAI factors as statistical contributors, \
never proven causes.
4. CONCISE PROSE: 3-6 sentences of direct, professional prose that answers the question. \
No JSON, no headings, no bullet lists unless essential.
5. SAFETY: for elevated risk, note referral to qualified support where relevant."""


def _safe(part: str) -> str:
    return part.replace(":", "_").replace("/", "_")


def build_user_prompt(question: str, contexts: list[str]) -> str:
    if contexts:
        blocks = "\n\n".join(
            f"[Context {i}]\n{text}" for i, text in enumerate(contexts, start=1)
        )
    else:
        blocks = "(No retrieved evidence available.)"
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{blocks}\n\n"
        "Write the grounded answer now."
    )


def answers_path(model: str, case_id: str) -> Path:
    return path("answers_dir") / _safe(model) / f"{_safe(case_id)}.json"


def load_answers(model: str, case_id: str) -> dict:
    fp = answers_path(model, case_id)
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    return {}


def save_answers(model: str, case_id: str, payload: dict) -> Path:
    fp = answers_path(model, case_id)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def generate_answer(model: str, question: str, contexts: list[str]) -> tuple[str, str | None]:
    """Return (answer_text, error). Empty answer on failure."""
    user = build_user_prompt(question, contexts)
    try:
        text = chat(
            model,
            ANSWER_SYSTEM,
            user,
            temperature=float(cfg("ollama", "temperature", default=0.1)),
        )
        return (text or "").strip(), None
    except OllamaError as exc:
        return "", str(exc)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
