"""OpenAI LLM client with structured JSON support."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get, load_dotenv  # noqa: E402


def _client():
    load_dotenv()
    from openai import OpenAI

    return OpenAI()


def chat(system: str, user: str, model: str | None = None, temperature: float | None = None) -> str:
    from generation.mock_llm import openai_available

    if not openai_available():
        return (
            "The model indicates elevated screening risk based on survey responses. "
            "This is a model-based screening signal and not a clinical diagnosis. "
            "Consider speaking with a qualified counselor or student support service."
        )
    model = model or cfg_get("llm", "generation_model", default="gpt-4o")
    temperature = temperature if temperature is not None else cfg_get("llm", "temperature", default=0.2)
    resp = _client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def structured_json(
    system: str,
    user: str,
    schema: dict[str, Any],
    schema_name: str = "response",
    model: str | None = None,
) -> dict:
    from generation.mock_llm import mock_narrative, mock_synthesis, openai_available

    if not openai_available():
        if schema_name == "evidence_synthesis":
            import re

            m = re.search(r"case_id[=:\s]+(\w+)", user, re.I)
            cid = m.group(1) if m else "high_risk_case"
            return mock_synthesis(cid)
        if schema_name in ("narrative", "safety_report"):
            import re

            m = re.search(r"case_id[=:\s\"]+([\w_]+)", user, re.I)
            cid = m.group(1) if m else "high_risk_case"
            elevated = "elevated" in user.lower()
            if schema_name == "safety_report":
                return {
                    "case_id": cid,
                    "diagnostic_overclaiming": "pass",
                    "medical_advice": "pass",
                    "causality_misuse": "pass",
                    "unsupported_claims": "pass",
                    "missing_disclaimer": "pass",
                    "missing_referral": "pass",
                    "stigmatizing_language": "pass",
                    "privacy_leakage": "pass",
                    "overall_safety": "pass",
                    "revision_required": False,
                    "comments": [],
                }
            return mock_narrative(cid, elevated)
        if schema_name == "faithfulness":
            return {"support_status": "partially_supported", "explanation": "Mock mode — API unavailable."}
        return {}
    model = model or cfg_get("llm", "synthesis_model", default="gpt-4o")
    resp = _client().chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or "{}"
    return json.loads(text)


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n---\n{user}".encode()).hexdigest()[:16]
