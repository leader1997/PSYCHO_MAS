"""Optional OpenAI moderation API check."""

from __future__ import annotations

from generation.config_loader import load_dotenv
from generation.llm import _client


def moderation_flagged(text: str) -> tuple[bool, str]:
    load_dotenv()
    try:
        resp = _client().moderations.create(input=text[:8000])
        result = resp.results[0]
        if result.flagged:
            cats = [k for k, v in result.categories.model_dump().items() if v]
            return True, f"Moderation flagged: {', '.join(cats)}"
        return False, ""
    except Exception as e:
        return False, f"Moderation skipped: {e}"
