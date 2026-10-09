"""Report agent evaluation stub."""

from __future__ import annotations

import json

from controlled_agentic_xai_rag.config_loader import path
from controlled_agentic_xai_rag.validators.safety_rules import check_safety_rules
from controlled_agentic_xai_rag.validators.xai_alignment import check_xai_alignment
from controlled_agentic_xai_rag.inputs.loader import get_xai_card


def eval_report(case_id: str) -> dict:
    finals = path("results_dir") / "final_narratives.jsonl"
    if not finals.exists():
        return {"case_id": case_id, "found": False}
    for line in finals.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("case_id") != case_id:
            continue
        card = get_xai_card(case_id)
        elevated = "elevated" in r.get("screening_summary", "").lower() or card.get("prediction_probability", 0) >= 0.5
        safety = check_safety_rules(r, elevated=elevated)
        xai = check_xai_alignment(r, card)
        return {
            "case_id": case_id,
            "found": True,
            "safety_pass": safety["passed"],
            "xai_alignment_pass": xai["passed"],
            "n_citations": len(r.get("citations", [])),
        }
    return {"case_id": case_id, "found": False}
