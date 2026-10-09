"""Load and flatten the frozen gemma4 query plans into per-question records."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from report_agent_eval.config import cfg, path


def _safe_model_dir(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


@dataclass(frozen=True)
class QuestionRecord:
    case_id: str
    stratum: str
    qid: str
    question: str
    retrieval_query: str
    question_type: str

    def as_dict(self) -> dict:
        return asdict(self)


def _plan_files(plan_model: str) -> list[Path]:
    plans_root = path("plans_dir") / _safe_model_dir(plan_model)
    if not plans_root.exists():
        raise FileNotFoundError(
            f"No plans directory for model '{plan_model}': {plans_root}"
        )
    return sorted(plans_root.glob("*.json"))


def _stratum_from_case(case_id: str) -> str:
    # case ids look like hms_high_01 / hms_low_20 / hms_borderline_09
    parts = case_id.split("_")
    return parts[1] if len(parts) >= 2 else ""


def load_question_records(
    plan_model: str | None = None,
    *,
    limit: int | None = None,
) -> list[QuestionRecord]:
    """Flatten every question in every cached plan into QuestionRecord rows."""
    plan_model = plan_model or str(cfg("plan_model", default="gemma4"))
    files = _plan_files(plan_model)
    if limit:
        files = files[:limit]

    records: list[QuestionRecord] = []
    for fp in files:
        payload = json.loads(fp.read_text(encoding="utf-8"))
        if not payload.get("parsed_ok", False):
            continue
        case_id = payload.get("case_id", fp.stem)
        stratum = payload.get("stratum") or _stratum_from_case(case_id)
        plan = payload.get("plan") or {}
        for q in plan.get("question_plan", []) or []:
            question = str(q.get("question", "")).strip()
            if not question:
                continue
            retrieval_query = str(
                q.get("retrieval_query") or question
            ).strip()
            records.append(
                QuestionRecord(
                    case_id=case_id,
                    stratum=stratum,
                    qid=str(q.get("question_id", f"Q{len(records) + 1}")),
                    question=question,
                    retrieval_query=retrieval_query,
                    question_type=str(q.get("question_type", "")),
                )
            )
    return records


def records_by_case(records: list[QuestionRecord]) -> dict[str, list[QuestionRecord]]:
    grouped: dict[str, list[QuestionRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.case_id, []).append(rec)
    return grouped
