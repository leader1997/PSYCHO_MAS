"""Drive full Phase 1 from notebook logic (resume-safe)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAS = ROOT.parent
for p in (ROOT / "src", MAS / "knowledge_db_contextual_eval" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from report_agent_eval.answer import generate_answer, load_answers, now_iso, save_answers
from report_agent_eval.config import cfg
from report_agent_eval.plans import load_question_records, records_by_case
from report_agent_eval.retrieval import (
    ContextualRetriever,
    context_texts,
    load_frozen_contexts,
    save_frozen_contexts,
)

SELECTED_CASE_IDS = [
    "hms_low_01",
    "hms_borderline_01",
    "hms_high_01",
]
FORCE_RETRIEVE = False
FORCE_ANSWERS = False
CANDIDATES = list(cfg("candidates", default=[]))
REFERENCE_MODEL = str(cfg("reference_model", default="gpt-oss:20b"))
PLAN_MODEL = str(cfg("plan_model", default="gemma4"))


def _selected_records():
    all_records = load_question_records(PLAN_MODEL, limit=None)
    order = {cid: i for i, cid in enumerate(SELECTED_CASE_IDS)}
    records = [r for r in all_records if r.case_id in SELECTED_CASE_IDS]
    return sorted(records, key=lambda r: (order.get(r.case_id, 999), r.qid))


def phase_retrieval() -> None:
    records = _selected_records()
    print(f"Questions: {len(records)} cases={SELECTED_CASE_IDS}", flush=True)
    retriever = ContextualRetriever()
    reused = fetched = 0
    for i, rec in enumerate(records, start=1):
        if not FORCE_RETRIEVE and load_frozen_contexts(rec.case_id, rec.qid) is not None:
            reused += 1
            continue
        hits = retriever.retrieve(rec.retrieval_query)
        save_frozen_contexts(
            rec.case_id,
            rec.qid,
            {
                "case_id": rec.case_id,
                "stratum": rec.stratum,
                "qid": rec.qid,
                "question": rec.question,
                "retrieval_query": rec.retrieval_query,
                "question_type": rec.question_type,
                "condition": retriever.condition,
                "mode": retriever.mode,
                "top_k": retriever.top_k,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "hits": hits,
            },
        )
        fetched += 1
        if fetched % 10 == 0 or i == len(records):
            print(f"[{i}/{len(records)}] fetched={fetched} reused={reused}", flush=True)
    print(f"Retrieval done. fetched={fetched} reused={reused}", flush=True)


def phase_answers() -> None:
    records = _selected_records()
    by_case = records_by_case(records)
    models = [REFERENCE_MODEL] + [m for m in CANDIDATES if m != REFERENCE_MODEL]
    total_jobs = len(models) * len(records)
    done = skipped = 0
    print(f"Cases={SELECTED_CASE_IDS} models={models} questions={len(records)}", flush=True)
    for model in models:
        for case_id, case_recs in by_case.items():
            store = {} if FORCE_ANSWERS else load_answers(model, case_id)
            store.setdefault("model", model)
            store.setdefault("case_id", case_id)
            answers = store.setdefault("answers", {})
            changed = False
            for rec in case_recs:
                done += 1
                if (
                    not FORCE_ANSWERS
                    and rec.qid in answers
                    and (answers[rec.qid].get("answer") or "").strip()
                ):
                    continue
                frozen = load_frozen_contexts(rec.case_id, rec.qid)
                if frozen is None:
                    skipped += 1
                    continue
                text, error = generate_answer(model, rec.question, context_texts(frozen))
                answers[rec.qid] = {
                    "qid": rec.qid,
                    "question": rec.question,
                    "question_type": rec.question_type,
                    "stratum": rec.stratum,
                    "answer": text,
                    "error": error,
                    "n_contexts": len(context_texts(frozen)),
                    "timestamp_utc": now_iso(),
                }
                changed = True
                status = "ok" if text else f"FAIL({error or 'empty'})"
                print(f"[{done}/{total_jobs}] {model} {case_id} {rec.qid} -> {status}", flush=True)
            if changed:
                save_answers(model, case_id, store)
    print(f"Answers done. skipped_ctx={skipped}", flush=True)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("retrieval", "all"):
        phase_retrieval()
    if phase in ("answers", "all"):
        phase_answers()
