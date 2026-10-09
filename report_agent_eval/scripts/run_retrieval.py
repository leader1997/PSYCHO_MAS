"""Phase 1a: freeze contextual-RAG contexts for every gemma4 question."""

import argparse
from datetime import datetime, timezone

import _bootstrap  # noqa: F401

from report_agent_eval.config import cfg
from report_agent_eval.plans import load_question_records
from report_agent_eval.retrieval import (
    ContextualRetriever,
    load_frozen_contexts,
    save_frozen_contexts,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze contextual RAG contexts per question")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    ap.add_argument("--force", action="store_true", help="Re-retrieve even if cached")
    args = ap.parse_args()

    plan_model = str(cfg("plan_model", default="gemma4"))
    records = load_question_records(plan_model, limit=args.limit)
    print(f"Loaded {len(records)} questions from '{plan_model}' plans.", flush=True)

    retriever = ContextualRetriever()

    total = len(records)
    done = 0
    reused = 0
    for rec in records:
        done += 1
        if not args.force and load_frozen_contexts(rec.case_id, rec.qid) is not None:
            reused += 1
            continue
        hits = retriever.retrieve(rec.retrieval_query)
        payload = {
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
        }
        save_frozen_contexts(rec.case_id, rec.qid, payload)
        print(f"[{done}/{total}] {rec.case_id} {rec.qid} -> {len(hits)} chunks", flush=True)

    print(f"Retrieval complete: {total} questions ({reused} reused from cache).")


if __name__ == "__main__":
    main()
