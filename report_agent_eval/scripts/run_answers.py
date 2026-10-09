"""Phase 1b: generate gpt-oss reference + candidate answers on frozen contexts."""

import argparse

import _bootstrap  # noqa: F401

from report_agent_eval.answer import (
    generate_answer,
    load_answers,
    now_iso,
    save_answers,
)
from report_agent_eval.config import cfg
from report_agent_eval.plans import load_question_records, records_by_case
from report_agent_eval.retrieval import context_texts, load_frozen_contexts


def _models(arg_models: list[str] | None) -> list[str]:
    reference = str(cfg("reference_model", default="gpt-oss:20b"))
    candidates = list(cfg("candidates", default=[]))
    if arg_models:
        selected = arg_models
    else:
        selected = candidates
    # Reference is always generated (needed as RAGAS ground_truth), and first.
    ordered = [reference] + [m for m in selected if m != reference]
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate answers per model from frozen contexts")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of candidate models")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    ap.add_argument("--force", action="store_true", help="Regenerate even if cached")
    args = ap.parse_args()

    plan_model = str(cfg("plan_model", default="gemma4"))
    records = load_question_records(plan_model, limit=args.limit)
    by_case = records_by_case(records)
    models = _models(args.models)

    total = len(models) * len(records)
    done = 0
    skipped_missing_ctx = 0
    for model in models:
        for case_id, case_records in by_case.items():
            store = {} if args.force else load_answers(model, case_id)
            store.setdefault("model", model)
            store.setdefault("case_id", case_id)
            answers = store.setdefault("answers", {})
            changed = False
            for rec in case_records:
                done += 1
                if rec.qid in answers and answers[rec.qid].get("answer") and not args.force:
                    continue
                frozen = load_frozen_contexts(rec.case_id, rec.qid)
                if frozen is None:
                    skipped_missing_ctx += 1
                    print(
                        f"[{done}/{total}] {model} {case_id} {rec.qid} -> SKIP (no frozen context)",
                        flush=True,
                    )
                    continue
                contexts = context_texts(frozen)
                text, error = generate_answer(model, rec.question, contexts)
                answers[rec.qid] = {
                    "qid": rec.qid,
                    "question": rec.question,
                    "question_type": rec.question_type,
                    "stratum": rec.stratum,
                    "answer": text,
                    "error": error,
                    "n_contexts": len(contexts),
                    "timestamp_utc": now_iso(),
                }
                changed = True
                status = "ok" if text else f"FAIL({error or 'empty'})"
                print(f"[{done}/{total}] {model} {case_id} {rec.qid} -> {status}", flush=True)
            if changed or not args.force:
                save_answers(model, case_id, store)

    print(
        f"Answer generation complete: {done} model-questions "
        f"across {len(models)} models ({skipped_missing_ctx} skipped for missing context)."
    )


if __name__ == "__main__":
    main()
