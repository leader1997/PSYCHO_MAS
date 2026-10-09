"""Phase 2: score candidate answers against gpt-oss reference with RAGAS.

Answer-only metrics; retrieval is fixed so context metrics are excluded.
Resumable: already-scored (model, case, qid) rows are skipped.
"""

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from report_agent_eval.answer import load_answers
from report_agent_eval.config import cfg, path
from report_agent_eval.plans import load_question_records
from report_agent_eval.ragas_score import ANSWER_METRICS, backend_info, score_batch
from report_agent_eval.retrieval import context_texts, load_frozen_contexts

PER_QUESTION_CSV = "ragas_per_question.csv"


def _row_id(model: str, case_id: str, qid: str) -> str:
    return f"{model}::{case_id}::{qid}"


def build_dataset(models: list[str], reference_model: str, records) -> pd.DataFrame:
    ref_cache: dict[str, dict] = {}

    def ref_answers(case_id: str) -> dict:
        if case_id not in ref_cache:
            ref_cache[case_id] = load_answers(reference_model, case_id).get("answers", {})
        return ref_cache[case_id]

    rows: list[dict] = []
    for model in models:
        for rec in records:
            store = load_answers(model, rec.case_id).get("answers", {})
            cand = store.get(rec.qid, {})
            answer = (cand.get("answer") or "").strip()
            if not answer:
                continue
            ground_truth = (ref_answers(rec.case_id).get(rec.qid, {}).get("answer") or "").strip()
            if not ground_truth:
                continue
            frozen = load_frozen_contexts(rec.case_id, rec.qid)
            if frozen is None:
                continue
            contexts = context_texts(frozen)
            if not contexts:
                contexts = ["No retrieved context available."]
            rows.append(
                {
                    "row_id": _row_id(model, rec.case_id, rec.qid),
                    "model": model,
                    "case_id": rec.case_id,
                    "qid": rec.qid,
                    "stratum": rec.stratum,
                    "question_type": rec.question_type,
                    "question": rec.question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score answers with RAGAS answer metrics")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of candidate models")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Rescore even if cached")
    args = ap.parse_args()

    reference_model = str(cfg("reference_model", default="gpt-oss:20b"))
    candidates = args.models or list(cfg("candidates", default=[]))
    plan_model = str(cfg("plan_model", default="gemma4"))
    records = load_question_records(plan_model, limit=args.limit)

    dataset = build_dataset(candidates, reference_model, records)
    if dataset.empty:
        print("No scorable (candidate, reference) answer pairs found. Run Phase 1 first.")
        return

    out_dir = path("outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / PER_QUESTION_CSV

    existing = pd.DataFrame()
    scored_ids: set[str] = set()
    if out_csv.exists() and not args.force:
        existing = pd.read_csv(out_csv)
        if "row_id" in existing.columns:
            scored_ids = set(existing["row_id"].astype(str))

    pending = dataset[~dataset["row_id"].astype(str).isin(scored_ids)]
    if pending.empty:
        print(f"All {len(dataset)} rows already scored. Nothing to do.")
        return

    batch_size = args.batch_size or int(cfg("ragas", "batch_size", default=8))
    print(f"RAGAS {backend_info()} | scoring {len(pending)} rows (batch={batch_size})", flush=True)

    combined = existing.copy()
    n = len(pending)
    n_batches = (n + batch_size - 1) // batch_size
    for start in range(0, n, batch_size):
        batch = pending.iloc[start : start + batch_size]
        print(
            f"  batch {start // batch_size + 1}/{n_batches} ({len(batch)} rows)...",
            flush=True,
        )
        scored = score_batch(batch)
        # Persist after every batch so interrupted runs keep progress and resume.
        combined = pd.concat([combined, scored], ignore_index=True) if not combined.empty else scored
        combined.drop_duplicates(subset=["row_id"], keep="last", inplace=True)
        combined.to_csv(out_csv, index=False)
        ok_so_far = combined[ANSWER_METRICS].notna().any(axis=1).sum()
        print(f"    saved {len(combined)} rows ({ok_so_far} with metrics) -> {out_csv}", flush=True)

    ok = combined[ANSWER_METRICS].notna().any(axis=1).sum()
    print(f"RAGAS scoring complete: {len(combined)} rows ({ok} with metrics) -> {out_csv}")


if __name__ == "__main__":
    main()
