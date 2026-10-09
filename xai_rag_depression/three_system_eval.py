"""Three-system ablation: profile + prediction + (local XAI | agg XAI | none) + RAG."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import llm_client
from case_profiles import load_case_profiles, prediction_block, user_profile_block

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
KB_DIR = OUTPUT_DIR / "kb"
FIG_DIR = OUTPUT_DIR / "three_system_eval" / "figures"

THREE_SYSTEMS = [
    ("local_xai", "Sys-1: Profile + prediction + local XAI + RAG"),
    ("agg_xai", "Sys-2: Profile + prediction + aggregated XAI + RAG"),
    ("profile_rag", "Sys-3: Profile + prediction + RAG (no XAI)"),
]

CASE_QUESTION = (
    "Explain this student's depression-risk screening result and the main contributing factors."
)

PROFILE_RAG_PROMPT = (
    "You are a careful research assistant for college student depression/anxiety SCREENING. "
    "Use ONLY the provided context (user profile, prediction, and retrieved literature). "
    "Do NOT provide a clinical diagnosis or treatment plan. "
    "Use non-clinical language: associated with, risk marker, screening signal. "
    "Avoid causal claims. Mention uncertainty. "
    "Do NOT invent SHAP values or feature-importance rankings. "
    "Structure: Prediction, Profile interpretation, Uncertainty, Recommendation."
)

LOCAL_XAI_PROMPT = PROFILE_RAG_PROMPT + (
    " Use the LOCAL XAI block to explain which factors contributed to this prediction. "
    "Distinguish model contribution from literature evidence."
)

AGG_XAI_PROMPT = PROFILE_RAG_PROMPT + (
    " Use the AGGREGATED XAI block to explain robust cross-model feature groups. "
    "Mention agreement levels and uncertainty. Distinguish model evidence from literature."
)

GROUNDED = (
    "Rate how well the ANSWER is grounded in (entailed by) the CONTEXT "
    "(1=fully grounded, 0=hallucinated)."
)
RELEVANT = (
    "Rate how relevant the ANSWER is to the QUESTION (1=fully relevant, 0=irrelevant)."
)
CTX_PREC = (
    "Rate the fraction of the RETRIEVED context that is relevant to the QUESTION "
    "(1=all relevant)."
)
CTX_REC = (
    "Rate the fraction of key facts in the ANSWER that are supported by the RETRIEVED context "
    "(1=all supported)."
)
HALLUC = (
    "Rate the fraction of statements in the ANSWER that are NOT supported by the CONTEXT "
    "(1=all unsupported, 0=none)."
)

RESULTS_CSV = OUTPUT_DIR / "three_system_eval_results.csv"
ANSWERS_CSV = OUTPUT_DIR / "three_system_eval_answers.csv"
ANSWERS_WIDE_CSV = OUTPUT_DIR / "three_system_eval_answers_wide.csv"
SUMMARY_CSV = OUTPUT_DIR / "three_system_eval_summary.csv"


def _load_kb():
    emb = np.load(KB_DIR / "embeddings.npy")
    with open(KB_DIR / "documents.json", encoding="utf-8") as f:
        docs = json.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_by_id = {d["id"]: d for d in docs}

    def retrieve(query: str, k: int = 6) -> list[dict]:
        q = embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        sims = emb @ q
        return [docs[i] for i in np.argsort(sims)[::-1][:k]]

    ethics = doc_by_id.get("ethics_limitations", {}).get("text", "")
    return retrieve, ethics


def _load_xai():
    with open(OUTPUT_DIR / "local_explanations.json", encoding="utf-8") as f:
        cases = {c["case_id"]: c for c in json.load(f)}
    with open(OUTPUT_DIR / "cards" / "aggregated_explanation.json", encoding="utf-8") as f:
        agg_card = json.load(f)
    return cases, agg_card


def _local_xai_block(case: dict) -> str:
    factors = "\n".join(
        f"- {c['feature']} ({c['level']}): {c['direction']} (SHAP {c['shap_value']})"
        for c in case["top_contributing_factors"]
    )
    return f"LOCAL XAI (CatBoost SHAP for this case):\n{factors}"


def _aggregated_xai_block(agg_card: dict) -> str:
    groups = "\n".join(
        f"- {g['group']}: importance {g['importance']}, agreement {g['agreement']}"
        for g in agg_card["top_feature_groups"][:5]
    )
    return (
        f"AGGREGATED GLOBAL XAI (9 models):\n{groups}\n\n"
        f"INSIGHT:\n{agg_card['final_insight']}"
    )


def _literature_block(hits: list[dict], k: int = 6) -> str:
    blocks = []
    for d in hits:
        if d["type"] == "literature":
            blocks.append(f"[{d['title']}]\n{d['text'][:700]}")
    if not blocks:
        for d in hits:
            blocks.append(f"[{d['type']}] {d['title']}: {d['text'][:400]}")
    return "RETRIEVED LITERATURE:\n" + "\n\n".join(blocks[:k])


def _literature_query(case: dict) -> str:
    top_feats = " ".join(c["feature"] for c in case["top_contributing_factors"][:3])
    return (
        f"college student depression anxiety {case['predicted_class']} {top_feats}"
    )


def system_prompt(system_id: str) -> str:
    if system_id == "local_xai":
        return LOCAL_XAI_PROMPT
    if system_id == "agg_xai":
        return AGG_XAI_PROMPT
    return PROFILE_RAG_PROMPT


def build_context(
    system_id: str,
    profile: dict,
    case: dict,
    agg_card: dict,
    ethics: str,
    hits: list[dict],
) -> str:
    """All systems share user profile + prediction + ethics + literature."""
    parts = []
    if ethics:
        parts.append(f"ETHICS AND LIMITATIONS:\n{ethics}")
    parts.append(user_profile_block(profile))
    parts.append(prediction_block(profile))

    if system_id == "local_xai":
        parts.append(_local_xai_block(case))
    elif system_id == "agg_xai":
        parts.append(_aggregated_xai_block(agg_card))

    parts.append(_literature_block(hits))
    return "\n\n".join(parts)


def evaluate_answer(
    system_id: str,
    ctx: str,
    answer: str,
    retrieve,
    case: dict,
) -> dict:
    row = {
        "faithfulness": llm_client.judge(GROUNDED, f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}"),
        "answer_relevancy": llm_client.judge(
            RELEVANT, f"QUESTION: {CASE_QUESTION}\n\nANSWER:\n{answer}"
        ),
        "hallucination_rate": llm_client.judge(
            HALLUC, f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}"
        ),
    }
    retrieved = retrieve(_literature_query(case), k=8)
    ret_text = "\n".join(
        f"[{d['type']}] {d['title']}: {d['text'][:200]}" for d in retrieved
    )
    row["context_precision"] = llm_client.judge(
        CTX_PREC, f"QUESTION: {CASE_QUESTION}\n\nRETRIEVED:\n{ret_text}"
    )
    row["context_recall"] = llm_client.judge(
        CTX_REC, f"ANSWER:\n{answer}\n\nRETRIEVED:\n{ret_text}"
    )
    return row


def run_three_system_eval(
    skip_existing: bool = True,
    run_judge: bool | None = None,
) -> pd.DataFrame:
    """Generate answers and optional GPT judge scores for 3 cases × 3 systems."""
    os.environ.setdefault("XAI_LLM_BACKEND", "ollama")
    if not llm_client.is_available():
        raise RuntimeError("Ollama unavailable. Start `ollama serve` and pull a chat model.")

    if run_judge is None:
        run_judge = llm_client.is_judge_available()

    profiles = load_case_profiles()
    cases, agg_card = _load_xai()
    retrieve, ethics = _load_kb()

    done: set[tuple[str, str]] = set()
    existing_answers = pd.DataFrame()
    existing_results = pd.DataFrame()
    if skip_existing and ANSWERS_CSV.exists():
        existing_answers = pd.read_csv(ANSWERS_CSV)
        done = {
            (str(r.case_id), str(r.system))
            for r in existing_answers.itertuples()
        }
    if skip_existing and RESULTS_CSV.exists():
        existing_results = pd.read_csv(RESULTS_CSV)

    answer_rows = []
    result_rows = []

    case_ids = ["high_risk_case", "low_risk_case", "borderline_case"]
    for case_id in case_ids:
        profile = profiles[case_id]
        case = cases[case_id]
        hits = retrieve(_literature_query(case), k=6)

        for system_id, system_label in THREE_SYSTEMS:
            if (case_id, system_id) in done:
                continue
            print(f"  {case_id} | {system_id}...", flush=True)
            ctx = build_context(system_id, profile, case, agg_card, ethics, hits)
            answer = llm_client.chat(
                system_prompt(system_id),
                f"CONTEXT:\n{ctx}\n\nQUESTION: {CASE_QUESTION}",
            )
            answer_rows.append(
                {
                    "case_id": case_id,
                    "system": system_id,
                    "system_label": system_label,
                    "question": CASE_QUESTION,
                    "answer": answer,
                    "llm_backend": llm_client.backend_name(),
                    "llm_model": llm_client.model_name(),
                }
            )
            if run_judge:
                scores = evaluate_answer(system_id, ctx, answer, retrieve, case)
                result_rows.append(
                    {
                        "case_id": case_id,
                        "system": system_id,
                        "system_label": system_label,
                        "llm_backend": llm_client.backend_name(),
                        "llm_model": llm_client.model_name(),
                        "judge_backend": llm_client.judge_backend_name(),
                        "judge_model": llm_client.judge_model_name(),
                        **scores,
                    }
                )

    if answer_rows:
        answers = pd.concat([existing_answers, pd.DataFrame(answer_rows)], ignore_index=True)
        answers.drop_duplicates(subset=["case_id", "system"], keep="last", inplace=True)
        answers.to_csv(ANSWERS_CSV, index=False)
        _write_answers_wide(answers)
    else:
        answers = existing_answers

    if result_rows:
        results = pd.concat([existing_results, pd.DataFrame(result_rows)], ignore_index=True)
        results.drop_duplicates(subset=["case_id", "system"], keep="last", inplace=True)
        results.to_csv(RESULTS_CSV, index=False)
        _write_summary(results)
    else:
        results = existing_results
        if not results.empty:
            _write_summary(results)

    return results if not results.empty else answers


def _write_answers_wide(answers: pd.DataFrame) -> None:
    meta = answers.drop_duplicates("case_id")[["case_id", "question"]]
    wide = answers.pivot_table(
        index="case_id", columns="system", values="answer", aggfunc="first"
    )
    wide.columns = [f"answer_{c}" for c in wide.columns]
    wide = wide.reset_index()
    order = [f"answer_{sid}" for sid, _ in THREE_SYSTEMS if f"answer_{sid}" in wide.columns]
    out = meta.merge(wide, on="case_id", how="left")
    out = out[["case_id", "question"] + order]
    out.to_csv(ANSWERS_WIDE_CSV, index=False)


def _write_summary(results: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        "faithfulness", "answer_relevancy", "hallucination_rate",
        "context_precision", "context_recall",
    ]
    cols = [c for c in metrics if c in results.columns]
    overall = results.groupby("system")[cols].mean().round(4)
    overall.to_csv(SUMMARY_CSV)


def summarize(results: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if results is None:
        if RESULTS_CSV.exists():
            results = pd.read_csv(RESULTS_CSV)
        else:
            return {}
    metrics = [
        "faithfulness", "answer_relevancy", "hallucination_rate",
        "context_precision", "context_recall",
    ]
    cols = [c for c in metrics if c in results.columns]
    if not cols:
        return {}
    order = [s[0] for s in THREE_SYSTEMS]
    overall = results.groupby("system")[cols].mean().round(4).reindex(order)
    by_case = results.groupby(["case_id", "system"])[cols].mean().round(4)
    return {"overall": overall, "by_case": by_case}
