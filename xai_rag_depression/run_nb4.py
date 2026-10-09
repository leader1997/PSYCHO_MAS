#!/usr/bin/env python3
"""Standalone runner for notebook 04_llm_generation_evaluation.ipynb."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import llm_client

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

OUTPUT_DIR = Path("outputs")
KB_DIR = OUTPUT_DIR / "kb"

SYSTEMS = [
    ("baseline", "S0: Baseline RAG"),
    ("rag_xai", "S1: RAG + local XAI + literature"),
    ("rag_agg_xai", "S2: RAG + aggregated XAI + literature"),
    ("full", "S3: RAG + local + aggregated XAI + literature"),
]

BASELINE_PROMPT = (
    "You are a careful research assistant explaining a depression-risk SCREENING model. "
    "Use ONLY the provided context. Do NOT provide a clinical diagnosis. "
    "Explain the model prediction clearly. Mention uncertainty. "
    "End with a careful, supportive, non-clinical recommendation. "
    "Structure the answer with these sections: Prediction, Interpretation, Uncertainty, Recommendation."
)

XAI_PROMPT = (
    "You are a careful research assistant explaining a depression-risk SCREENING model. "
    "Use ONLY the provided context. Do NOT provide a clinical diagnosis. Explain the model "
    "prediction using the provided XAI evidence and retrieved student mental-health literature. "
    "Explicitly mention uncertainty and any model disagreement. End with a careful, supportive, "
    "non-clinical recommendation grounded in the literature when possible. "
    "Structure the answer with these sections: Prediction, Main contributing factors, "
    "Model agreement, Uncertainty, Interpretation, Recommendation."
)

QUESTION = (
    "Explain this student's depression-risk screening result and the main contributing factors."
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


def load_artifacts():
    emb = np.load(KB_DIR / "embeddings.npy")
    with open(KB_DIR / "documents.json", encoding="utf-8") as f:
        docs = json.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(query: str, k: int = 5) -> list[dict]:
        q = embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        sims = emb @ q
        return [docs[i] for i in np.argsort(sims)[::-1][:k]]

    with open(OUTPUT_DIR / "local_explanations.json", encoding="utf-8") as f:
        local_cards = json.load(f)
    with open(OUTPUT_DIR / "cards" / "aggregated_explanation.json", encoding="utf-8") as f:
        agg_card = json.load(f)
    perf = pd.read_csv(OUTPUT_DIR / "model_performance.csv")
    stab = pd.read_csv(OUTPUT_DIR / "stability_scores.csv")
    faith = pd.read_csv(OUTPUT_DIR / "faithfulness.csv")
    dataset_doc = next(d for d in docs if d["id"] == "dataset_overview")["text"]

    return retrieve, local_cards, agg_card, perf, stab, faith, dataset_doc


def baseline_context(case: dict, dataset_doc: str) -> str:
    return (
        f"DATASET:\n{dataset_doc}\n\n"
        f"PREDICTION:\nPredicted class: {case['predicted_class']} "
        f"(probability {case['predicted_probability']})."
    )


def local_xai_block(case: dict) -> str:
    factors = "\n".join(
        f"- {c['feature']} ({c['level']}): {c['direction']} (SHAP {c['shap_value']})"
        for c in case["top_contributing_factors"]
    )
    return f"LOCAL XAI EVIDENCE (per-case top factors from CatBoost):\n{factors}"


def aggregated_xai_block(agg_card: dict) -> str:
    groups = "\n".join(
        f"- {g['group']}: importance {g['importance']}, agreement {g['agreement']}"
        for g in agg_card["top_feature_groups"][:5]
    )
    return (
        f"AGGREGATED GLOBAL XAI EVIDENCE (feature groups across 9 models):\n{groups}\n\n"
        f"AGGREGATED INSIGHT:\n{agg_card['final_insight']}"
    )


def literature_context(case: dict, retrieve, k: int = 6) -> str:
    query = (
        f"college student depression anxiety {case['predicted_class']} "
        + " ".join(c["feature"] for c in case["top_contributing_factors"][:3])
    )
    hits = retrieve(query, k=k)
    blocks = []
    for d in hits:
        if d["type"] == "literature":
            blocks.append(f"[{d['title']}]\n{d['text'][:700]}")
    if not blocks:
        for d in hits:
            blocks.append(f"[{d['type']}] {d['title']}: {d['text'][:400]}")
    return "RETRIEVED LITERATURE (vector KB):\n" + "\n\n".join(blocks[:k])


def build_context(
    system_id: str,
    case: dict,
    dataset_doc: str,
    agg_card: dict,
    retrieve,
) -> str:
    base = baseline_context(case, dataset_doc)
    if system_id == "baseline":
        return base
    if system_id == "rag_xai":
        return f"{base}\n\n{local_xai_block(case)}\n\n{literature_context(case, retrieve)}"
    if system_id == "rag_agg_xai":
        return f"{base}\n\n{aggregated_xai_block(agg_card)}\n\n{literature_context(case, retrieve)}"
    if system_id == "full":
        return (
            f"{base}\n\n{local_xai_block(case)}\n\n"
            f"{aggregated_xai_block(agg_card)}\n\n"
            f"{literature_context(case, retrieve)}"
        )
    raise ValueError(f"Unknown system: {system_id}")


def system_prompt(system_id: str) -> str:
    return BASELINE_PROMPT if system_id == "baseline" else XAI_PROMPT


def evaluate_row(
    case: dict,
    system_id: str,
    ctx: str,
    ans: str,
    retrieve,
) -> dict:
    row = {
        "case": case["case_id"],
        "system": system_id,
        "llm_backend": llm_client.backend_name(),
        "llm_model": llm_client.model_name(),
        "judge_backend": llm_client.judge_backend_name(),
        "judge_model": llm_client.judge_model_name(),
        "faithfulness": llm_client.judge(GROUNDED, f"CONTEXT:\n{ctx}\n\nANSWER:\n{ans}"),
        "answer_relevancy": llm_client.judge(RELEVANT, f"QUESTION: {QUESTION}\n\nANSWER:\n{ans}"),
        "hallucination_rate": llm_client.judge(HALLUC, f"CONTEXT:\n{ctx}\n\nANSWER:\n{ans}"),
    }
    if system_id == "baseline":
        row["context_precision"] = np.nan
        row["context_recall"] = np.nan
    else:
        retrieved = retrieve(QUESTION + " " + case["predicted_class"], k=8)
        ret_text = "\n".join(
            f"[{d['type']}] {d['title']}: {d['text'][:200]}" for d in retrieved
        )
        row["context_precision"] = llm_client.judge(
            CTX_PREC, f"QUESTION: {QUESTION}\n\nRETRIEVED:\n{ret_text}"
        )
        row["context_recall"] = llm_client.judge(
            CTX_REC, f"ANSWER:\n{ans}\n\nRETRIEVED:\n{ret_text}"
        )
    return row


def run_evaluation(retrieve, local_cards, agg_card, dataset_doc):
    case_limit = int(os.environ.get("XAI_CASE_LIMIT", str(len(local_cards))))
    rows = []
    example_answers: dict[str, dict[str, str]] = {}

    for case in local_cards[:case_limit]:
        print(f"=== {case['case_id']} ===", flush=True)
        example_answers[case["case_id"]] = {}

        for system_id, label in SYSTEMS:
            print(f"  {label}...", flush=True)
            ctx = build_context(system_id, case, dataset_doc, agg_card, retrieve)
            ans = llm_client.chat(
                system_prompt(system_id),
                f"CONTEXT:\n{ctx}\n\nQUESTION: {QUESTION}",
            )
            example_answers[case["case_id"]][system_id] = ans
            rows.append(evaluate_row(case, system_id, ctx, ans, retrieve))

        print(f"  done {case['case_id']}", flush=True)

    evaldf = pd.DataFrame(rows)
    evaldf.to_csv(OUTPUT_DIR / "rag_evaluation.csv", index=False)
    evaldf.to_csv(OUTPUT_DIR / "rag_ablation_evaluation.csv", index=False)
    with open(OUTPUT_DIR / "llm_example_answers.json", "w", encoding="utf-8") as f:
        json.dump(example_answers, f, indent=2, ensure_ascii=False)
    return evaldf, example_answers


def print_evaluation_summary(evaldf: pd.DataFrame, perf: pd.DataFrame, faith: pd.DataFrame, stab: pd.DataFrame):
    if evaldf.empty:
        print("No evaluation results to summarize.")
        return

    order = [s[0] for s in SYSTEMS]
    summary = (
        evaldf.groupby("system")[
            ["faithfulness", "answer_relevancy", "hallucination_rate", "context_precision", "context_recall"]
        ]
        .mean()
        .reindex(order)
    )
    print("\n=== RAG ablation comparison (LLM-as-judge means) ===")
    print(summary.round(3).to_string())

    print("\n=== Per-case breakdown ===")
    for case_id in evaldf["case"].unique():
        sub = evaldf[evaldf.case == case_id].set_index("system").reindex(order)
        print(f"\n{case_id}:")
        print(sub[["faithfulness", "answer_relevancy", "hallucination_rate"]].round(3).to_string())

    print("\n=== Model performance (test set) ===")
    print(
        perf[["model", "family", "auc", "macro_f1", "recall", "pr_auc", "brier"]]
        .round(4)
        .to_string(index=False)
    )
    print("\n=== Explanation faithfulness (deletion test, AUC drop) ===")
    print(faith.round(4).to_string(index=False))
    print("\n=== Explanation stability and model weights ===")
    print(
        stab[["model", "stability_score", "calibration_score", "model_weight"]]
        .round(4)
        .to_string(index=False)
    )


def main() -> int:
    print(f"Working directory: {Path.cwd()}")

    if not llm_client.is_available():
        print(
            "LLM unavailable. Default is Ollama (`ollama serve` + pull llama3).\n"
            "Or set XAI_LLM_BACKEND=openai and OPENAI_API_KEY in .env.\n"
            "Skipping live generation and evaluation."
        )
        return 1

    if not llm_client.is_judge_available():
        print(
            "Judge unavailable. Set OPENAI_API_KEY for GPT judge, or use Ollama:\n"
            "  XAI_JUDGE_BACKEND=ollama\n"
            "Skipping evaluation."
        )
        return 1

    retrieve, local_cards, agg_card, perf, stab, faith, dataset_doc = load_artifacts()
    with open(KB_DIR / "documents.json", encoding="utf-8") as f:
        n_docs = len(json.load(f))
    print(
        f"Loaded KB ({n_docs} docs), "
        f"{len(local_cards)} local cases, aggregated explanation card."
    )
    print("Systems:", ", ".join(f"{sid} ({label})" for sid, label in SYSTEMS))
    print(
        f"Generation ({llm_client.backend_name()}): {llm_client.model_name()} | "
        f"Judge ({llm_client.judge_backend_name()}): {llm_client.judge_model_name()}"
    )

    evaldf, example_answers = run_evaluation(retrieve, local_cards, agg_card, dataset_doc)
    print(f"\nSaved evaluation to {OUTPUT_DIR / 'rag_evaluation.csv'}")
    print(f"Saved ablation to {OUTPUT_DIR / 'rag_ablation_evaluation.csv'}")
    print(f"Saved example answers to {OUTPUT_DIR / 'llm_example_answers.json'}")

    if "high_risk_case" in example_answers:
        print("\n=== Full system answer: high_risk_case ===")
        print(example_answers["high_risk_case"].get("full", ""))
    elif example_answers:
        first = next(iter(example_answers))
        print(f"\n=== Full system answer: {first} ===")
        print(example_answers[first].get("full", ""))

    print_evaluation_summary(evaldf, perf, faith, stab)
    print("\nDONE run_nb4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
