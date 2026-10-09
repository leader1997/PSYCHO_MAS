"""RAGAS answer-only scoring via the gpt-oss (Ollama) backend.

Mirrors xai_rag_depression/ragas_eval.py wiring but restricts the metric set to
the four answer metrics (retrieval is fixed, so context metrics are excluded):
  - faithfulness            (answer vs contexts)
  - answer_relevancy        (answer vs question)
  - answer_correctness      (answer vs gpt-oss reference / ground_truth)
  - semantic_similarity     (answer vs gpt-oss reference / ground_truth)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from report_agent_eval.config import cfg

ANSWER_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "semantic_similarity",
]

_RENAME = {"answer_similarity": "semantic_similarity"}


def _get_objects():
    """Build (evaluate, llm, embeddings, metrics) for the Ollama/gpt-oss backend."""
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        faithfulness,
    )

    backend = str(cfg("ragas", "backend", default="ollama")).lower()
    base_url = str(cfg("ollama", "base_url", default="http://localhost:11434")).rstrip("/")
    model = str(cfg("ragas", "model", default="gpt-oss:20b"))
    embed_model = str(cfg("ragas", "embedding_model", default="nomic-embed-text"))

    if backend == "openai":
        import os

        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY required for RAGAS OpenAI backend")
        llm = ChatOpenAI(model=model, temperature=0)
        embeddings = OpenAIEmbeddings(model=embed_model)
    else:
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings

        llm = ChatOllama(model=model, base_url=base_url, temperature=0)
        embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)

    metrics = [faithfulness, answer_relevancy, answer_correctness, answer_similarity]
    return evaluate, llm, embeddings, metrics


def backend_info() -> str:
    return (
        f"backend={cfg('ragas', 'backend', default='ollama')} "
        f"model={cfg('ragas', 'model', default='gpt-oss:20b')} "
        f"embed={cfg('ragas', 'embedding_model', default='nomic-embed-text')}"
    )


def _run_config():
    """Sequential execution + generous timeout for a slow local 20B judge."""
    from ragas.run_config import RunConfig

    timeout = int(cfg("ragas", "timeout_seconds", default=1200))
    max_workers = int(cfg("ragas", "max_workers", default=1))
    return RunConfig(timeout=timeout, max_workers=max_workers, max_retries=2)


def score_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Score one batch. Expects columns: question, answer, contexts, ground_truth.

    Returns the id columns plus the four answer metrics. On failure returns NaNs
    so the pipeline is resumable and never loses cached rows.
    """
    from datasets import Dataset

    evaluate, llm, embeddings, metrics = _get_objects()
    ds = Dataset.from_dict(
        {
            "question": batch_df["question"].tolist(),
            "answer": batch_df["answer"].tolist(),
            "contexts": batch_df["contexts"].tolist(),
            "ground_truth": batch_df["ground_truth"].tolist(),
        }
    )
    try:
        result = evaluate(
            ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=_run_config(),
        )
        scored = result.to_pandas().rename(columns=_RENAME)
    except Exception as exc:  # noqa: BLE001 - keep pipeline resumable
        print(f"  RAGAS batch error: {exc}", flush=True)
        scored = pd.DataFrame({m: [np.nan] * len(batch_df) for m in ANSWER_METRICS})

    id_cols = ["row_id", "model", "case_id", "qid", "stratum", "question_type"]
    out = batch_df[[c for c in id_cols if c in batch_df.columns]].reset_index(drop=True)
    for col in ANSWER_METRICS:
        src = col if col in scored.columns else _RENAME.get(col, col)
        out[col] = scored[src].values if src in scored.columns else np.nan
    return out
