"""RAGAS evaluation aligned with Batal et al. (2026) Sustainability 18(2):875."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RAGAS_DIR = OUTPUT_DIR / "ragas_eval"
BENCHMARK_CSV = OUTPUT_DIR / "rag_eval_benchmark.csv"
FULL_RESULTS_CSV = OUTPUT_DIR / "rag_eval_full_results.csv"
RAGAS_RESULTS_CSV = OUTPUT_DIR / "rag_eval_ragas_results.csv"
RAGAS_SUMMARY_CSV = OUTPUT_DIR / "rag_eval_ragas_summary.csv"
RAGAS_BY_CATEGORY_CSV = OUTPUT_DIR / "rag_eval_ragas_by_category.csv"

RAGAS_METRICS = [
    "context_precision",
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "answer_correctness",
    "semantic_similarity",
]

RENAME_MAP = {"answer_similarity": "semantic_similarity"}

_RAGAS_READY: bool | None = None


def ragas_backend_name() -> str:
    return os.environ.get(
        "XAI_RAGAS_BACKEND", os.environ.get("XAI_LLM_BACKEND", "ollama")
    ).lower()


def ragas_ollama_url() -> str:
    return os.environ.get(
        "XAI_RAGAS_OLLAMA_URL", os.environ.get("XAI_OLLAMA_URL", "http://localhost:11434")
    )


def ragas_model_name() -> str:
    if ragas_backend_name() == "ollama":
        import llm_client

        return os.environ.get("XAI_RAGAS_MODEL") or llm_client.model_name()
    return os.environ.get("XAI_RAGAS_MODEL", "gpt-4o")


def ragas_embedding_model_name() -> str:
    if ragas_backend_name() == "ollama":
        return os.environ.get("XAI_RAGAS_EMBEDDING_MODEL", "nomic-embed-text")
    return os.environ.get("XAI_RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")


def is_ragas_available() -> bool:
    """True if RAGAS can run (Ollama or OpenAI configured)."""
    if ragas_backend_name() == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    import llm_client

    return llm_client.is_available()


def _ensure_ragas_imports():
    """Lazy-import ragas/langchain (avoids kernel crash on incompatible env)."""
    global _RAGAS_READY
    if _RAGAS_READY is True:
        return
    try:
        from ragas import evaluate  # noqa: F401
        from ragas.metrics import (  # noqa: F401
            answer_correctness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            faithfulness,
        )
        if ragas_backend_name() == "ollama":
            from langchain_community.chat_models import ChatOllama  # noqa: F401
            from langchain_community.embeddings import OllamaEmbeddings  # noqa: F401
        else:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: F401
        _RAGAS_READY = True
    except Exception as exc:
        _RAGAS_READY = False
        versions = []
        for pkg in ("ragas", "langchain-openai", "langchain-core", "pydantic"):
            try:
                import importlib.metadata as md

                versions.append(f"  {pkg}=={md.version(pkg)}")
            except Exception:
                versions.append(f"  {pkg}=<not installed>")
        raise RuntimeError(
            "RAGAS dependencies are missing or incompatible for this Python environment. "
            "This is usually a langchain-openai / pydantic version clash (not a bug in your RAG code).\n"
            "Fix: restart the kernel, then run:\n"
            "  pip install --force-reinstall -r reproducibility/requirements-ragas.txt\n"
            "Installed versions:\n"
            + "\n".join(versions)
            + f"\nOriginal error: {exc}"
        ) from exc


def _get_ragas_objects():
    _ensure_ragas_imports()
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
        context_recall,
        faithfulness,
    )

    if ragas_backend_name() == "ollama":
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings

        base_url = ragas_ollama_url()
        llm = ChatOllama(
            model=ragas_model_name(),
            base_url=base_url,
            temperature=0,
        )
        embeddings = OllamaEmbeddings(
            model=ragas_embedding_model_name(),
            base_url=base_url,
        )
    else:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for RAGAS evaluation (OpenAI backend)")
        llm = ChatOpenAI(model=ragas_model_name(), temperature=0)
        embeddings = OpenAIEmbeddings(model=ragas_embedding_model_name())

    metrics = [
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_correctness,
        answer_similarity,
    ]
    return evaluate, llm, embeddings, metrics


def _parse_contexts(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(c) for c in value if str(c).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(c) for c in parsed if str(c).strip()]
        except json.JSONDecodeError:
            pass
    return [c for c in text.split("|||") if c.strip()]


def build_ragas_dataset(
    benchmark: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    bench = benchmark.set_index("question_id")
    qids = set(benchmark["question_id"].astype(str))
    results = results[results["question_id"].astype(str).isin(qids)].copy()
    merged = results.merge(
        benchmark,
        on="question_id",
        how="left",
        suffixes=("", "_bench"),
    )
    rows = []
    for _, row in merged.iterrows():
        qid = row["question_id"]
        contexts = _parse_contexts(row.get("retrieved_contexts"))
        if not contexts:
            contexts = ["No retrieved context available."]
        question = row.get("question")
        if pd.isna(question):
            question = bench.loc[qid, "question"]
        ground_truth = row.get("expected_answer_points")
        if pd.isna(ground_truth):
            ground_truth = bench.loc[qid, "expected_answer_points"]
        category = row.get("category")
        if pd.isna(category):
            category = bench.loc[qid, "category"]
        answer_type = row.get("answer_type")
        if pd.isna(answer_type):
            answer_type = bench.loc[qid, "answer_type"]
        rows.append(
            {
                "question_id": qid,
                "category": category,
                "answer_type": answer_type,
                "question": question,
                "answer": row["answer"],
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    from datasets import Dataset

    evaluate, llm, embeddings, metrics = _get_ragas_objects()
    ds = Dataset.from_dict(
        {
            "question": batch_df["question"].tolist(),
            "answer": batch_df["answer"].tolist(),
            "contexts": batch_df["contexts"].tolist(),
            "ground_truth": batch_df["ground_truth"].tolist(),
        }
    )
    try:
        result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
        scored = result.to_pandas()
    except Exception as exc:
        print(f"  RAGAS batch error: {exc}", flush=True)
        scored = pd.DataFrame(
            {
                "question": batch_df["question"].tolist(),
                **{m: [np.nan] * len(batch_df) for m in RAGAS_METRICS},
            }
        )
    scored = scored.rename(columns=RENAME_MAP)
    out = batch_df[["question_id", "category", "answer_type"]].reset_index(drop=True)
    for col in RAGAS_METRICS:
        ragas_col = col if col in scored.columns else RENAME_MAP.get(col, col)
        if ragas_col in scored.columns:
            out[col] = scored[ragas_col].values
        else:
            out[col] = np.nan
    return out


def run_ragas_benchmark(
    benchmark: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
    batch_size: int | None = None,
    skip_existing: bool = True,
    ragas_results_csv: Path | None = None,
    ragas_summary_csv: Path | None = None,
    ragas_by_category_csv: Path | None = None,
    full_results_csv: Path | None = None,
) -> pd.DataFrame:
    """Score benchmark answers with six RAGAS metrics (GPT-4o default)."""
    _ensure_ragas_imports()

    ragas_results_csv = ragas_results_csv or RAGAS_RESULTS_CSV
    ragas_summary_csv = ragas_summary_csv or RAGAS_SUMMARY_CSV
    ragas_by_category_csv = ragas_by_category_csv or RAGAS_BY_CATEGORY_CSV
    full_results_csv = full_results_csv or FULL_RESULTS_CSV

    if benchmark is None:
        benchmark = pd.read_csv(BENCHMARK_CSV)
    if results is None:
        if not full_results_csv.exists():
            raise FileNotFoundError(
                f"Missing {full_results_csv}. Run generation first."
            )
        results = pd.read_csv(full_results_csv)

    eval_df = build_ragas_dataset(benchmark, results)
    if batch_size is None:
        batch_size = int(os.environ.get("XAI_RAGAS_BATCH_SIZE", "10"))

    existing_ids: set[str] = set()
    scored_parts: list[pd.DataFrame] = []
    if ragas_results_csv.exists() and skip_existing:
        prev = pd.read_csv(ragas_results_csv)
        if "context_precision" in prev.columns:
            existing_ids = set(prev["question_id"].astype(str))
            scored_parts.append(prev)

    pending = eval_df[~eval_df["question_id"].astype(str).isin(existing_ids)]
    if pending.empty:
        combined = pd.concat(scored_parts, ignore_index=True) if scored_parts else eval_df
        _write_summaries(
            combined,
            ragas_summary_csv=ragas_summary_csv,
            ragas_by_category_csv=ragas_by_category_csv,
        )
        return combined

    print(
        f"RAGAS backend: {ragas_backend_name()} | "
        f"evaluator: {ragas_model_name()} | embedding: {ragas_embedding_model_name()}",
        flush=True,
    )

    new_parts: list[pd.DataFrame] = []
    n = len(pending)
    for start in range(0, n, batch_size):
        batch = pending.iloc[start : start + batch_size]
        print(
            f"  RAGAS batch {start // batch_size + 1}/{(n + batch_size - 1) // batch_size} "
            f"({len(batch)} questions)...",
            flush=True,
        )
        new_parts.append(_evaluate_batch(batch))

    new_scored = pd.concat(new_parts, ignore_index=True)
    combined = pd.concat(scored_parts + [new_scored], ignore_index=True)
    combined.drop_duplicates(subset=["question_id"], keep="last", inplace=True)
    combined.to_csv(ragas_results_csv, index=False)

    metric_cols = ["question_id"] + [c for c in RAGAS_METRICS if c in combined.columns]
    full = results.drop(columns=[c for c in RAGAS_METRICS if c in results.columns], errors="ignore")
    full = full.merge(combined[metric_cols].drop_duplicates("question_id"), on="question_id", how="left")
    full.to_csv(full_results_csv, index=False)

    _write_summaries(
        combined,
        ragas_summary_csv=ragas_summary_csv,
        ragas_by_category_csv=ragas_by_category_csv,
    )
    return combined


def _write_summaries(
    scored: pd.DataFrame,
    ragas_summary_csv: Path | None = None,
    ragas_by_category_csv: Path | None = None,
) -> None:
    ragas_summary_csv = ragas_summary_csv or RAGAS_SUMMARY_CSV
    ragas_by_category_csv = ragas_by_category_csv or RAGAS_BY_CATEGORY_CSV
    RAGAS_DIR.mkdir(parents=True, exist_ok=True)
    overall = scored[RAGAS_METRICS].mean().round(4).to_frame("mean").T
    overall.to_csv(ragas_summary_csv, index=False)
    if "category" in scored.columns:
        by_cat = scored.groupby("category")[RAGAS_METRICS].mean().round(4)
        by_cat.to_csv(ragas_by_category_csv)


def summarize_ragas(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if df is None:
        if RAGAS_RESULTS_CSV.exists():
            df = pd.read_csv(RAGAS_RESULTS_CSV)
        elif FULL_RESULTS_CSV.exists():
            df = pd.read_csv(FULL_RESULTS_CSV)
        else:
            return {}
    cols = [c for c in RAGAS_METRICS if c in df.columns]
    if not cols:
        return {}
    overall = df[cols].mean().round(4).to_frame("mean").T
    by_category = df.groupby("category")[cols].mean().round(4) if "category" in df.columns else pd.DataFrame()
    by_type = df.groupby("answer_type")[cols].mean().round(4) if "answer_type" in df.columns else pd.DataFrame()
    return {"overall": overall, "by_category": by_category, "by_answer_type": by_type}
