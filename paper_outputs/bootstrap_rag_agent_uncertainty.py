"""Bootstrap mean ± SD and 95% CIs for RAG/agent eval metrics from existing long CSVs.

No new LLM calls. Resamples:
  - query_agent_eval/outputs/per_case_long.csv  (by case within each model)
  - report_agent_eval/outputs/ragas_per_question.csv  (by question within each model)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAS = Path(__file__).resolve().parents[1]
QUERY_LONG = MAS / "query_agent_eval" / "outputs" / "per_case_long.csv"
RAGAS_LONG = MAS / "report_agent_eval" / "outputs" / "ragas_per_question.csv"
OUT_DIR = MAS / "paper_outputs"
SEED = 42
N_BOOT = 2000

QUERY_METRICS = [
    "judge_overall",
    "judge_relevance",
    "judge_xai_alignment",
    "judge_kdb_answerability",
    "judge_report_usefulness",
    "judge_safety",
    "schema_valid",
    "xai_factor_coverage",
    "required_type_coverage",
]

RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "semantic_similarity",
]


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        means.append(sample.mean())
    means = np.asarray(means)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(values.std(ddof=1) if n > 1 else 0.0), float(lo), float(hi), int(n)


def run_query() -> pd.DataFrame:
    df = pd.read_csv(QUERY_LONG)
    # Judge scores in long file are 1-5; paper reports as %. Keep both.
    rows = []
    for model, g in df.groupby("model"):
        for metric in QUERY_METRICS:
            if metric not in g.columns:
                continue
            vals = g[metric].astype(float).values
            # Convert 1-5 judge dims to % for reporting consistency with paper tables
            scale = 20.0 if metric.startswith("judge_") else 100.0 if metric in {
                "schema_valid", "xai_factor_coverage", "required_type_coverage"
            } else 1.0
            # schema_valid is bool/0-1; coverage already 0-1
            if metric == "schema_valid":
                scaled = vals.astype(float) * 100.0
            elif metric in {"xai_factor_coverage", "required_type_coverage"}:
                scaled = vals.astype(float) * 100.0
            elif metric.startswith("judge_"):
                scaled = vals.astype(float) * 20.0  # 1-5 -> %
            else:
                scaled = vals
            mean, sd, lo, hi, n = bootstrap_ci(scaled)
            rows.append({
                "eval": "query_planning",
                "model": model,
                "metric": metric,
                "unit": "percent" if scale != 1.0 or metric.startswith("judge_") or metric == "schema_valid" else "raw",
                "n_instances": n,
                "mean": round(mean, 4),
                "sd": round(sd, 4),
                "ci95_lo": round(lo, 4),
                "ci95_hi": round(hi, 4),
                "report": f"{mean:.1f} ± {sd:.1f} (95% CI {lo:.1f}–{hi:.1f})",
            })
    return pd.DataFrame(rows)


def run_ragas() -> pd.DataFrame:
    df = pd.read_csv(RAGAS_LONG)
    rows = []
    for model, g in df.groupby("model"):
        for metric in RAGAS_METRICS:
            if metric not in g.columns:
                continue
            vals = g[metric].astype(float).values
            mean, sd, lo, hi, n = bootstrap_ci(vals)
            rows.append({
                "eval": "report_agent_ragas",
                "model": model,
                "metric": metric,
                "unit": "score_0_1",
                "n_instances": n,
                "mean": round(mean, 4),
                "sd": round(sd, 4),
                "ci95_lo": round(lo, 4),
                "ci95_hi": round(hi, 4),
                "report": f"{mean:.3f} ± {sd:.3f} (95% CI {lo:.3f}–{hi:.3f})",
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q = run_query()
    r = run_ragas()
    all_df = pd.concat([q, r], ignore_index=True)
    q_path = OUT_DIR / "query_agent_bootstrap_ci.csv"
    r_path = OUT_DIR / "report_agent_ragas_bootstrap_ci.csv"
    all_path = OUT_DIR / "rag_agent_bootstrap_ci_all.csv"
    q.to_csv(q_path, index=False)
    r.to_csv(r_path, index=False)
    all_df.to_csv(all_path, index=False)
    print("Wrote", q_path)
    print("Wrote", r_path)
    print("Wrote", all_path)
    # Compact leaderboard-style prints
    print("\nQuery judge_overall:")
    print(q[q["metric"] == "judge_overall"][["model", "n_instances", "mean", "sd", "ci95_lo", "ci95_hi"]].to_string(index=False))
    print("\nRAGAS faithfulness:")
    print(r[r["metric"] == "faithfulness"][["model", "n_instances", "mean", "sd", "ci95_lo", "ci95_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
