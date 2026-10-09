"""Build interim Table X from retrieval + claim support (+ RAGAS if available)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "paper_outputs" / "query_strategy_ablation_v2"
SEED = 42
N_BOOT = 2000


def boot_case(series_by_case: dict):
    vals = np.asarray(list(series_by_case.values()), float)
    vals = vals[~np.isnan(vals)]
    rng = np.random.default_rng(SEED)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi), series_by_case


def paired(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    d = np.asarray([a[k] - b[k] for k in keys], float)
    rng = np.random.default_rng(SEED)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    ret = pd.read_csv(OUT / "retrieval_by_case_arm.csv")
    ans = pd.read_csv(OUT / "answers_index.csv")
    case_claim = ans.groupby(["case_id", "arm"], as_index=False)["claim_support_rate"].mean()

    metrics = [
        ("retrieval", ret, m)
        for m in ["recall_at_k", "ndcg_at_k", "mrr_at_k", "hit_at_k", "xai_coverage"]
    ]
    metrics.append(("report", case_claim, "claim_support_rate"))

    if (OUT / "ragas_scores.csv").exists():
        sc = pd.read_csv(OUT / "ragas_scores.csv")
        if len(sc):
            cm = sc.groupby(["case_id", "arm"], as_index=False)[
                ["faithfulness", "answer_relevancy"]
            ].mean()
            for m in ["faithfulness", "answer_relevancy"]:
                if cm[m].notna().any():
                    metrics.append(("report", cm, m))

    label = {
        "recall_at_k": "Recall@5",
        "ndcg_at_k": "nDCG@5",
        "mrr_at_k": "MRR@5",
        "hit_at_k": "Hit@5",
        "xai_coverage": "XAI evidence coverage",
        "claim_support_rate": "Claim support rate",
        "faithfulness": "RAGAS faithfulness",
        "answer_relevancy": "Answer relevancy",
    }

    out_rows = []
    for _level, df, m in metrics:
        means = {}
        for arm in ["direct", "template", "llm_plan"]:
            g = df[df.arm == arm].set_index("case_id")[m].to_dict()
            mean, lo, hi, raw = boot_case(g)
            means[arm] = (mean, lo, hi, raw)
        d1, l1, h1 = paired(means["llm_plan"][3], means["direct"][3])
        d2, l2, h2 = paired(means["llm_plan"][3], means["template"][3])
        out_rows.append(
            {
                "metric": label[m],
                "direct": f"{means['direct'][0]:.3f} ({means['direct'][1]:.3f}-{means['direct'][2]:.3f})",
                "template": f"{means['template'][0]:.3f} ({means['template'][1]:.3f}-{means['template'][2]:.3f})",
                "llm_plan": f"{means['llm_plan'][0]:.3f} ({means['llm_plan'][1]:.3f}-{means['llm_plan'][2]:.3f})",
                "delta_llm_direct": f"{d1:+.3f}",
                "ci_llm_direct": f"[{l1:+.3f}, {h1:+.3f}]",
                "delta_llm_template": f"{d2:+.3f}",
                "ci_llm_template": f"[{l2:+.3f}, {h2:+.3f}]",
            }
        )
    tab = pd.DataFrame(out_rows)
    tab.to_csv(OUT / "table_x_interim_retrieval_claim.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
