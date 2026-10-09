#!/usr/bin/env python3
"""Compare aggregation weighting schemes (uniform, F1, stability, calibration, combined)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from pipeline_common import ALL_FEATURES, OUTPUT_DIR

OUT = OUTPUT_DIR / "strengthened"
TOPK = 10


def jaccard_topk(a, b, k=TOPK):
    sa = set(np.argsort(a)[::-1][:k])
    sb = set(np.argsort(b)[::-1][:k])
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def aggregate(base_imp: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights / weights.sum()
    return (base_imp * w).sum(axis=1)


def main() -> int:
    os.chdir(Path(__file__).resolve().parent)
    OUT.mkdir(parents=True, exist_ok=True)

    gi = pd.read_csv(OUTPUT_DIR / "global_importance_per_model.csv", index_col=0).loc[ALL_FEATURES]
    stab = pd.read_csv(OUTPUT_DIR / "stability_scores.csv").set_index("model")
    perf = pd.read_csv(OUTPUT_DIR / "model_performance.csv").set_index("model")

    schemes = {
        "uniform": pd.Series(1.0, index=gi.columns),
        "performance_f1": perf["macro_f1"],
        "stability": stab["stability_score"],
        "calibration": stab["calibration_score"],
        "combined": stab["macro_f1"] * stab["stability_score"] * stab["calibration_score"],
    }

    agg_vectors = {name: aggregate(gi, w.reindex(gi.columns).fillna(0)) for name, w in schemes.items()}
    ref = agg_vectors["combined"]
    rows = []
    for name, vec in agg_vectors.items():
        if name == "combined":
            continue
        sp = spearmanr(ref, vec).correlation
        kt = kendalltau(ref, vec).correlation
        rows.append({
            "scheme": name,
            "spearman_vs_combined": 0 if np.isnan(sp) else sp,
            "kendall_vs_combined": 0 if np.isnan(kt) else kt,
            "jaccard_top10_vs_combined": jaccard_topk(ref.values, vec.values),
            "top5_features": ", ".join(vec.sort_values(ascending=False).head(5).index),
        })

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "aggregation_weighting_comparison.csv", index=False)

    top10 = pd.DataFrame({k: v.sort_values(ascending=False).head(10) for k, v in agg_vectors.items()})
    top10.to_csv(OUT / "aggregation_top10_by_scheme.csv")

    print("=== Weighting scheme robustness vs combined ===")
    print(comp.round(4).to_string(index=False))
    print(f"\nSaved -> {OUT / 'aggregation_weighting_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
