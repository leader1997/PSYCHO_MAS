#!/usr/bin/env python3
"""Feature-ablation experiments E1–E6 (Table A for paper)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

from pipeline_common import (
    EXPERIMENT_FEATURE_SETS,
    OUTPUT_DIR,
    build_ml_models,
    load_hms_dataframe,
    normalize,
    prepare_splits,
    set_seeds,
    tree_shap_values,
)

# Representative models for ablation (use --all-models for full 6 ML)
DEFAULT_MODELS = ["Logistic Regression", "XGBoost", "LightGBM"]


def top_features_from_shap(clf, name: str, X_sample: np.ndarray, features: list[str], k: int = 5) -> list[str]:
    if name == "Logistic Regression":
        imp = normalize(np.abs(clf.coef_[0]))
    else:
        sv = tree_shap_values(name, clf, X_sample[: min(200, len(X_sample))])
        imp = normalize(np.abs(sv).mean(0))
    order = np.argsort(imp)[::-1]
    return [features[i] for i in order[:k]]


def run_experiment(exp_id: str, meta: dict, models: list[str]) -> list[dict]:
    features = meta["features"]
    print(f"\n=== {exp_id}: {meta['description']} ({len(features)} features) ===", flush=True)
    df = load_hms_dataframe(features)
    splits = prepare_splits(df, features)
    rows = []
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    w_test = splits["w_test"]
    ml_models = build_ml_models(splits["scale_pos_weight"])

    for name in models:
        if name not in ml_models:
            continue
        clf = ml_models[name]
        clf.fit(splits["X_train"], splits["y_train"], sample_weight=splits["w_train"])
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        top5 = top_features_from_shap(clf, name, X_test, features)
        rows.append({
            "experiment": exp_id,
            "experiment_description": meta["description"],
            "n_features": len(features),
            "model": name,
            "auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "recall": recall_score(y_test, pred),
            "brier": brier_score_loss(y_test, proba),
            "top5_features": ", ".join(top5),
        })
        print(f"  {name}: AUC={rows[-1]['auc']:.4f} | top5={top5}", flush=True)
    return rows


def main() -> int:
    set_seeds()
    os.chdir(Path(__file__).resolve().parent)
    models = DEFAULT_MODELS
    if "--all-models" in sys.argv:
        models = list(build_ml_models(1.0).keys())

    all_rows = []
    for exp_id, meta in EXPERIMENT_FEATURE_SETS.items():
        all_rows.extend(run_experiment(exp_id, meta, models))

    out = OUTPUT_DIR / "strengthened"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "feature_ablation_performance.csv", index=False)

    # Paper Table A — pivot XGBoost across experiments
    xgb = df[df.model == "XGBoost"].copy()
    table_a = xgb[[
        "experiment", "experiment_description", "n_features",
        "auc", "pr_auc", "f1", "brier", "top5_features",
    ]].rename(columns={"experiment": "setting"})
    table_a.to_csv(out / "table_a_feature_ablation_xgb.csv", index=False)

    print("\n=== Table A (XGBoost across ablations) ===")
    print(table_a.round(4).to_string(index=False))
    print(f"\nSaved -> {out / 'feature_ablation_performance.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
