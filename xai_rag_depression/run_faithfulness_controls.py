#!/usr/bin/env python3
"""XAI faithfulness controls: top-5, random-5 (100x), bottom-5 deletion (Table E)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline_common import (
    ALL_FEATURES,
    OUTPUT_DIR,
    build_ml_models,
    load_hms_dataframe,
    normalize,
    prepare_splits,
    set_seeds,
    tree_shap_values,
)

OUT = OUTPUT_DIR / "strengthened"
MODELS = ["XGBoost", "LightGBM", "CatBoost", "Logistic Regression"]
N_RANDOM = int(os.environ.get("XAI_RANDOM_DEL", "20" if os.environ.get("XAI_FAST", "1") == "1" else "100"))


def feature_importance_order(clf, name: str, X_sample: np.ndarray, features: list[str]) -> list[str]:
    if name == "Logistic Regression":
        imp = normalize(np.abs(clf.coef_[0]))
    else:
        sv = tree_shap_values(name, clf, X_sample[: min(300, len(X_sample))])
        imp = normalize(np.abs(sv).mean(0))
    return [features[i] for i in np.argsort(imp)[::-1]]


def auc_after_mask(clf, X_test, y_test, features, order, k, median_scaled):
    Xk = X_test.copy()
    for f in order[:k]:
        Xk[:, features.index(f)] = median_scaled[features.index(f)]
    proba = clf.predict_proba(Xk)[:, 1]
    return roc_auc_score(y_test, proba)


def main() -> int:
    set_seeds()
    os.chdir(Path(__file__).resolve().parent)
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_hms_dataframe(ALL_FEATURES)
    splits = prepare_splits(df, ALL_FEATURES)
    features = ALL_FEATURES
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    median_scaled = np.median(splits["X_train"], axis=0)
    rng = np.random.default_rng(42)

    rows = []
    for name in MODELS:
        path = OUTPUT_DIR / "models" / f"{name.replace(' ', '_')}.joblib"
        if path.exists():
            clf = joblib.load(path)
        else:
            clf = build_ml_models(splits["scale_pos_weight"])[name]
            clf.fit(splits["X_train"], splits["y_train"], sample_weight=splits["w_train"])

        order = feature_importance_order(clf, name, X_test, features)
        base_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        top_drop = base_auc - auc_after_mask(clf, X_test, y_test, features, order, 5, median_scaled)
        bottom_drop = base_auc - auc_after_mask(
            clf, X_test, y_test, features, list(reversed(order)), 5, median_scaled,
        )
        random_drops = []
        for _ in range(N_RANDOM):
            rand_feats = list(rng.choice(features, size=5, replace=False))
            rand_order = rand_feats + [f for f in features if f not in rand_feats]
            random_drops.append(
                base_auc - auc_after_mask(clf, X_test, y_test, features, rand_order, 5, median_scaled)
            )

        rows.append({
            "model": name,
            "base_auc": base_auc,
            "top5_auc_drop": top_drop,
            "random5_auc_drop_mean": float(np.mean(random_drops)),
            "random5_auc_drop_std": float(np.std(random_drops)),
            "bottom5_auc_drop": bottom_drop,
            "n_random_trials": N_RANDOM,
        })
        print(f"{name}: top5={top_drop:.4f} random5={np.mean(random_drops):.4f} bottom5={bottom_drop:.4f}")

    table_e = pd.DataFrame(rows)
    table_e.to_csv(OUT / "table_e_faithfulness_controls.csv", index=False)
    print(f"\nSaved -> {OUT / 'table_e_faithfulness_controls.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
