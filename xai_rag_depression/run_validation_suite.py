#!/usr/bin/env python3
"""
Validation suite: bootstrap CIs, calibration/ECE, threshold metrics, fairness subgroups.
Uses saved XGBoost from outputs/models/ or trains fresh on E1 full features.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline_common import (
    ALL_FEATURES,
    OUTPUT_DIR,
    build_ml_models,
    load_hms_dataframe,
    prepare_splits,
    set_seeds,
)

OUT = OUTPUT_DIR / "strengthened"
FIG = OUT / "figures"
N_BOOT = int(os.environ.get("XAI_BOOTSTRAP_N", "500" if os.environ.get("XAI_FAST", "1") == "1" else "1000"))
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
PRIMARY_MODEL = "XGBoost"


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric_fn(y_true[idx], y_score[idx]))
    if not stats:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def calibration_slope_intercept(y_true, y_prob):
  # logistic calibration: logit(p) ~ a + b * logit(p_hat)
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(logit_p.reshape(-1, 1), y_true)
    return float(lr.coef_[0]), float(lr.intercept_)


def get_model_and_proba(splits):
    model_path = OUTPUT_DIR / "models" / "XGBoost.joblib"
    if model_path.exists() and len(splits["features"]) == len(ALL_FEATURES):
        clf = joblib.load(model_path)
        preprocessor = joblib.load(OUTPUT_DIR / "models" / "preprocessor.joblib")
        X_test = preprocessor.transform(splits["X_te_raw"]).astype(np.float32)
        proba = clf.predict_proba(X_test)[:, 1]
        return clf, proba, X_test
    clf = build_ml_models(splits["scale_pos_weight"])[PRIMARY_MODEL]
    clf.fit(splits["X_train"], splits["y_train"], sample_weight=splits["w_train"])
    proba = clf.predict_proba(splits["X_test"])[:, 1]
    return clf, proba, splits["X_test"]


def threshold_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0
    return {
        "threshold": threshold,
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": spec,
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0,
    }


def subgroup_metrics(df, splits, proba):
    from sklearn.model_selection import train_test_split

    features = splits["features"]
    idx = df.index.to_numpy()
    y = df[TARGET].values.astype(int)
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    y_tmp = df.loc[tmp_idx, TARGET].values.astype(int)
    _, te_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42, stratify=y_tmp)
    test_df = df.loc[te_idx].copy()
    test_df["proba"] = proba
    test_df["y"] = test_df[TARGET].astype(int)

    def eval_subgroup(mask, label):
        sub = test_df[mask]
        if len(sub) < 50 or sub["y"].nunique() < 2:
            return None
        p = sub["proba"].values
        yt = sub["y"].values
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        return {
            "subgroup": label,
            "n": len(sub),
            "auc": roc_auc_score(yt, p),
            "recall": recall_score(yt, pred, zero_division=0),
            "fpr": fp / (fp + tn) if (fp + tn) else 0,
            "fnr": fn / (fn + tp) if (fn + tp) else 0,
            "ece": expected_calibration_error(yt, p),
        }

    specs = [
        (test_df["gender_male"] == 1, "Male"),
        (test_df["gender_female"] == 1, "Female"),
        (test_df["international"] == 1, "International"),
        (test_df["international"] == 0, "Domestic"),
        (test_df["inst_public"] == 1, "Public institution"),
        (test_df["inst_public"] == 0, "Private institution"),
    ]
    rows = []
    for mask, label in specs:
        row = eval_subgroup(mask, label)
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


TARGET = "dep_or_anx"


def main() -> int:
    set_seeds()
    os.chdir(Path(__file__).resolve().parent)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = load_hms_dataframe(ALL_FEATURES)
    splits = prepare_splits(df, ALL_FEATURES)
    y_test = splits["y_test"]
    _, proba, _ = get_model_and_proba(splits)

    # Table B — bootstrap CIs
    metrics = {
        "auc": lambda yt, ys: roc_auc_score(yt, ys),
        "pr_auc": lambda yt, ys: average_precision_score(yt, ys),
        "f1": lambda yt, ys: f1_score(yt, (ys >= 0.5).astype(int)),
        "brier": lambda yt, ys: brier_score_loss(yt, ys),
    }
    table_b = []
    for mname, fn in metrics.items():
        mean, lo, hi = bootstrap_ci(y_test, proba, fn)
        table_b.append({
            "model": PRIMARY_MODEL,
            "metric": mname,
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "ci_95": f"{mean:.3f} [{lo:.3f}–{hi:.3f}]" if not np.isnan(mean) else "",
        })
    table_b_df = pd.DataFrame(table_b)
    table_b_df.to_csv(OUT / "table_b_bootstrap_ci.csv", index=False)

    # Table C — calibration
    ece = expected_calibration_error(y_test, proba)
    slope, intercept = calibration_slope_intercept(y_test, proba)
    table_c = pd.DataFrame([{
        "model": PRIMARY_MODEL,
        "brier": brier_score_loss(y_test, proba),
        "ece": ece,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }])
    table_c.to_csv(OUT / "table_c_calibration.csv", index=False)

    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", label=PRIMARY_MODEL)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration curve — {PRIMARY_MODEL}")
    ax.legend()
    fig.savefig(FIG / "calibration_curve_xgboost.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Threshold table
    thresh_rows = [threshold_metrics(y_test, proba, t) for t in THRESHOLDS]
    pd.DataFrame(thresh_rows).to_csv(OUT / "threshold_screening_metrics.csv", index=False)

    # Table D — fairness
    fair_df = subgroup_metrics(df, splits, proba)
    fair_df.to_csv(OUT / "table_d_fairness_subgroups.csv", index=False)

    print("=== Table B: Bootstrap 95% CI (XGBoost) ===")
    print(table_b_df[["metric", "ci_95"]].to_string(index=False))
    print("\n=== Table C: Calibration ===")
    print(table_c.round(4).to_string(index=False))
    print("\n=== Table D: Subgroup performance ===")
    print(fair_df.round(4).to_string(index=False) if not fair_df.empty else "(insufficient subgroup n)")
    print(f"\nSaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
