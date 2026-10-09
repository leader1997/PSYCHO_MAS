#!/usr/bin/env python3
"""Notebook 1 - Models and Explainability (standalone runner)."""
import json
import os
import random
import warnings
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from captum.attr import IntegratedGradients
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

os.environ.setdefault("XAI_FAST", "1")

warnings.filterwarnings("ignore")

FAST = os.environ.get("XAI_FAST", "1") == "1"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("..").resolve()
OUTPUT_DIR = Path("outputs")
(OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "cards").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "kb").mkdir(parents=True, exist_ok=True)

TARGET = "dep_or_anx"
FEATURES = [
    "age", "gender_male", "gender_female",
    "international", "fincur", "finpast", "food_worry",
    "sleep_wknight", "sleep_wkend", "exerc",
    "belong1", "belong2", "lonely", "aca_impa", "flourish", "positiveMH",
    "sub_any", "binge1pl", "smok_cig",
    "discrim_race", "discrim_gender", "discrim_sexual",
    "inst_type", "inst_size", "inst_public",
]

N_TREES = 60 if FAST else 300
DL_EPOCHS = 4 if FAST else 35
DL_PATIENCE = 2 if FAST else 8
SHAP_N = 300 if FAST else 1500
IG_N = 150 if FAST else 600
BATCH_SIZE = 512
LR = 1e-3


def read_hms_csv(path, **kwargs):
    """Read an HMS CSV, handling mixed encodings in open-text responses."""
    opts = {"low_memory": False, **kwargs}
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **opts)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **opts)


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.tokenizer = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        tokens = torch.stack([t(x[:, i:i + 1]) for i, t in enumerate(self.tokenizer)], dim=1)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(h[:, 0]).squeeze(-1)


class TabTransformer(nn.Module):
    def __init__(self, n_features, d_model=32, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(n_features * d_model, 128),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x):
        h = torch.stack([e(x[:, i:i + 1]) for i, e in enumerate(self.embed)], dim=1)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)


class CrossNetV2(nn.Module):
    def __init__(self, n_features, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_features, n_features) for _ in range(n_layers)])

    def forward(self, x0):
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x


class DCNv2(nn.Module):
    def __init__(self, n_features, cross_layers=2, hidden=128, dropout=0.1):
        super().__init__()
        self.cross = CrossNetV2(n_features, cross_layers)
        self.deep = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.head = nn.Linear(n_features + hidden, 1)

    def forward(self, x):
        return self.head(torch.cat([self.cross(x), self.deep(x)], dim=1)).squeeze(-1)


def build_ml_models(scale_pos_weight):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=N_TREES, min_samples_leaf=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=N_TREES, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=N_TREES, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, class_weight="balanced", random_state=SEED,
            n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=N_TREES, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_state=SEED, verbose=0,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=N_TREES, min_samples_leaf=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
    }


def make_loader(Xa, ya, wa, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xa, dtype=torch.float32),
        torch.tensor(ya, dtype=torch.float32),
        torch.tensor(wa, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def dl_proba(model, Xa):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 1024):
            xb = torch.tensor(Xa[i:i + 1024], dtype=torch.float32, device=DEVICE)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def train_dl(model, name, train_loader, X_val, y_val, w_val):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    best_auc, best_state, wait = -1, None, 0
    for _epoch in range(1, DL_EPOCHS + 1):
        model.train()
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).sum() / wb.sum()
            loss.backward()
            opt.step()
        auc = roc_auc_score(y_val, dl_proba(model, X_val), sample_weight=w_val)
        if auc > best_auc:
            best_auc, best_state, wait = auc, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= DL_PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    print(f"  {name}: best val AUC={best_auc:.4f}", flush=True)
    return model


def normalize(vec):
    a = np.abs(np.asarray(vec, dtype=float))
    s = a.sum()
    return a / s if s > 0 else a


def tree_shap_values(name, ml_fitted, X_shap):
    if name == "XGBoost":
        contribs = ml_fitted[name].get_booster().predict(
            xgb.DMatrix(X_shap), pred_contribs=True,
        )
        return np.asarray(contribs)[:, :-1]
    sv = shap.TreeExplainer(ml_fitted[name]).shap_values(X_shap)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv


def bucket(f, raw_val, tertiles):
    if np.isnan(raw_val):
        return "unknown"
    lo, hi = tertiles[f]
    return "low" if raw_val <= lo else ("high" if raw_val > hi else "medium")


def main():
    print(f"Device: {DEVICE} | FAST={FAST} | features={len(FEATURES)}", flush=True)

    raw = read_hms_csv(DATA_DIR / "HMS_2024-2025_PUBLIC_instchars.csv")
    cols = FEATURES + [TARGET, "nrweight", "Finished"]
    df = raw[[c for c in cols if c in raw.columns]].copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Finished"] == 1].copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    min_features = int(0.6 * len(FEATURES))
    df = df[df[FEATURES].notna().sum(axis=1) >= min_features].copy()
    print(f"Cleaned rows: {len(df):,} | positive rate {df[TARGET].mean():.1%}", flush=True)

    X = df[FEATURES].values
    y = df[TARGET].values.astype(int)
    w = df["nrweight"].values.astype(float)

    X_tr_raw, X_tmp_raw, y_train, y_temp, w_train, w_temp = train_test_split(
        X, y, w, test_size=0.2, random_state=SEED, stratify=y,
    )
    X_val_raw, X_te_raw, y_val, y_test, w_val, w_test = train_test_split(
        X_tmp_raw, y_temp, w_temp, test_size=0.5, random_state=SEED, stratify=y_temp,
    )

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train = preprocessor.fit_transform(X_tr_raw).astype(np.float32)
    X_val = preprocessor.transform(X_val_raw).astype(np.float32)
    X_test = preprocessor.transform(X_te_raw).astype(np.float32)
    n_features = X_train.shape[1]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}", flush=True)

    ml_fitted = {}
    for name, clf in build_ml_models(scale_pos_weight).items():
        print(f"Training {name}...", flush=True)
        clf.fit(X_train, y_train, sample_weight=w_train)
        ml_fitted[name] = clf

    train_loader = make_loader(X_train, y_train, w_train, True)
    dl_defs = {
        "FT-Transformer": FTTransformer(n_features),
        "TabTransformer": TabTransformer(n_features),
        "DCNv2": DCNv2(n_features),
    }
    dl_fitted = {}
    for name, model in dl_defs.items():
        print(f"Training {name}...", flush=True)
        dl_fitted[name] = train_dl(model, name, train_loader, X_val, y_val, w_val)

    all_models = list(ml_fitted) + list(dl_fitted)

    def proba_of(name, Xa):
        if name in ml_fitted:
            return ml_fitted[name].predict_proba(Xa)[:, 1]
        return dl_proba(dl_fitted[name], Xa)

    perf_rows = []
    test_proba = {}
    for name in all_models:
        p = proba_of(name, X_test)
        test_proba[name] = p
        pred = (p >= 0.5).astype(int)
        perf_rows.append({
            "model": name,
            "auc": roc_auc_score(y_test, p),
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "recall": recall_score(y_test, pred),
            "pr_auc": average_precision_score(y_test, p),
            "brier": brier_score_loss(y_test, p),
            "family": "ML" if name in ml_fitted else "DL",
        })
    perf = pd.DataFrame(perf_rows).sort_values("auc", ascending=False)
    perf.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)
    print(perf.round(4).to_string(), flush=True)

    rng = np.random.default_rng(SEED)
    shap_idx = rng.choice(len(X_test), size=min(SHAP_N, len(X_test)), replace=False)
    ig_idx = rng.choice(len(X_test), size=min(IG_N, len(X_test)), replace=False)
    X_shap = X_test[shap_idx]
    X_ig = torch.tensor(X_test[ig_idx], dtype=torch.float32, device=DEVICE)
    baseline = torch.zeros((1, n_features), dtype=torch.float32, device=DEVICE)

    global_imp = {}
    method_used = {}

    lr = ml_fitted["Logistic Regression"]
    lr_shap = shap.LinearExplainer(lr, X_train).shap_values(X_shap)
    global_imp["Logistic Regression"] = normalize(np.abs(lr_shap).mean(0))
    method_used["Logistic Regression"] = "coefficients+LinearSHAP"
    odds_ratios = dict(zip(FEATURES, np.exp(lr.coef_[0])))

    for name in ["Random Forest", "Extra Trees", "XGBoost", "LightGBM", "CatBoost"]:
        global_imp[name] = normalize(np.abs(tree_shap_values(name, ml_fitted, X_shap)).mean(0))
        method_used[name] = "TreeSHAP"

    perm_check = {}
    for name in ["Random Forest", "Extra Trees"]:
        r = permutation_importance(
            ml_fitted[name], X_shap, y_test[shap_idx],
            n_repeats=5, random_state=SEED, scoring="roc_auc",
        )
        perm_check[name] = normalize(r.importances_mean)

    for name, model in dl_fitted.items():
        model.eval()
        ig = IntegratedGradients(lambda x: model(x).unsqueeze(1))
        attr = ig.attribute(X_ig, baselines=baseline, target=0, n_steps=32)
        global_imp[name] = normalize(attr.abs().mean(0).detach().cpu().numpy())
        method_used[name] = "IntegratedGradients"

    imp_df = pd.DataFrame(global_imp, index=FEATURES)
    imp_df.to_csv(OUTPUT_DIR / "global_importance_per_model.csv")
    pd.Series(method_used).to_csv(OUTPUT_DIR / "xai_method_per_model.csv", header=["method"])
    for name in all_models:
        top5 = imp_df[name].sort_values(ascending=False).head(5).index.tolist()
        print(f"  {name}: {top5}", flush=True)

    faith_rows = []
    median_scaled = np.median(X_train, axis=0)
    for name in all_models:
        order = imp_df[name].sort_values(ascending=False).index.tolist()
        base_auc = roc_auc_score(y_test, test_proba[name])
        Xk = X_test.copy()
        for f in order[:5]:
            Xk[:, FEATURES.index(f)] = median_scaled[FEATURES.index(f)]
        masked_auc = roc_auc_score(y_test, proba_of(name, Xk))
        faith_rows.append({
            "model": name,
            "base_auc": base_auc,
            "masked_top5_auc": masked_auc,
            "auc_drop": base_auc - masked_auc,
        })
    faith = pd.DataFrame(faith_rows).sort_values("auc_drop", ascending=False)
    faith.to_csv(OUTPUT_DIR / "faithfulness.csv", index=False)
    print(faith.round(4).to_string(), flush=True)

    tertiles = {f: np.nanpercentile(df[f].values, [33, 66]) for f in FEATURES}
    cat_expl = shap.TreeExplainer(ml_fitted["CatBoost"])
    ref_proba = test_proba["CatBoost"]
    picks = {
        "high_risk_case": int(np.argmax(ref_proba)),
        "low_risk_case": int(np.argmin(ref_proba)),
        "borderline_case": int(np.argmin(np.abs(ref_proba - 0.5))),
    }
    local_cards = []
    for label, idx in picks.items():
        sv = np.asarray(cat_expl.shap_values(X_test[idx:idx + 1]))
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        sv = sv[0]
        contrib = sorted(zip(FEATURES, sv), key=lambda t: abs(t[1]), reverse=True)[:5]
        raw_row = X_te_raw[idx]
        local_cards.append({
            "case_id": label,
            "predicted_probability": round(float(ref_proba[idx]), 3),
            "predicted_class": (
                "elevated depression/anxiety risk" if ref_proba[idx] >= 0.5 else "not elevated"
            ),
            "explainer": "CatBoost TreeSHAP",
            "top_contributing_factors": [
                {
                    "feature": f,
                    "level": bucket(f, raw_row[FEATURES.index(f)], tertiles),
                    "direction": "increases risk" if v > 0 else "decreases risk",
                    "shap_value": round(float(v), 4),
                }
                for f, v in contrib
            ],
        })
    with open(OUTPUT_DIR / "local_explanations.json", "w", encoding="utf-8") as fh:
        json.dump(local_cards, fh, indent=2)
    print(json.dumps(local_cards[0], indent=2), flush=True)

    for name, clf in ml_fitted.items():
        joblib.dump(clf, OUTPUT_DIR / "models" / f"{name.replace(' ', '_')}.joblib")
    for name, model in dl_fitted.items():
        torch.save(model.state_dict(), OUTPUT_DIR / "models" / f"{name.replace(' ', '_')}.pt")
    joblib.dump(preprocessor, OUTPUT_DIR / "models" / "preprocessor.joblib")
    with open(OUTPUT_DIR / "lr_and_perm_extras.json", "w", encoding="utf-8") as fh:
        json.dump({
            "odds_ratios": {k: round(float(v), 4) for k, v in odds_ratios.items()},
            "perm_check_rf": {
                f: round(float(v), 4)
                for f, v in zip(FEATURES, perm_check["Random Forest"])
            },
            "perm_check_extra_trees": {
                f: round(float(v), 4)
                for f, v in zip(FEATURES, perm_check["Extra Trees"])
            },
        }, fh, indent=2)

    print("Saved models + artifacts to outputs/.", flush=True)
    print("DONE run_nb1", flush=True)


if __name__ == "__main__":
    main()
