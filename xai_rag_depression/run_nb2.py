#!/usr/bin/env python3
"""Standalone runner for notebook 02_stability_aggregation_cards.ipynb."""

from __future__ import annotations

import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from captum.attr import IntegratedGradients
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

FAST = os.environ.get("XAI_FAST", "1") == "1"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("..").resolve()
OUTPUT_DIR = Path("outputs")
(OUTPUT_DIR / "cards").mkdir(parents=True, exist_ok=True)

TARGET = "dep_or_anx"
FEATURES = [
    "age", "gender_male", "gender_female", "international", "fincur", "finpast",
    "food_worry", "sleep_wknight", "sleep_wkend", "exerc", "belong1", "belong2",
    "lonely", "aca_impa", "flourish", "positiveMH", "sub_any", "binge1pl",
    "smok_cig", "discrim_race", "discrim_gender", "discrim_sexual",
    "inst_type", "inst_size", "inst_public",
]
N_FEATURES = len(FEATURES)
N_TREES = 60 if FAST else 300
SEEDS = [42, 1, 2] if FAST else [42, 1, 2, 3, 4]
N_FOLDS = 2 if FAST else 3
DL_SEEDS = [42, 1] if FAST else [42, 1, 2]
DL_EPOCHS = 2 if FAST else 15
SHAP_N = 200 if FAST else 800
IG_N = 100 if FAST else 400
TOPK = 10

FEATURE_GROUPS = {
    "Demographics": ["age", "gender_male", "gender_female", "international"],
    "Financial stress": ["fincur", "finpast", "food_worry"],
    "Sleep and lifestyle": ["sleep_wknight", "sleep_wkend", "exerc"],
    "Social support and belonging": ["belong1", "belong2", "lonely"],
    "Academic impact": ["aca_impa"],
    "Positive mental health": ["flourish", "positiveMH"],
    "Substance use": ["sub_any", "binge1pl", "smok_cig"],
    "Discrimination": ["discrim_race", "discrim_gender", "discrim_sexual"],
    "Institution": ["inst_type", "inst_size", "inst_public"],
}
ML_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Extra Trees"]
DL_NAMES = ["FT-Transformer", "TabTransformer", "DCNv2"]
ALL_NAMES = ML_NAMES + DL_NAMES


def read_hms_csv(path, **kwargs):
    opts = {"low_memory": False, **kwargs}
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **opts)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **opts)


def load_data():
    raw = read_hms_csv(DATA_DIR / "HMS_2024-2025_PUBLIC_instchars.csv")
    cols = FEATURES + [TARGET, "nrweight", "Finished"]
    df = raw[[c for c in cols if c in raw.columns]].copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Finished"] == 1].dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[FEATURES].notna().sum(axis=1) >= int(0.6 * N_FEATURES)].copy()
    Xraw = df[FEATURES].values
    yall = df[TARGET].values.astype(int)
    wall = df["nrweight"].values.astype(float)
    print(f"Rows {len(df):,} | FAST={FAST} | seeds={SEEDS} | folds={N_FOLDS}", flush=True)
    return Xraw, yall, wall


def build_ml_models(spw):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=N_TREES,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=N_TREES,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=N_TREES,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=N_TREES,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_state=SEED,
            verbose=0,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=N_TREES,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def normalize(vec):
    a = np.abs(np.asarray(vec, dtype=float))
    s = a.sum()
    return a / s if s > 0 else a


def ml_importance(name, model, Xexpl):
    if name == "Logistic Regression":
        return normalize(model.coef_[0])
    if name == "XGBoost":
        c = model.get_booster().predict(xgb.DMatrix(Xexpl), pred_contribs=True)
        return normalize(np.abs(np.asarray(c)[:, :-1]).mean(0))
    sv = shap.TreeExplainer(model).shap_values(Xexpl)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return normalize(np.abs(sv).mean(0))


class FTTransformer(nn.Module):
    def __init__(self, n, d=64, h=4, l=2, p=0.1):
        super().__init__()
        self.tok = nn.ModuleList([nn.Linear(1, d) for _ in range(n)])
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        e = nn.TransformerEncoderLayer(d, h, d * 4, p, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(e, l)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        t = torch.stack([m(x[:, i : i + 1]) for i, m in enumerate(self.tok)], 1)
        c = self.cls.expand(x.size(0), -1, -1)
        return self.head(self.enc(torch.cat([c, t], 1))[:, 0]).squeeze(-1)


class TabTransformer(nn.Module):
    def __init__(self, n, d=32, h=4, l=2, p=0.1):
        super().__init__()
        self.emb = nn.ModuleList([nn.Linear(1, d) for _ in range(n)])
        e = nn.TransformerEncoderLayer(d, h, d * 4, p, batch_first=True)
        self.enc = nn.TransformerEncoder(e, l)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(n * d, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 1)
        )

    def forward(self, x):
        h = torch.stack([m(x[:, i : i + 1]) for i, m in enumerate(self.emb)], 1)
        return self.head(self.enc(h)).squeeze(-1)


class CrossNetV2(nn.Module):
    def __init__(self, n, l=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n, n) for _ in range(l)])

    def forward(self, x0):
        x = x0
        for w in self.layers:
            x = x0 * w(x) + x
        return x


class DCNv2(nn.Module):
    def __init__(self, n, cl=2, hid=128, p=0.1):
        super().__init__()
        self.cross = CrossNetV2(n, cl)
        self.deep = nn.Sequential(
            nn.Linear(n, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hid, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
        )
        self.head = nn.Linear(n + hid, 1)

    def forward(self, x):
        return self.head(torch.cat([self.cross(x), self.deep(x)], 1)).squeeze(-1)


def make_dl(name):
    return {"FT-Transformer": FTTransformer, "TabTransformer": TabTransformer, "DCNv2": DCNv2}[
        name
    ](N_FEATURES)


def train_dl_quick(model, Xt, yt, wt, seed):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xt, dtype=torch.float32),
        torch.tensor(yt, dtype=torch.float32),
        torch.tensor(wt, dtype=torch.float32),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)
    for _ in range(DL_EPOCHS):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).sum() / wb.sum()
            loss.backward()
            opt.step()
    return model


def dl_importance(model, Xexpl):
    model.eval()
    ig = IntegratedGradients(lambda x: model(x).unsqueeze(1))
    Xt = torch.tensor(Xexpl, dtype=torch.float32, device=DEVICE)
    base = torch.zeros((1, N_FEATURES), dtype=torch.float32, device=DEVICE)
    attr = ig.attribute(Xt, baselines=base, target=0, n_steps=24)
    return normalize(attr.abs().mean(0).detach().cpu().numpy())


def jaccard_topk(a, b, k):
    sa = set(np.argsort(a)[::-1][:k])
    sb = set(np.argsort(b)[::-1][:k])
    return len(sa & sb) / len(sa | sb)


def ranking_similarity(a, b):
    sp = spearmanr(a, b).correlation
    kt = kendalltau(a, b).correlation
    sp = 0 if np.isnan(sp) else sp
    kt = 0 if np.isnan(kt) else kt
    return float(np.mean([(sp + 1) / 2, (kt + 1) / 2, jaccard_topk(a, b, TOPK)]))


def run_stability(Xraw, yall, wall, base_imp):
    runs = {n: [] for n in ALL_NAMES}

    print("ML stability: seeds", flush=True)
    for sd in SEEDS:
        Xtr, Xte, ytr, yte, wtr, wte = train_test_split(
            Xraw, yall, wall, test_size=0.2, random_state=sd, stratify=yall
        )
        pre = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        Xtr_s = pre.fit_transform(Xtr).astype(np.float32)
        Xte_s = pre.transform(Xte).astype(np.float32)
        spw = (ytr == 0).sum() / (ytr == 1).sum()
        eidx = np.random.default_rng(sd).choice(len(Xte_s), size=min(SHAP_N, len(Xte_s)), replace=False)
        models = build_ml_models(spw)
        for name in ML_NAMES:
            models[name].fit(Xtr_s, ytr, sample_weight=wtr)
            runs[name].append(ml_importance(name, models[name], Xte_s[eidx]))
        print(f"  ML seed {sd} done", flush=True)

    print("ML stability: CV folds", flush=True)
    Xall_s = (
        Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        .fit_transform(Xraw)
        .astype(np.float32)
    )
    for tr_i, te_i in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xall_s, yall):
        spw = (yall[tr_i] == 0).sum() / (yall[tr_i] == 1).sum()
        eidx = np.random.default_rng(7).choice(te_i, size=min(SHAP_N, len(te_i)), replace=False)
        models = build_ml_models(spw)
        for name in ML_NAMES:
            models[name].fit(Xall_s[tr_i], yall[tr_i], sample_weight=wall[tr_i])
            runs[name].append(ml_importance(name, models[name], Xall_s[eidx]))
        print("  ML fold done", flush=True)

    print("ML stability: perturbation", flush=True)
    rng = np.random.default_rng(123)
    pidx = rng.choice(len(Xall_s), size=min(SHAP_N, len(Xall_s)), replace=False)
    models_full = build_ml_models((yall == 0).sum() / (yall == 1).sum())
    for name in ML_NAMES:
        models_full[name].fit(Xall_s, yall, sample_weight=wall)
    for _ in range(2):
        Xp = Xall_s[pidx] + rng.normal(0, 0.05, size=(len(pidx), N_FEATURES)).astype(np.float32)
        for name in ML_NAMES:
            runs[name].append(ml_importance(name, models_full[name], Xp))
    print("  ML perturbation done", flush=True)

    print("DL stability: reduced protocol", flush=True)
    pre_dl = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    Xtr_dl = pre_dl.fit_transform(Xraw).astype(np.float32)
    rng = np.random.default_rng(321)
    didx = rng.choice(len(Xtr_dl), size=min(IG_N, len(Xtr_dl)), replace=False)
    for name in DL_NAMES:
        for sd in DL_SEEDS:
            m = train_dl_quick(make_dl(name), Xtr_dl, yall.astype(np.float32), wall.astype(np.float32), sd)
            runs[name].append(dl_importance(m, Xtr_dl[didx]))
            runs[name].append(
                dl_importance(
                    m,
                    Xtr_dl[didx]
                    + rng.normal(0, 0.05, size=(len(didx), N_FEATURES)).astype(np.float32),
                )
            )
        print(f"  DL {name} done", flush=True)

    return runs


def compute_stability_scores(base_imp, runs):
    stab_rows = []
    for name in ALL_NAMES:
        base = base_imp[name].values
        sims = [ranking_similarity(base, r) for r in runs[name]]
        stab_rows.append(
            {
                "model": name,
                "stability_score": float(np.mean(sims)),
                "stability_std": float(np.std(sims)),
                "n_runs": len(sims),
            }
        )
    stab = pd.DataFrame(stab_rows)
    perf = pd.read_csv(OUTPUT_DIR / "model_performance.csv")
    stab = stab.merge(perf[["model", "macro_f1", "brier"]], on="model")
    stab["calibration_score"] = 1 - stab["brier"]
    stab["model_weight"] = stab["macro_f1"] * stab["stability_score"] * stab["calibration_score"]
    stab = stab.sort_values("model_weight", ascending=False)
    stab.to_csv(OUTPUT_DIR / "stability_scores.csv", index=False)
    return stab, perf


def compute_aggregated_importance(base_imp, stab):
    weights = stab.set_index("model")["model_weight"]
    agg = np.zeros(N_FEATURES)
    for name in ALL_NAMES:
        agg += weights[name] * base_imp[name].values
    agg_norm = agg / weights.sum()
    agg_norm = agg_norm / agg_norm.sum()
    agg_df = pd.DataFrame({"feature": FEATURES, "final_importance": agg_norm}).sort_values(
        "final_importance", ascending=False
    )

    topk_sets = {
        name: set(base_imp[name].sort_values(ascending=False).head(TOPK).index) for name in ALL_NAMES
    }
    tree_models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Extra Trees"]
    agg_df["agreement_score"] = agg_df["feature"].map(
        lambda f: sum(f in topk_sets[n] for n in ALL_NAMES) / len(ALL_NAMES)
    )
    agg_df["agreement_level"] = pd.cut(
        agg_df["agreement_score"], [-0.01, 0.4, 0.7, 1.01], labels=["low", "medium", "high"]
    )
    agg_df["tree_topk_frac"] = agg_df["feature"].map(
        lambda f: sum(f in topk_sets[n] for n in tree_models) / len(tree_models)
    )
    agg_df["neural_topk_frac"] = agg_df["feature"].map(
        lambda f: sum(f in topk_sets[n] for n in DL_NAMES) / len(DL_NAMES)
    )
    agg_df.to_csv(OUTPUT_DIR / "aggregated_importance.csv", index=False)
    return agg_df


def write_cards(agg_df, stab, perf):
    group_imp = []
    imp_by_feat = agg_df.set_index("feature")
    for g, feats in FEATURE_GROUPS.items():
        val = float(imp_by_feat.loc[feats, "final_importance"].sum())
        agr = float(np.mean([imp_by_feat.loc[f, "agreement_score"] for f in feats]))
        group_imp.append(
            {
                "group": g,
                "importance": round(val, 4),
                "agreement": "high" if agr >= 0.7 else ("medium" if agr >= 0.4 else "low"),
            }
        )
    group_imp = sorted(group_imp, key=lambda d: d["importance"], reverse=True)

    top_groups = group_imp[:3]
    insight = (
        "The aggregated explanation suggests that "
        + ", ".join(g["group"].lower() for g in top_groups)
        + " are the strongest contributors to depression-risk classification."
    )
    agg_card = {
        "card_type": "aggregated_explanation",
        "task": "depression_classification",
        "top_feature_groups": group_imp,
        "final_insight": insight,
    }
    with open(OUTPUT_DIR / "cards" / "aggregated_explanation.json", "w", encoding="utf-8") as f:
        json.dump(agg_card, f, indent=2)

    for _, r in perf.iterrows():
        sc = stab[stab.model == r["model"]].iloc[0]
        card = {
            "card_type": "model_performance",
            "model": r["model"],
            "family": r["family"],
            "metrics": {
                k: round(float(r[k]), 4)
                for k in ["auc", "accuracy", "f1", "macro_f1", "recall", "pr_auc", "brier"]
            },
            "stability_score": round(float(sc["stability_score"]), 4),
            "calibration_score": round(float(sc["calibration_score"]), 4),
            "model_weight": round(float(sc["model_weight"]), 4),
        }
        card_path = OUTPUT_DIR / "cards" / f"perf_{r['model'].replace(' ', '_')}.json"
        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)

    gcard = {
        "card_type": "global_feature_importance",
        "task": "depression_classification",
        "top_features": [
            {
                "feature": row["feature"],
                "importance": round(float(row["final_importance"]), 4),
                "agreement_score": round(float(row["agreement_score"]), 2),
                "agreement_level": str(row["agreement_level"]),
            }
            for _, row in agg_df.head(10).iterrows()
        ],
    }
    with open(OUTPUT_DIR / "cards" / "global_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(gcard, f, indent=2)

    local_path = OUTPUT_DIR / "local_explanations.json"
    with open(local_path, encoding="utf-8") as f:
        local_cards = json.load(f)
    for lc in local_cards:
        local_card = {"card_type": "local_explanation", "task": "depression_classification", **lc}
        out_path = OUTPUT_DIR / "cards" / f"local_{lc['case_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(local_card, f, indent=2)

    return agg_card


def main() -> int:
    print(f"Working directory: {Path.cwd()}", flush=True)

    gi_path = OUTPUT_DIR / "global_importance_per_model.csv"
    perf_path = OUTPUT_DIR / "model_performance.csv"
    if not gi_path.exists():
        print(f"Missing {gi_path}. Run run_nb1.py first.", file=sys.stderr)
        return 1
    if not perf_path.exists():
        print(f"Missing {perf_path}. Run run_nb1.py first.", file=sys.stderr)
        return 1

    base_imp = pd.read_csv(gi_path, index_col=0).loc[FEATURES]
    Xraw, yall, wall = load_data()
    runs = run_stability(Xraw, yall, wall, base_imp)
    stab, perf = compute_stability_scores(base_imp, runs)
    agg_df = compute_aggregated_importance(base_imp, stab)
    agg_card = write_cards(agg_df, stab, perf)

    print(json.dumps(agg_card, indent=2))
    print("\nDONE run_nb2", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
