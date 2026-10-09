"""Shared data loading, feature sets, and model utilities for the XAI-RAG pipeline."""

from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

SEED = 42
TARGET = "dep_or_anx"
ALL_FEATURES = [
    "age", "gender_male", "gender_female", "international",
    "fincur", "finpast", "food_worry", "sleep_wknight", "sleep_wkend", "exerc",
    "belong1", "belong2", "lonely", "aca_impa", "flourish", "positiveMH",
    "sub_any", "binge1pl", "smok_cig",
    "discrim_race", "discrim_gender", "discrim_sexual",
    "inst_type", "inst_size", "inst_public",
]

FAST = os.environ.get("XAI_FAST", "1") == "1"
N_TREES = 60 if FAST else 300
DL_EPOCHS = 4 if FAST else 35
DL_PATIENCE = 2 if FAST else 8
BATCH_SIZE = 512
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENT_FEATURE_SETS: dict[str, dict] = {
    "E1_full": {
        "description": "Full model (all 25 features)",
        "features": ALL_FEATURES,
    },
    "E2_no_aca_impa": {
        "description": "Remove aca_impa (circularity test)",
        "features": [f for f in ALL_FEATURES if f != "aca_impa"],
    },
    "E3_no_proximal": {
        "description": "Remove outcome-proximal MH variables",
        "features": [f for f in ALL_FEATURES if f not in {"aca_impa", "positiveMH", "flourish"}],
    },
    "E4_social_financial_sleep": {
        "description": "Social, financial, sleep, lifestyle only",
        "features": [
            "belong1", "belong2", "lonely", "fincur", "finpast", "food_worry",
            "sleep_wknight", "sleep_wkend", "exerc",
        ],
    },
    "E5_no_demographics": {
        "description": "Remove demographics",
        "features": [f for f in ALL_FEATURES if f not in {"age", "gender_male", "gender_female", "international"}],
    },
    "E6_no_institution": {
        "description": "Remove institution variables",
        "features": [f for f in ALL_FEATURES if f not in {"inst_type", "inst_size", "inst_public"}],
    },
}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_hms_csv(path: Path, **kwargs) -> pd.DataFrame:
    opts = {"low_memory": False, **kwargs}
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **opts)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **opts)


def load_hms_dataframe(features: list[str] | None = None) -> pd.DataFrame:
    features = features or ALL_FEATURES
    raw = read_hms_csv(DATA_DIR / "HMS_2024-2025_PUBLIC_instchars.csv")
    cols = features + [TARGET, "nrweight", "Finished"]
    df = raw[[c for c in cols if c in raw.columns]].copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Finished"] == 1].copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    min_features = int(0.6 * len(features))
    df = df[df[features].notna().sum(axis=1) >= min_features].copy()
    return df


def prepare_splits(df: pd.DataFrame, features: list[str], seed: int = SEED):
    X = df[features].values
    y = df[TARGET].values.astype(int)
    w = df["nrweight"].values.astype(float)
    X_tr_raw, X_tmp_raw, y_train, y_temp, w_train, w_temp = train_test_split(
        X, y, w, test_size=0.2, random_state=seed, stratify=y,
    )
    X_val_raw, X_te_raw, y_val, y_test, w_val, w_test = train_test_split(
        X_tmp_raw, y_temp, w_temp, test_size=0.5, random_state=seed, stratify=y_temp,
    )
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train = preprocessor.fit_transform(X_tr_raw).astype(np.float32)
    X_val = preprocessor.transform(X_val_raw).astype(np.float32)
    X_test = preprocessor.transform(X_te_raw).astype(np.float32)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_te_raw": X_te_raw,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "w_train": w_train,
        "w_val": w_val,
        "w_test": w_test,
        "preprocessor": preprocessor,
        "scale_pos_weight": scale_pos_weight,
        "n_features": len(features),
        "features": features,
        "df": df,
    }


# --- DL model classes (same as run_nb1) ---


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
        tokens = torch.stack([t(x[:, i : i + 1]) for i, t in enumerate(self.tokenizer)], dim=1)
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
        h = torch.stack([e(x[:, i : i + 1]) for i, e in enumerate(self.embed)], dim=1)
        return self.head(self.encoder(h)).squeeze(-1)


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


def build_ml_models(scale_pos_weight: float, seed: int = SEED) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=N_TREES, min_samples_leaf=5, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=N_TREES, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=N_TREES, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, class_weight="balanced", random_state=seed,
            n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=N_TREES, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_state=seed, verbose=0,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=N_TREES, min_samples_leaf=5, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
    }


def dl_proba(model: nn.Module, Xa: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xa), 1024):
            xb = torch.tensor(Xa[i : i + 1024], dtype=torch.float32, device=DEVICE)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def train_dl(model, train_loader, X_val, y_val, w_val):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    best_auc, best_state, wait = -1, None, 0
    for _ in range(1, DL_EPOCHS + 1):
        model.train()
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).sum() / wb.sum()
            loss.backward()
            opt.step()
        auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
            y_val, dl_proba(model, X_val), sample_weight=w_val,
        )
        if auc > best_auc:
            best_auc, best_state, wait = auc, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= DL_PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def make_loader(Xa, ya, wa, shuffle: bool):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xa, dtype=torch.float32),
        torch.tensor(ya, dtype=torch.float32),
        torch.tensor(wa, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def tree_shap_values(name: str, clf, X_shap: np.ndarray) -> np.ndarray:
    import shap
    if name == "XGBoost":
        contribs = clf.get_booster().predict(xgb.DMatrix(X_shap), pred_contribs=True)
        return np.asarray(contribs)[:, :-1]
    return np.asarray(shap.TreeExplainer(clf).shap_values(X_shap))


def normalize(vec: np.ndarray) -> np.ndarray:
    a = np.abs(np.asarray(vec, dtype=float))
    s = a.sum()
    return a / s if s > 0 else a
