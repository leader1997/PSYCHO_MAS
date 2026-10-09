"""Full HMS feature profiles for the three de-identified evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DATA_CSV = ROOT.parent / "HMS_2024-2025_PUBLIC_instchars.csv"
PROFILES_JSON = OUTPUT_DIR / "case_profiles.json"

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
SEED = 42


def _read_hms_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)


def _bucket(feature: str, raw_val: float, tertiles: dict[str, tuple[float, float]]) -> str:
    if np.isnan(raw_val):
        return "unknown"
    lo, hi = tertiles[feature]
    return "low" if raw_val <= lo else ("high" if raw_val > hi else "medium")


def _prepare_dataframe() -> pd.DataFrame:
    raw = _read_hms_csv(DATA_CSV)
    cols = FEATURES + [TARGET, "nrweight", "Finished"]
    df = raw[[c for c in cols if c in raw.columns]].copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Finished"] == 1].copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    min_features = int(0.6 * len(FEATURES))
    return df[df[FEATURES].notna().sum(axis=1) >= min_features].copy()


def build_case_profiles(force: bool = False) -> dict[str, dict]:
    """Rebuild full 25-feature profiles for high/low/borderline cases."""
    if PROFILES_JSON.exists() and not force:
        return json.loads(PROFILES_JSON.read_text(encoding="utf-8"))

    df = _prepare_dataframe()
    tertiles = {f: tuple(np.nanpercentile(df[f].values, [33, 66])) for f in FEATURES}

    X = df[FEATURES].values
    y = df[TARGET].values.astype(int)
    w = df["nrweight"].values.astype(float)

    X_tr_raw, X_tmp_raw, y_train, y_temp, w_train, _w_temp = train_test_split(
        X, y, w, test_size=0.2, random_state=SEED, stratify=y,
    )
    _X_val_raw, X_te_raw, _y_val, _y_test, _w_val, _w_test = train_test_split(
        X_tmp_raw, y_temp, _w_temp, test_size=0.5, random_state=SEED, stratify=y_temp,
    )

    preprocessor = joblib.load(OUTPUT_DIR / "models" / "preprocessor.joblib")
    cat_model: CatBoostClassifier = joblib.load(OUTPUT_DIR / "models" / "CatBoost.joblib")
    X_test = preprocessor.transform(X_te_raw).astype(np.float32)
    ref_proba = cat_model.predict_proba(X_test)[:, 1]

    picks = {
        "high_risk_case": int(np.argmax(ref_proba)),
        "low_risk_case": int(np.argmin(ref_proba)),
        "borderline_case": int(np.argmin(np.abs(ref_proba - 0.5))),
    }

    with open(OUTPUT_DIR / "local_explanations.json", encoding="utf-8") as f:
        local_cards = {c["case_id"]: c for c in json.load(f)}

    profiles: dict[str, dict] = {}
    for case_id, idx in picks.items():
        raw_row = X_te_raw[idx]
        card = local_cards[case_id]
        feature_levels = {
            f: _bucket(f, float(raw_row[i]), tertiles)
            for i, f in enumerate(FEATURES)
        }
        profiles[case_id] = {
            "case_id": case_id,
            "predicted_probability": card["predicted_probability"],
            "predicted_class": card["predicted_class"],
            "feature_levels": feature_levels,
            "raw_values": {
                f: None if np.isnan(raw_row[i]) else round(float(raw_row[i]), 4)
                for i, f in enumerate(FEATURES)
            },
        }

    PROFILES_JSON.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    return profiles


def load_case_profiles() -> dict[str, dict]:
    if not PROFILES_JSON.exists():
        return build_case_profiles()
    return json.loads(PROFILES_JSON.read_text(encoding="utf-8"))


def user_profile_block(profile: dict) -> str:
    lines = [
        f"- {feature}: {level}"
        for feature, level in profile["feature_levels"].items()
    ]
    return "USER PROFILE (HMS 2024-25 survey features):\n" + "\n".join(lines)


def prediction_block(profile: dict) -> str:
    return (
        f"PREDICTION:\n"
        f"Class: {profile['predicted_class']} "
        f"(probability {profile['predicted_probability']})."
    )
