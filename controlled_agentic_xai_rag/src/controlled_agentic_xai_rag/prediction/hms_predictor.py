"""Live HMS CatBoost prediction + TreeSHAP → XAI card + case profile (Phase 1 models)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from controlled_agentic_xai_rag.config_loader import cfg_get, project_root

TARGET = "dep_or_anx"
SEED = 42
FEATURES = [
    "age", "gender_male", "gender_female",
    "international", "fincur", "finpast", "food_worry",
    "sleep_wknight", "sleep_wkend", "exerc",
    "belong1", "belong2", "lonely", "aca_impa", "flourish", "positiveMH",
    "sub_any", "binge1pl", "smok_cig",
    "discrim_race", "discrim_gender", "discrim_sexual",
    "inst_type", "inst_size", "inst_public",
]

FEATURE_DISPLAY = {
    "age": "Age",
    "gender_male": "Male gender",
    "gender_female": "Female gender",
    "international": "International student status",
    "fincur": "Current financial situation",
    "finpast": "Past financial stress",
    "food_worry": "Food insecurity worry",
    "sleep_wknight": "Weeknight sleep",
    "sleep_wkend": "Weekend sleep",
    "exerc": "Exercise frequency",
    "belong1": "Campus belonging (item 1)",
    "belong2": "Campus belonging (item 2)",
    "lonely": "Loneliness",
    "aca_impa": "Academic impairment",
    "flourish": "Flourishing",
    "positiveMH": "Positive mental health",
    "sub_any": "Substance use",
    "binge1pl": "Binge drinking",
    "smok_cig": "Cigarette smoking",
    "discrim_race": "Racial discrimination",
    "discrim_gender": "Gender discrimination",
    "discrim_sexual": "Sexual-orientation discrimination",
    "inst_type": "Institution type",
    "inst_size": "Institution size",
    "inst_public": "Public institution",
}

WARNING = "This is a model-based screening signal and not a clinical diagnosis."

PRESET_ALIASES = {
    "high_risk_case": "high_risk_case",
    "low_risk_case": "low_risk_case",
    "borderline_risk_case": "borderline_risk_case",
    "borderline_case": "borderline_risk_case",
}


class PredictionUnavailableError(RuntimeError):
    """Raised when Phase 1 model artifacts or HMS CSV are missing."""


@dataclass
class HMSPredictionResult:
    case_id: str
    card: dict
    profile: dict
    test_index: int
    prediction_probability: float
    predicted_class: str

    def to_audit_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "test_index": self.test_index,
            "prediction_probability": self.prediction_probability,
            "predicted_class": self.predicted_class,
            "model_name": self.card.get("model_name"),
            "top_positive_features": [f["feature"] for f in self.card.get("top_positive_factors", [])],
        }


def _phase1_root() -> Path:
    raw = cfg_get("paths", "phase1_root")
    if raw:
        return Path(raw)
    return (project_root().parent / "xai_rag_depression").resolve()


def _hms_csv_path() -> Path:
    raw = cfg_get("paths", "hms_csv")
    if raw:
        return Path(raw)
    return (project_root().parent.parent / "HMS_2024-2025_PUBLIC_instchars.csv").resolve()


def _patch_preprocessor(preprocessor) -> object:
    """Restore sklearn SimpleImputer attrs missing after cross-version unpickling."""
    from sklearn.impute import SimpleImputer

    for _, step in getattr(preprocessor, "steps", []):
        if isinstance(step, SimpleImputer) and not hasattr(step, "_fill_dtype"):
            step._fill_dtype = np.dtype(np.float64)
    return preprocessor


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


def _display_name(feature: str) -> str:
    return FEATURE_DISPLAY.get(feature, feature.replace("_", " ").title())


def _factor_entry(feature: str, shap_value: float, level: str) -> dict:
    if shap_value >= 0:
        direction = "increases predicted screening risk"
        interpretation = (
            f"This feature contributed positively to the model's risk prediction (level: {level})."
        )
    else:
        direction = "decreases predicted screening risk"
        interpretation = (
            f"This feature contributed negatively to the model's risk prediction (level: {level})."
        )
    return {
        "feature": feature,
        "display_name": _display_name(feature),
        "contribution": round(float(shap_value), 4),
        "direction": direction,
        "interpretation": interpretation,
    }


@lru_cache(maxsize=1)
def _load_context() -> dict:
    phase1 = _phase1_root()
    model_dir = phase1 / "outputs" / "models"
    pre_path = model_dir / "preprocessor.joblib"
    cat_path = model_dir / "CatBoost.joblib"
    hms_path = _hms_csv_path()

    missing = [str(p) for p in (pre_path, cat_path, hms_path) if not p.exists()]
    if missing:
        raise PredictionUnavailableError(
            "HMS live prediction unavailable. Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun Phase 1 (xai_rag_depression/run_nb1.py) or set paths in config.yaml."
        )

    try:
        import shap
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise PredictionUnavailableError(
            "Install prediction dependencies: pip install catboost shap scikit-learn joblib"
        ) from exc

    preprocessor = _patch_preprocessor(joblib.load(pre_path))
    cat_model: CatBoostClassifier = joblib.load(cat_path)

    raw = _read_hms_csv(hms_path)
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

    tertiles = {f: tuple(np.nanpercentile(df[f].values, [33, 66])) for f in FEATURES}

    X = df[FEATURES].values
    y = df[TARGET].values.astype(int)
    w = df["nrweight"].values.astype(float)

    _x_tr, X_tmp_raw, _y_tr, y_temp, _w_tr, w_temp = train_test_split(
        X, y, w, test_size=0.2, random_state=SEED, stratify=y,
    )
    _x_val, X_te_raw, _y_val, _y_test, _w_val, _w_test = train_test_split(
        X_tmp_raw, y_temp, w_temp, test_size=0.5, random_state=SEED, stratify=y_temp,
    )

    X_test = preprocessor.transform(X_te_raw).astype(np.float32)
    test_proba = cat_model.predict_proba(X_test)[:, 1]
    explainer = shap.TreeExplainer(cat_model)

    preset_indices = {
        "high_risk_case": int(np.argmax(test_proba)),
        "low_risk_case": int(np.argmin(test_proba)),
        "borderline_risk_case": int(np.argmin(np.abs(test_proba - 0.5))),
    }

    return {
        "X_te_raw": X_te_raw,
        "X_test": X_test,
        "tertiles": tertiles,
        "test_proba": test_proba,
        "explainer": explainer,
        "preset_indices": preset_indices,
        "n_test": len(X_test),
    }


def get_test_set_size() -> int:
    return int(_load_context()["n_test"])


def get_preset_indices() -> dict[str, int]:
    return dict(_load_context()["preset_indices"])


def _build_card_and_profile(case_id: str, test_index: int) -> HMSPredictionResult:
    ctx = _load_context()
    n_test = ctx["n_test"]
    if test_index < 0 or test_index >= n_test:
        raise ValueError(f"hms test_index must be in [0, {n_test - 1}], got {test_index}")

    X_test = ctx["X_test"]
    X_te_raw = ctx["X_te_raw"]
    tertiles = ctx["tertiles"]
    explainer = ctx["explainer"]
    test_proba = ctx["test_proba"]

    prob = float(test_proba[test_index])
    predicted_class = (
        "elevated depression/anxiety risk" if prob >= 0.5 else "not elevated"
    )
    prediction_label = "elevated_risk" if prob >= 0.5 else "not_elevated_risk"

    sv = np.asarray(explainer.shap_values(X_test[test_index : test_index + 1]))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    sv = sv[0]

    raw_row = X_te_raw[test_index]
    feature_levels = {
        f: _bucket(f, float(raw_row[i]), tertiles) for i, f in enumerate(FEATURES)
    }
    raw_values = {
        f: None if np.isnan(raw_row[i]) else round(float(raw_row[i]), 4)
        for i, f in enumerate(FEATURES)
    }

    pairs = list(zip(FEATURES, sv))
    positive = sorted([(f, v) for f, v in pairs if v > 0], key=lambda t: -t[1])[:5]
    protective = sorted([(f, v) for f, v in pairs if v <= 0], key=lambda t: t[1])[:5]

    card = {
        "case_id": case_id,
        "prediction_label": prediction_label,
        "prediction_probability": round(prob, 4),
        "model_name": "CatBoost",
        "calibration_status": "calibrated",
        "top_positive_factors": [
            _factor_entry(f, v, feature_levels[f]) for f, v in positive
        ],
        "top_protective_factors": [
            _factor_entry(f, v, feature_levels[f]) for f, v in protective
        ],
        "important_warning": WARNING,
    }

    profile = {
        "case_id": case_id,
        "predicted_probability": round(prob, 4),
        "predicted_class": predicted_class,
        "feature_levels": feature_levels,
        "raw_values": raw_values,
    }

    return HMSPredictionResult(
        case_id=case_id,
        card=card,
        profile=profile,
        test_index=test_index,
        prediction_probability=prob,
        predicted_class=predicted_class,
    )


def predict_from_test_index(test_index: int, case_id: str | None = None) -> HMSPredictionResult:
    cid = case_id or f"live_test_{test_index}"
    return _build_card_and_profile(cid, test_index)


def predict_preset_case(preset: str) -> HMSPredictionResult:
    key = PRESET_ALIASES.get(preset, preset)
    indices = get_preset_indices()
    if key not in indices:
        raise KeyError(f"Unknown preset case: {preset}")
    return _build_card_and_profile(key, indices[key])


def resolve_live_case(preset: str | None, test_index: int | None) -> HMSPredictionResult:
    """Resolve live prediction from explicit test row or named preset."""
    if test_index is not None:
        cid = preset or f"live_test_{test_index}"
        return predict_from_test_index(test_index, case_id=cid)
    if preset:
        return predict_preset_case(preset)
    raise ValueError("Provide preset case id or hms_test_index for live prediction")
