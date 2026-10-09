"""Sample 50 real HMS test cases and build structured cases for the query agent.

Reuses the controlled_agentic_xai_rag prediction stack (CatBoost + TreeSHAP).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from query_agent_eval.config import cfg, path

_CAI_SRC = path("cai_src")
if str(_CAI_SRC) not in sys.path:
    sys.path.insert(0, str(_CAI_SRC))

from controlled_agentic_xai_rag.prediction.hms_predictor import (  # noqa: E402
    _load_context,
    predict_from_test_index,
)


def _threshold() -> float:
    return float(cfg("profiles", "risk_threshold", default=0.5))


def _band() -> float:
    return float(cfg("profiles", "near_threshold_band", default=0.1))


def _planning_strategy(prob: float) -> str:
    thr = _threshold()
    if abs(prob - thr) <= _band():
        return "borderline"
    return "high_risk" if prob >= thr else "low_risk"


def _risk_label(prob: float) -> str:
    return "elevated_screening_risk" if prob >= _threshold() else "not_elevated_screening_risk"


def select_indices() -> list[tuple[str, int]]:
    """Return [(stratum, test_index)] stratified high/low/borderline."""
    ctx = _load_context()
    proba = np.asarray(ctx["test_proba"], dtype=float)
    thr = _threshold()

    n_high = int(cfg("profiles", "n_high", default=20))
    n_low = int(cfg("profiles", "n_low", default=20))
    n_border = int(cfg("profiles", "n_borderline", default=10))

    order_desc = list(np.argsort(-proba))
    order_asc = list(np.argsort(proba))
    order_border = list(np.argsort(np.abs(proba - thr)))

    used: set[int] = set()
    picked: list[tuple[str, int]] = []

    def take(order, stratum, count):
        remaining = count
        for idx in order:
            if remaining <= 0:
                break
            idx = int(idx)
            if idx in used:
                continue
            used.add(idx)
            picked.append((stratum, idx))
            remaining -= 1

    take(order_desc, "high", n_high)
    take(order_asc, "low", n_low)
    take(order_border, "borderline", n_border)
    return picked


def _structured_case(case_id: str, card: dict, profile: dict) -> dict:
    prob = float(card.get("prediction_probability", 0.5))
    thr = _threshold()
    near = abs(prob - thr) <= _band()
    return {
        "case_id": case_id,
        "prediction_probability": prob,
        "decision_threshold": thr,
        "risk_label": _risk_label(prob),
        "calibration": {
            "calibration_status": card.get("calibration_status", "unknown"),
            "near_threshold": bool(near),
        },
        "xai_contributors": card.get("top_positive_factors", []),
        "protective_contributors": card.get("top_protective_factors", []),
        "student_profile": {
            "feature_levels": profile.get("feature_levels", {}),
        },
        "model_name": card.get("model_name", "CatBoost"),
    }


def _expected(card: dict, prob: float, stratum: str) -> dict:
    xai_features = [f.get("feature") for f in card.get("top_positive_factors", []) if f.get("feature")]
    protective = [f.get("feature") for f in card.get("top_protective_factors", []) if f.get("feature")]
    return {
        "xai_features": xai_features,
        "protective_features": protective,
        "has_protective": len(protective) > 0,
        "planning_strategy": _planning_strategy(prob),
        "stratum": stratum,
    }


def build_profiles() -> list[dict]:
    picks = select_indices()
    counters = {"high": 0, "low": 0, "borderline": 0}
    records: list[dict] = []
    for stratum, test_index in picks:
        counters[stratum] += 1
        case_id = f"hms_{stratum}_{counters[stratum]:02d}"
        result = predict_from_test_index(test_index, case_id=case_id)
        card = result.card
        profile = result.profile
        prob = float(card.get("prediction_probability", 0.5))
        records.append(
            {
                "case_id": case_id,
                "stratum": stratum,
                "test_index": int(test_index),
                "structured_case": _structured_case(case_id, card, profile),
                "expected": _expected(card, prob, stratum),
            }
        )
    return records


def save_profiles(records: list[dict]) -> Path:
    out = path("profiles_jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def load_profiles() -> list[dict]:
    out = path("profiles_jsonl")
    if not out.exists():
        raise FileNotFoundError(f"Profiles not found: {out}. Run build_profiles.py first.")
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
