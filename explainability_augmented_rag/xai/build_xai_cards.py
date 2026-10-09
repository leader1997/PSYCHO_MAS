"""Standardize XAI cards from phase-1 outputs (WP2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.config_loader import cfg_get, path  # noqa: E402
from retrieval.build_corpus import FEATURE_DISPLAY  # noqa: E402

WARNING = (
    "This is a model-based screening signal and not a clinical diagnosis."
)


def _display_name(feature: str) -> str:
    return FEATURE_DISPLAY.get(feature, feature.replace("_", " ").title())


def _factor_entry(feature: str, contribution: float, level: str = "") -> dict:
    if contribution >= 0:
        direction = "increases predicted screening risk"
        interpretation = (
            f"This feature contributed positively to the model's risk prediction"
            + (f" (level: {level})." if level else ".")
        )
    else:
        direction = "decreases predicted screening risk"
        interpretation = (
            f"This feature contributed negatively to the model's risk prediction"
            + (f" (level: {level})." if level else ".")
        )
    return {
        "feature": feature,
        "display_name": _display_name(feature),
        "contribution": round(contribution, 4),
        "direction": direction,
        "interpretation": interpretation,
    }


def _label_from_class(predicted_class: str, prob: float) -> str:
    if "elevated" in predicted_class.lower() or prob >= 0.5:
        return "elevated_risk"
    return "not_elevated_risk"


def card_from_local_explanation(card_path: Path, case_id: str) -> dict:
    data = json.loads(card_path.read_text(encoding="utf-8"))
    prob = float(data.get("predicted_probability", 0.5))
    positive = []
    protective = []
    for f in data.get("top_contributing_factors", []):
        shap = float(f.get("shap_value", 0))
        entry = _factor_entry(f["feature"], shap, f.get("level", ""))
        if shap >= 0:
            positive.append(entry)
        else:
            protective.append(entry)
    positive.sort(key=lambda x: -x["contribution"])
    protective.sort(key=lambda x: x["contribution"])
    return {
        "case_id": case_id,
        "prediction_label": _label_from_class(data.get("predicted_class", ""), prob),
        "prediction_probability": round(prob, 4),
        "model_name": data.get("explainer", "CatBoost").replace(" TreeSHAP", ""),
        "calibration_status": "calibrated",
        "top_positive_factors": positive[:5],
        "top_protective_factors": protective[:5],
        "important_warning": WARNING,
    }


def build_xai_cards() -> list[dict]:
    artifacts = path("project_artifacts")
    cards_dir = artifacts / "cards"
    mapping = {
        "high_risk_case": cards_dir / "local_high_risk_case.json",
        "low_risk_case": cards_dir / "local_low_risk_case.json",
        "borderline_risk_case": cards_dir / "local_borderline_case.json",
    }
    cards = []
    for case_id, p in mapping.items():
        if not p.exists():
            print(f"  SKIP missing {p.name}")
            continue
        cards.append(card_from_local_explanation(p, case_id))

    schema = json.loads((ROOT / "xai" / "xai_card_schema.json").read_text(encoding="utf-8"))
    for card in cards:
        jsonschema.validate(card, schema)

    out = path("xai_cards_jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cards)} XAI cards -> {out}")
    return cards


def load_xai_cards() -> list[dict]:
    out = path("xai_cards_jsonl")
    if not out.exists():
        build_xai_cards()
    cards = []
    with open(out, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cards.append(json.loads(line))
    return cards


def main() -> int:
    build_xai_cards()
    return 0


if __name__ == "__main__":
    sys.exit(main())
