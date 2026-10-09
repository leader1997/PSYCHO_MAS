"""Build expert_evaluation_app case bank from real HMS eval profiles."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "query_agent_eval" / "data" / "case_profiles_50.jsonl"
CTX = ROOT / "report_agent_eval" / "outputs" / "contexts"
ANS_DIRS = [
    ROOT / "report_agent_eval" / "outputs" / "answers" / "gemma4",
    ROOT / "report_agent_eval" / "outputs" / "answers" / "gpt-oss_20b",
]
OUT = ROOT / "expert_evaluation_app" / "src" / "data" / "cases.json"
CAI_SRC = ROOT / "controlled_agentic_xai_rag" / "src"

SELECTED = [
    "hms_high_01",
    "hms_high_02",
    "hms_high_03",
    "hms_borderline_01",
    "hms_borderline_02",
    "hms_borderline_03",
    "hms_low_01",
    "hms_low_02",
    "hms_low_03",
]

FEATURE_LABELS = {
    "age": "Age",
    "gender_male": "Gender (male)",
    "gender_female": "Gender (female)",
    "international": "International student",
    "fincur": "Current financial situation",
    "finpast": "Past financial stress",
    "food_worry": "Food insecurity worry",
    "sleep_wknight": "Weeknight sleep",
    "sleep_wkend": "Weekend sleep",
    "exerc": "Exercise",
    "belong1": "Belonging (1)",
    "belong2": "Belonging (2)",
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

SALIENT_ORDER = [
    "aca_impa",
    "flourish",
    "lonely",
    "fincur",
    "food_worry",
    "belong1",
    "belong2",
    "positiveMH",
    "sleep_wknight",
    "sleep_wkend",
    "finpast",
    "age",
    "international",
]

LABEL_MAP = {
    "elevated_screening_risk": "Above threshold",
    "borderline_screening_risk": "Near threshold (borderline)",
    "low_screening_risk": "Below threshold",
}


def load_raw_values(case_id: str, test_index: int | None) -> dict[str, float | None]:
    """Pull original HMS feature values for this test-row via the predictor."""
    if test_index is None:
        return {}
    import sys

    if str(CAI_SRC) not in sys.path:
        sys.path.insert(0, str(CAI_SRC))
    from controlled_agentic_xai_rag.prediction.hms_predictor import (  # noqa: E402
        predict_from_test_index,
    )

    result = predict_from_test_index(int(test_index), case_id=case_id)
    return dict((result.profile or {}).get("raw_values") or {})


def format_value(v) -> str | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv == int(fv):
        return str(int(fv))
    return f"{fv:.4g}"


def load_answers(case_id: str):
    for d in ANS_DIRS:
        p = d / f"{case_id}.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), d.name
    return None, None


def load_evidence(case_id: str, top_n: int = 5):
    folder = CTX / case_id
    if not folder.is_dir():
        return []
    best: OrderedDict = OrderedDict()
    for qpath in sorted(folder.glob("Q*.json")):
        data = json.loads(qpath.read_text(encoding="utf-8"))
        for hit in data.get("hits") or []:
            title = (hit.get("title") or "").strip() or hit.get("doc_id") or "Untitled"
            doc_id = hit.get("doc_id") or ""
            score = float(hit.get("score") or 0)
            key = (doc_id, title)
            if key not in best or score > best[key]["score"]:
                best[key] = {
                    "title": title,
                    "doc_id": doc_id,
                    "score": round(score, 4),
                }
    items = sorted(best.values(), key=lambda x: -x["score"])
    return items[:top_n]


def report_from_answers(ans_data) -> str:
    answers = ans_data.get("answers") or {}
    parts = []
    for qid in sorted(answers.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99):
        text = (answers[qid].get("answer") or "").strip()
        if not text:
            continue
        if len(text) > 420:
            text = text[:417].rsplit(" ", 1)[0] + "…"
        parts.append(f"{qid}: {text}")
    return "\n\n".join(parts)


def report_fallback(sc, evidence) -> str:
    p = sc["prediction_probability"]
    thr = sc["decision_threshold"]
    label = sc["risk_label"]
    xai = sc.get("xai_contributors") or []
    tops = ", ".join(
        f"{x['display_name']} ({x['contribution']:+.2f})" for x in xai[:4]
    )
    ev = "; ".join(e["title"][:80] for e in evidence[:3]) or "retrieved literature passages"
    return (
        f"Screening probability {p:.4f} vs threshold {thr:.2f} "
        f"({label.replace('_', ' ')}). "
        f"Leading statistical contributors include {tops}. "
        f"Retrieved evidence titles include: {ev}. "
        f"This output is for expert review of a non-diagnostic screening "
        f"decision-support pipeline; it is not a clinical diagnosis."
    )


def risk_label_ui(sc) -> str:
    rl = sc.get("risk_label") or ""
    if rl in LABEL_MAP:
        return LABEL_MAP[rl]
    p = sc["prediction_probability"]
    thr = sc["decision_threshold"]
    if abs(p - thr) <= 0.05:
        return "Near threshold (borderline)"
    return "Above threshold" if p >= thr else "Below threshold"


def main() -> None:
    rows = [
        json.loads(L)
        for L in PROFILES.read_text(encoding="utf-8").splitlines()
        if L.strip()
    ]
    by_id = {r["case_id"]: r for r in rows}
    cases = []
    for i, cid in enumerate(SELECTED, start=1):
        row = by_id[cid]
        sc = row["structured_case"]
        levels = (sc.get("student_profile") or {}).get("feature_levels") or {}
        raw_values = load_raw_values(cid, row.get("test_index"))
        # Keep full HMS feature set for the expert dropdown (numeric raw values).
        ordered_feats = list(FEATURE_LABELS.keys()) + [
            f for f in levels.keys() if f not in FEATURE_LABELS
        ]
        profile_rows = []
        for feat in ordered_feats:
            if feat not in levels and feat not in raw_values:
                continue
            raw = raw_values.get(feat)
            profile_rows.append(
                {
                    "feature": feat,
                    "display_name": FEATURE_LABELS.get(feat, feat),
                    "value": raw,
                    "value_display": format_value(raw) if raw is not None else "missing",
                }
            )

        xai = []
        for x in (sc.get("xai_contributors") or [])[:5]:
            xai.append(
                {
                    "feature": x["feature"],
                    "display_name": x.get("display_name")
                    or FEATURE_LABELS.get(x["feature"], x["feature"]),
                    "shap": round(float(x["contribution"]), 4),
                }
            )
        for x in (sc.get("protective_contributors") or [])[:2]:
            if len(xai) >= 6:
                break
            if any(t["feature"] == x["feature"] for t in xai):
                continue
            xai.append(
                {
                    "feature": x["feature"],
                    "display_name": x.get("display_name")
                    or FEATURE_LABELS.get(x["feature"], x["feature"]),
                    "shap": round(float(x["contribution"]), 4),
                }
            )

        evidence = load_evidence(cid)
        ans_data, ans_model = load_answers(cid)
        if ans_data:
            report = report_from_answers(ans_data)
            report_source = f"report_agent_answers:{ans_model}"
        else:
            report = report_fallback(sc, evidence)
            report_source = "profile_evidence_fallback"

        cases.append(
            {
                "id": f"HMS_{i:03d}",
                "source_case_id": cid,
                "stratum": row["stratum"],
                "test_index": row.get("test_index"),
                "student_profile": profile_rows,
                "screening": {
                    "probability": round(float(sc["prediction_probability"]), 4),
                    "threshold": float(sc["decision_threshold"]),
                    "label": risk_label_ui(sc),
                    "risk_label": sc.get("risk_label"),
                    "disclaimer": "This is a screening result, not a diagnosis.",
                    "model_name": sc.get("model_name") or "CatBoost",
                },
                "xai": xai,
                "evidence": evidence,
                "report_summary": report,
                "report_source": report_source,
            }
        )

    OUT.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} n={len(cases)}")
    for c in cases:
        print(
            c["id"],
            c["source_case_id"],
            c["stratum"],
            c["report_source"],
            "ev",
            len(c["evidence"]),
        )


if __name__ == "__main__":
    main()
