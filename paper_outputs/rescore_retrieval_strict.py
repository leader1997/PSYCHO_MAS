"""Rescore pooled retrieval with stricter XAI-linked relevance labels."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MAS = Path(__file__).resolve().parents[1]
OUT = MAS / "paper_outputs" / "query_strategy_ablation"
CASE_IDS = ["hms_low_01", "hms_borderline_01", "hms_high_01"]
FINAL_BUDGET = 5
N_BOOT = 2000
SEED = 42

FEATURE_CATEGORIES = {
    "aca_impa": {"D_academic_impairment"},
    "lonely": {"C_loneliness_isolation"},
    "flourish": {"F_sleep_wellbeing", "A_student_mental_health"},
    "positiveMH": {"F_sleep_wellbeing"},
    "fincur": {"E_financial_food"},
    "finpast": {"E_financial_food"},
    "food_worry": {"E_financial_food"},
    "sleep_wknight": {"F_sleep_wellbeing"},
    "sleep_wkend": {"F_sleep_wellbeing"},
    "belong1": {"A_student_mental_health", "C_loneliness_isolation"},
    "belong2": {"A_student_mental_health", "C_loneliness_isolation"},
    "sub_any": {"B_college_depression_anxiety"},
    "binge1pl": {"B_college_depression_anxiety"},
    "discrim_gender": {"A_student_mental_health"},
    "discrim_race": {"A_student_mental_health"},
    "discrim_sexual": {"A_student_mental_health"},
}
FEATURE_KEYWORDS = {
    "aca_impa": ["academic impairment", "academic performance", "grades"],
    "lonely": ["loneliness", "lonely", "social isolation"],
    "flourish": ["flourishing", "flourish"],
    "positiveMH": ["positive mental health", "flourishing"],
    "fincur": ["financial stress", "financial", "economic hardship"],
    "finpast": ["financial"],
    "food_worry": ["food insecurity", "hunger"],
    "belong1": ["belonging"],
    "belong2": ["belonging"],
    "binge1pl": ["binge", "alcohol"],
    "sub_any": ["substance"],
}


def boot(vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan,) * 4
    rng = np.random.default_rng(SEED)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), sd, float(lo), float(hi)


def dcg(rels):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def targets(sc):
    feats = [x["feature"] for x in (sc.get("xai_contributors") or [])]
    cats, kws = set(), set()
    for f in feats:
        cats |= FEATURE_CATEGORIES.get(f, set())
        for kw in FEATURE_KEYWORDS.get(f, []):
            kws.add(kw.lower())
        kws.add(str(f).replace("_", " ").lower())
    return feats, cats, kws


def is_rel(hit, cats, kws):
    cat = str(hit.get("category_code") or "")
    if cat in cats:
        return True
    text = f"{hit.get('title', '')} {hit.get('text', '')}".lower()
    return any(kw in text for kw in kws if len(kw) >= 4)


def main():
    profiles = {}
    for line in (MAS / "query_agent_eval" / "data" / "case_profiles_50.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        if o["case_id"] in CASE_IDS:
            profiles[o["case_id"]] = o

    # merge xai coverage from previous file
    prev = pd.read_csv(OUT / "retrieval_by_case_arm.csv")

    rows = []
    for case_id in CASE_IDS:
        sc = profiles[case_id]["structured_case"]
        feats, cats, kws = targets(sc)
        for arm in ["direct", "template", "llm_plan"]:
            pooled = json.loads(
                (OUT / "pooled_contexts" / f"{case_id}__{arm}.json").read_text(encoding="utf-8")
            )["pooled"]
            rels = [1 if is_rel(h, cats, kws) else 0 for h in pooled]
            n_rel = sum(rels)
            hit = 1.0 if n_rel > 0 else 0.0
            recall = min(1.0, n_rel / FINAL_BUDGET)
            prec = n_rel / max(len(pooled), 1)
            mrr = 0.0
            for i, r in enumerate(rels, 1):
                if r:
                    mrr = 1.0 / i
                    break
            ideal = sorted(rels, reverse=True)
            ndcg = (dcg(rels) / dcg(ideal)) if sum(ideal) else 0.0
            xc = float(
                prev[(prev.case_id == case_id) & (prev.arm == arm)]["xai_coverage"].iloc[0]
            )
            rows.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "hit_at_k": hit,
                    "precision_at_k": prec,
                    "recall_at_k": recall,
                    "mrr_at_k": mrr,
                    "ndcg_at_k": ndcg,
                    "xai_coverage": xc,
                    "n_relevant_in_pool": n_rel,
                    "n_pooled": len(pooled),
                    "n_unique_docs": len({h.get("doc_id") for h in pooled}),
                }
            )

    strict = pd.DataFrame(rows)
    strict.to_csv(OUT / "retrieval_by_case_arm_strict.csv", index=False)

    sumrows = []
    for arm, g in strict.groupby("arm"):
        row = {"arm": arm, "n_cases": len(g)}
        for m in ["hit_at_k", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "xai_coverage"]:
            mean, sd, lo, hi = boot(g[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_sd"] = sd
            row[f"{m}_ci95_lo"] = lo
            row[f"{m}_ci95_hi"] = hi
        sumrows.append(row)
    sumdf = pd.DataFrame(sumrows)
    sumdf.to_csv(OUT / "retrieval_summary_by_arm_strict.csv", index=False)

    deltas = []
    for arm in ("template", "llm_plan"):
        for m in ["hit_at_k", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "xai_coverage"]:
            wide = strict.pivot(index="case_id", columns="arm", values=m)
            a = wide[arm].values
            b = wide["direct"].values
            d = a - b
            rng = np.random.default_rng(SEED)
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            deltas.append(
                {
                    "level": "retrieval_strict",
                    "metric": m,
                    "contrast": f"{arm}_minus_direct",
                    "delta_mean": float(d.mean()),
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                }
            )
    for m in ["hit_at_k", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "xai_coverage"]:
        wide = strict.pivot(index="case_id", columns="arm", values=m)
        d = wide["llm_plan"].values - wide["template"].values
        rng = np.random.default_rng(SEED)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        deltas.append(
            {
                "level": "retrieval_strict",
                "metric": m,
                "contrast": "llm_plan_minus_template",
                "delta_mean": float(d.mean()),
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
            }
        )
    pd.DataFrame(deltas).to_csv(OUT / "paired_bootstrap_deltas_retrieval_strict.csv", index=False)
    print(strict.to_string(index=False))
    print(sumdf.to_string(index=False))


if __name__ == "__main__":
    main()
