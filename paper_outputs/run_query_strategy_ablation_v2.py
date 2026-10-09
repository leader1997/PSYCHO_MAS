"""Strengthened end-to-end query-strategy ablation (v2).

Improvements vs pilot:
  - 20 stratified cases (7 high / 7 low / 6 borderline)
  - Independent deterministic evaluation questions (not Gemma plan text)
  - Template and LLM arms use exactly 6 retrieval queries
  - Same Case/XAI inputs for all arms
  - True RAGAS faithfulness + answer_relevancy (gpt-oss:20b + nomic)
  - Claim support rate
  - Case-level paired bootstrap deltas
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MAS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAS / "report_agent_eval" / "src"))
sys.path.insert(0, str(MAS / "knowledge_db_contextual_eval" / "src"))

from report_agent_eval.answer import ANSWER_SYSTEM, build_user_prompt  # noqa: E402
from report_agent_eval.ollama_client import OllamaError, chat  # noqa: E402
from report_agent_eval.retrieval import ContextualRetriever, load_frozen_contexts  # noqa: E402

OUT = MAS / "paper_outputs" / "query_strategy_ablation_v2"
OUT.mkdir(parents=True, exist_ok=True)

PLAN_MODEL = "gemma4"
REPORT_MODEL = "gemma4"
PER_QUERY_K = 5
FINAL_BUDGET = 5
N_QUERIES = 6
N_BOOT = 2000
SEED = 42
N_HIGH, N_LOW, N_BORDER = 7, 7, 6

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
    "sleep_wknight": ["sleep", "insomnia"],
    "sleep_wkend": ["sleep"],
}


def load_all_profiles() -> list[dict]:
    path = MAS / "query_agent_eval" / "data" / "case_profiles_50.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def select_cases(profiles: list[dict]) -> list[dict]:
    by = {"high": [], "low": [], "borderline": []}
    for p in profiles:
        by[p["stratum"]].append(p)
    rng = np.random.default_rng(SEED)
    picked = []
    for stratum, n in [("high", N_HIGH), ("low", N_LOW), ("borderline", N_BORDER)]:
        idxs = rng.choice(len(by[stratum]), size=n, replace=False)
        for i in sorted(idxs.tolist()):
            picked.append(by[stratum][i])
    # stable order: high, borderline, low by case_id
    order = {"high": 0, "borderline": 1, "low": 2}
    picked.sort(key=lambda p: (order[p["stratum"]], p["case_id"]))
    return picked


def load_plan(case_id: str) -> dict:
    fp = MAS / "query_agent_eval" / "outputs" / "plans" / PLAN_MODEL / f"{case_id}.json"
    return (json.loads(fp.read_text(encoding="utf-8")).get("plan") or {})


def top_xai(sc: dict, n: int = 3) -> list[dict]:
    return list(sc.get("xai_contributors") or [])[:n]


def protective(sc: dict, n: int = 3) -> list[dict]:
    return list(sc.get("protective_contributors") or [])[:n]


def display(x: dict) -> str:
    return x.get("display_name") or x.get("feature") or "factor"


def independent_eval_questions(sc: dict) -> list[dict]:
    """Six fixed objectives, filled from Case/XAI card — independent of LLM planner text."""
    xs = top_xai(sc, 3)
    while len(xs) < 3:
        xs.append({"feature": "general", "display_name": "overall screening contributors"})
    prots = protective(sc, 3)
    prot_names = ", ".join(display(p) for p in prots) if prots else "available protective contributors"
    prob = sc.get("prediction_probability")
    thr = sc.get("decision_threshold", 0.5)
    risk = sc.get("risk_label", "screening_risk").replace("_", " ")
    cal = sc.get("calibration") or {}
    return [
        {
            "qid": "E1",
            "question_type": "screening_interpretation",
            "question": (
                f"How should an expert interpret a non-diagnostic college-student depression/anxiety "
                f"screening probability of {prob} relative to threshold {thr} "
                f"(risk label: {risk}; near_threshold={cal.get('near_threshold')})?"
            ),
        },
        {
            "qid": "E2",
            "question_type": "xai_contributor_1",
            "question": (
                f"What evidence is relevant to understanding how {display(xs[0])} "
                f"({xs[0].get('feature')}) can contribute to an elevated screening-model prediction "
                f"in college students, without treating it as a causal diagnosis?"
            ),
            "linked_feature": xs[0].get("feature"),
        },
        {
            "qid": "E3",
            "question_type": "xai_contributor_2",
            "question": (
                f"What evidence is relevant to understanding how {display(xs[1])} "
                f"({xs[1].get('feature')}) relates to student depression/anxiety screening risk "
                f"as a statistical contributor rather than a cause?"
            ),
            "linked_feature": xs[1].get("feature"),
        },
        {
            "qid": "E4",
            "question_type": "xai_contributor_3",
            "question": (
                f"What evidence is relevant to understanding how {display(xs[2])} "
                f"({xs[2].get('feature')}) may appear among model contributors for student "
                f"mental-health screening support?"
            ),
            "linked_feature": xs[2].get("feature"),
        },
        {
            "qid": "E5",
            "question_type": "protective_factors",
            "question": (
                f"What evidence discusses protective or lower-risk factors relevant to "
                f"{prot_names} in college student mental-health screening contexts?"
            ),
        },
        {
            "qid": "E6",
            "question_type": "limitations_safety_support",
            "question": (
                "What evidence-based guidance exists for communicating non-diagnostic screening "
                "results, stating limitations, and suggesting referral/support pathways for "
                "college students without providing treatment prescriptions?"
            ),
        },
    ]


def arm_direct_queries(sc: dict) -> list[dict]:
    xs = top_xai(sc, 5)
    names = ", ".join(display(x) for x in xs)
    q = (
        f"college student {str(sc.get('risk_label','')).replace('_',' ')} "
        f"screening probability {sc.get('prediction_probability')} "
        f"XAI contributors: {names}; protective factors; screening communication"
    )
    return [{"qid": "D1", "retrieval_query": q, "source": "direct"}]


def arm_template_queries(sc: dict) -> list[dict]:
    """Exactly 6 deterministic templates from the same Case/XAI inputs."""
    xs = top_xai(sc, 3)
    while len(xs) < 3:
        xs.append({"feature": "screening", "display_name": "depression anxiety screening"})
    prots = protective(sc, 1)
    prot_name = display(prots[0]) if prots else "protective factors"
    return [
        {
            "qid": "T1",
            "retrieval_query": (
                "interpreting college student depression anxiety screening probability "
                "threshold non-diagnostic guidelines"
            ),
            "source": "template",
        },
        {
            "qid": "T2",
            "retrieval_query": f"{display(xs[0])} college student depression anxiety screening evidence",
            "source": "template",
            "linked_feature": xs[0].get("feature"),
        },
        {
            "qid": "T3",
            "retrieval_query": f"{display(xs[1])} college student mental health screening evidence",
            "source": "template",
            "linked_feature": xs[1].get("feature"),
        },
        {
            "qid": "T4",
            "retrieval_query": f"{display(xs[2])} student depression anxiety screening contributor evidence",
            "source": "template",
            "linked_feature": xs[2].get("feature"),
        },
        {
            "qid": "T5",
            "retrieval_query": f"{prot_name} protective factors college student mental health screening",
            "source": "template",
        },
        {
            "qid": "T6",
            "retrieval_query": (
                "non-diagnostic screening communication limitations referral college counseling support"
            ),
            "source": "template",
        },
    ]


def arm_llm_queries(plan: dict) -> list[dict]:
    """Exactly 6 retrieval queries from frozen gemma4 plan (truncate/pad)."""
    qplan = list(plan.get("question_plan") or [])
    rows = []
    for q in qplan[:N_QUERIES]:
        rows.append(
            {
                "qid": q.get("question_id"),
                "retrieval_query": q.get("retrieval_query"),
                "source": "llm_plan",
                "linked_feature": q.get("linked_xai_factor"),
            }
        )
    # pad if fewer than 6
    pads = [
        "college student depression anxiety screening interpretation non-diagnostic",
        "protective factors student mental health screening",
        "screening communication referral campus counseling limitations",
    ]
    i = 0
    while len(rows) < N_QUERIES:
        rows.append({"qid": f"P{len(rows)+1}", "retrieval_query": pads[i % len(pads)], "source": "llm_plan_pad"})
        i += 1
    return rows[:N_QUERIES]


def retrieve_queries(retriever: ContextualRetriever, queries: list[dict]) -> list[dict]:
    all_hits = []
    for q in queries:
        rq = (q.get("retrieval_query") or "").strip()
        if not rq:
            continue
        for h in retriever.retrieve(rq):
            row = dict(h)
            row["query_qid"] = q.get("qid")
            row["retrieval_query"] = rq
            all_hits.append(row)
    return all_hits


def pool_hits(hits: list[dict], budget: int = FINAL_BUDGET) -> list[dict]:
    best: dict[str, dict] = {}
    for h in hits:
        cid = h.get("chunk_id") or f"{h.get('doc_id')}:{h.get('rank')}"
        prev = best.get(cid)
        if prev is None or float(h.get("score") or 0) > float(prev.get("score") or 0):
            best[cid] = h
    ranked = sorted(best.values(), key=lambda x: float(x.get("score") or 0), reverse=True)
    out = []
    for i, h in enumerate(ranked[:budget], start=1):
        row = dict(h)
        row["pool_rank"] = i
        out.append(row)
    return out


def relevance_targets(sc: dict):
    feats = [x.get("feature") for x in top_xai(sc, 5)]
    cats, kws = set(), set()
    for f in feats:
        cats |= FEATURE_CATEGORIES.get(f, set())
        for kw in FEATURE_KEYWORDS.get(f, []):
            kws.add(kw.lower())
        kws.add(str(f).replace("_", " ").lower())
    return feats, cats, kws


def is_relevant(hit: dict, cats: set, kws: set) -> bool:
    if str(hit.get("category_code") or "") in cats:
        return True
    text = f"{hit.get('title', '')} {hit.get('text', '')}".lower()
    return any(kw in text for kw in kws if len(kw) >= 4)


def dcg(rels: list[int]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def retrieval_metrics(pooled: list[dict], cats: set, kws: set) -> dict:
    rels = [1 if is_relevant(h, cats, kws) else 0 for h in pooled]
    n_rel = sum(rels)
    hit = 1.0 if n_rel > 0 else 0.0
    prec = n_rel / max(len(pooled), 1)
    recall = min(1.0, n_rel / FINAL_BUDGET)
    mrr = 0.0
    for i, r in enumerate(rels, 1):
        if r:
            mrr = 1.0 / i
            break
    ideal = sorted(rels, reverse=True)
    ndcg = (dcg(rels) / dcg(ideal)) if sum(ideal) else 0.0
    return {
        "hit_at_k": hit,
        "precision_at_k": prec,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "ndcg_at_k": ndcg,
        "n_relevant_in_pool": n_rel,
        "n_pooled": len(pooled),
        "n_unique_docs": len({h.get("doc_id") for h in pooled}),
    }


def xai_coverage(pooled: list[dict], sc: dict) -> float:
    text = " ".join(f"{h.get('title', '')} {h.get('text', '')}" for h in pooled).lower()
    factors = top_xai(sc, 5)
    if not factors:
        return 0.0
    covered = 0
    for x in factors:
        feat = x.get("feature")
        name = (display(x) or "").lower()
        kws = FEATURE_KEYWORDS.get(feat, []) + [name, str(feat).replace("_", " ")]
        if any(kw.lower() in text for kw in kws if kw):
            covered += 1
    return covered / len(factors)


def claim_support_rate(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences with lexical support in retrieved evidence."""
    if not answer or not contexts:
        return 0.0
    ctx = " ".join(contexts).lower()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        toks = [t for t in re.findall(r"[a-zA-Z]{4,}", s.lower())]
        if not toks:
            continue
        if sum(1 for t in toks if t in ctx) / len(toks) >= 0.25:
            ok += 1
    return ok / len(sents)


def generate_answer(question: str, contexts: list[str]) -> tuple[str, str | None]:
    try:
        text = chat(
            REPORT_MODEL,
            ANSWER_SYSTEM,
            build_user_prompt(question, contexts),
            temperature=0.1,
        )
        return (text or "").strip(), None
    except OllamaError as exc:
        return "", str(exc)


def score_ragas_row(question: str, answer: str, contexts: list[str]) -> dict:
    """True RAGAS faithfulness + answer_relevancy via gpt-oss:20b / nomic."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness
    from ragas.run_config import RunConfig
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

    if not answer:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "error": "empty_answer"}

    llm = ChatOllama(model="gpt-oss:20b", base_url="http://localhost:11434", temperature=0)
    emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    ds = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
    )
    try:
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=emb,
            run_config=RunConfig(timeout=1800, max_workers=1, max_retries=2),
        )
        df = result.to_pandas()
        return {
            "faithfulness": float(df["faithfulness"].iloc[0]) if "faithfulness" in df else np.nan,
            "answer_relevancy": float(df["answer_relevancy"].iloc[0]) if "answer_relevancy" in df else np.nan,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"faithfulness": np.nan, "answer_relevancy": np.nan, "error": str(exc)}


def case_level_boot_mean(case_means: np.ndarray):
    vals = np.asarray(case_means, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), sd, float(lo), float(hi)


def paired_case_delta(a_by_case: dict, b_by_case: dict):
    keys = sorted(set(a_by_case) & set(b_by_case))
    if not keys:
        return np.nan, np.nan, np.nan
    d = np.asarray([a_by_case[k] - b_by_case[k] for k in keys], dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main(phase: str = "all") -> None:
    profiles = select_cases(load_all_profiles())
    case_ids = [p["case_id"] for p in profiles]
    (OUT / "selected_cases.json").write_text(
        json.dumps(
            [{"case_id": p["case_id"], "stratum": p["stratum"]} for p in profiles],
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Selected cases:", case_ids, flush=True)

    retriever = ContextualRetriever(top_k=PER_QUERY_K)
    query_rows, retrieval_rows = [], []
    pooled_by = {}

    if phase in ("all", "retrieve", "answers", "score"):
        for p in profiles:
            case_id = p["case_id"]
            sc = p["structured_case"]
            plan = load_plan(case_id)
            arms = {
                "direct": arm_direct_queries(sc),
                "template": arm_template_queries(sc),
                "llm_plan": arm_llm_queries(plan),
            }
            assert len(arms["template"]) == 6 and len(arms["llm_plan"]) == 6
            _, cats, kws = relevance_targets(sc)

            for arm, queries in arms.items():
                for q in queries:
                    query_rows.append(
                        {
                            "case_id": case_id,
                            "arm": arm,
                            "qid": q.get("qid"),
                            "retrieval_query": q.get("retrieval_query"),
                            "source": q.get("source"),
                        }
                    )
                cache = OUT / "pooled_contexts" / f"{case_id}__{arm}.json"
                if cache.exists() and phase != "retrieve":
                    payload = json.loads(cache.read_text(encoding="utf-8"))
                    pooled = payload["pooled"]
                    hits_n = payload.get("n_raw_hits", 0)
                else:
                    print(f"Retrieve {case_id} {arm} nq={len(queries)}", flush=True)
                    if arm == "llm_plan":
                        hits = []
                        for q in queries:
                            frozen = load_frozen_contexts(case_id, q["qid"]) if q.get("qid") else None
                            if frozen and frozen.get("hits") and not str(q.get("qid", "")).startswith("P"):
                                for h in frozen["hits"]:
                                    row = dict(h)
                                    row["query_qid"] = q.get("qid")
                                    hits.append(row)
                            else:
                                hits.extend(retrieve_queries(retriever, [q]))
                    else:
                        hits = retrieve_queries(retriever, queries)
                    pooled = pool_hits(hits, FINAL_BUDGET)
                    hits_n = len(hits)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(
                        json.dumps(
                            {
                                "case_id": case_id,
                                "arm": arm,
                                "n_queries": len(queries),
                                "n_raw_hits": hits_n,
                                "final_budget": FINAL_BUDGET,
                                "pooled": pooled,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                pooled_by[(case_id, arm)] = pooled
                rm = retrieval_metrics(pooled, cats, kws)
                retrieval_rows.append(
                    {
                        "case_id": case_id,
                        "stratum": p["stratum"],
                        "arm": arm,
                        "n_queries": len(queries),
                        **rm,
                        "xai_coverage": xai_coverage(pooled, sc),
                    }
                )

        pd.DataFrame(query_rows).to_csv(OUT / "queries_by_arm.csv", index=False)
        pd.DataFrame(retrieval_rows).to_csv(OUT / "retrieval_by_case_arm.csv", index=False)
        print("Retrieval done", flush=True)

    if phase in ("all", "answers", "score"):
        # ensure pooled_by filled
        if not pooled_by:
            for p in profiles:
                for arm in ("direct", "template", "llm_plan"):
                    payload = json.loads(
                        (OUT / "pooled_contexts" / f"{p['case_id']}__{arm}.json").read_text(encoding="utf-8")
                    )
                    pooled_by[(p["case_id"], arm)] = payload["pooled"]

        answer_rows = []
        for p in profiles:
            case_id = p["case_id"]
            sc = p["structured_case"]
            questions = independent_eval_questions(sc)
            for arm in ("direct", "template", "llm_plan"):
                pooled = pooled_by[(case_id, arm)]
                contexts = [h["text"] for h in pooled if h.get("text")]
                fp = OUT / "answers" / arm / f"{case_id}.json"
                if fp.exists():
                    payload = json.loads(fp.read_text(encoding="utf-8"))
                else:
                    print(f"Answers {case_id} {arm}", flush=True)
                    answers = {}
                    for q in questions:
                        ans, err = generate_answer(q["question"], contexts)
                        answers[q["qid"]] = {
                            **q,
                            "answer": ans,
                            "error": err,
                            "n_contexts": len(contexts),
                        }
                        print(f"  {arm} {case_id} {q['qid']} len={len(ans)}", flush=True)
                    payload = {
                        "case_id": case_id,
                        "arm": arm,
                        "model": REPORT_MODEL,
                        "answers": answers,
                        "contexts": contexts,
                    }
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                for qid, a in (payload.get("answers") or {}).items():
                    answer_rows.append(
                        {
                            "case_id": case_id,
                            "stratum": p["stratum"],
                            "arm": arm,
                            "qid": qid,
                            "question_type": a.get("question_type"),
                            "question": a.get("question"),
                            "answer": a.get("answer"),
                            "n_contexts": a.get("n_contexts"),
                            "claim_support_rate": claim_support_rate(a.get("answer") or "", contexts),
                        }
                    )
        pd.DataFrame(answer_rows).to_csv(OUT / "answers_index.csv", index=False)
        print("Answers indexed", flush=True)

    if phase in ("all", "score"):
        idx = pd.read_csv(OUT / "answers_index.csv")
        score_path = OUT / "ragas_scores.csv"
        done = set()
        rows = []
        if score_path.exists():
            prev = pd.read_csv(score_path)
            rows = prev.to_dict("records")
            done = {(r["case_id"], r["arm"], r["qid"]) for r in rows}

        for _, r in idx.iterrows():
            key = (r["case_id"], r["arm"], r["qid"])
            if key in done:
                continue
            # load contexts from pooled file
            pooled = json.loads(
                (OUT / "pooled_contexts" / f"{r['case_id']}__{r['arm']}.json").read_text(encoding="utf-8")
            )["pooled"]
            contexts = [h["text"] for h in pooled if h.get("text")]
            print(f"RAGAS {r['arm']} {r['case_id']} {r['qid']}", flush=True)
            sc = score_ragas_row(str(r["question"]), str(r["answer"] or ""), contexts)
            rows.append(
                {
                    "case_id": r["case_id"],
                    "stratum": r["stratum"],
                    "arm": r["arm"],
                    "qid": r["qid"],
                    "question_type": r["question_type"],
                    "faithfulness": sc["faithfulness"],
                    "answer_relevancy": sc["answer_relevancy"],
                    "claim_support_rate": r["claim_support_rate"],
                    "ragas_error": sc.get("error"),
                }
            )
            pd.DataFrame(rows).to_csv(score_path, index=False)
            done.add(key)

        scores = pd.read_csv(score_path)
        retrieval = pd.read_csv(OUT / "retrieval_by_case_arm.csv")

        # case-level means for report metrics
        case_rep = (
            scores.groupby(["case_id", "arm"], as_index=False)[
                ["faithfulness", "answer_relevancy", "claim_support_rate"]
            ].mean()
        )
        case_rep.to_csv(OUT / "report_by_case_arm.csv", index=False)

        # summaries
        def summarize(df, metrics, key="arm"):
            out = []
            for arm, g in df.groupby(key):
                row = {key: arm, "n": len(g)}
                for m in metrics:
                    # case-level bootstrap if case_id present
                    if "case_id" in g.columns:
                        cm = g.groupby("case_id")[m].mean().values
                        mean, sd, lo, hi = case_level_boot_mean(cm)
                    else:
                        mean, sd, lo, hi = case_level_boot_mean(g[m].values)
                    row[f"{m}_mean"] = mean
                    row[f"{m}_sd"] = sd
                    row[f"{m}_ci95_lo"] = lo
                    row[f"{m}_ci95_hi"] = hi
                out.append(row)
            return pd.DataFrame(out)

        ret_sum = summarize(
            retrieval,
            ["hit_at_k", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "xai_coverage"],
        )
        rep_sum = summarize(case_rep, ["faithfulness", "answer_relevancy", "claim_support_rate"])
        ret_sum.to_csv(OUT / "retrieval_summary_by_arm.csv", index=False)
        rep_sum.to_csv(OUT / "report_summary_by_arm.csv", index=False)

        # paired deltas case-level
        deltas = []
        for metric_df, level, metrics in [
            (retrieval, "retrieval", ["hit_at_k", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "xai_coverage"]),
            (case_rep, "report", ["faithfulness", "answer_relevancy", "claim_support_rate"]),
        ]:
            for arm in ("template", "llm_plan"):
                for m in metrics:
                    a = metric_df[metric_df.arm == arm].set_index("case_id")[m].to_dict()
                    b = metric_df[metric_df.arm == "direct"].set_index("case_id")[m].to_dict()
                    dmean, lo, hi = paired_case_delta(a, b)
                    deltas.append(
                        {
                            "level": level,
                            "metric": m,
                            "contrast": f"{arm}_minus_direct",
                            "delta_mean": dmean,
                            "ci95_lo": lo,
                            "ci95_hi": hi,
                        }
                    )
            for m in metrics:
                a = metric_df[metric_df.arm == "llm_plan"].set_index("case_id")[m].to_dict()
                b = metric_df[metric_df.arm == "template"].set_index("case_id")[m].to_dict()
                dmean, lo, hi = paired_case_delta(a, b)
                deltas.append(
                    {
                        "level": level,
                        "metric": m,
                        "contrast": "llm_plan_minus_template",
                        "delta_mean": dmean,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                    }
                )
        pd.DataFrame(deltas).to_csv(OUT / "paired_case_bootstrap_deltas.csv", index=False)

        # Table X wide
        arms = ["direct", "template", "llm_plan"]
        table = {"metric": []}
        for arm in arms:
            table[arm] = []
        table["delta_llm_direct"] = []
        table["delta_llm_template"] = []
        table["ci_llm_direct"] = []
        table["ci_llm_template"] = []

        def fmt(mean, lo, hi):
            if pd.isna(mean):
                return ""
            return f"{mean:.3f} ({lo:.3f}–{hi:.3f})"

        metric_map = [
            ("retrieval", "recall_at_k", "Recall@5"),
            ("retrieval", "ndcg_at_k", "nDCG@5"),
            ("retrieval", "mrr_at_k", "MRR@5"),
            ("retrieval", "hit_at_k", "Hit@5"),
            ("retrieval", "xai_coverage", "XAI evidence coverage"),
            ("report", "faithfulness", "RAGAS faithfulness"),
            ("report", "answer_relevancy", "Answer relevancy"),
            ("report", "claim_support_rate", "Claim support rate"),
        ]
        sum_lookup = {
            "retrieval": ret_sum.set_index("arm"),
            "report": rep_sum.set_index("arm"),
        }
        delta_df = pd.DataFrame(deltas)
        for level, key, label in metric_map:
            table["metric"].append(label)
            s = sum_lookup[level]
            for arm in arms:
                table[arm].append(
                    fmt(s.loc[arm, f"{key}_mean"], s.loc[arm, f"{key}_ci95_lo"], s.loc[arm, f"{key}_ci95_hi"])
                )
            d1 = delta_df[(delta_df.level == level) & (delta_df.metric == key) & (delta_df.contrast == "llm_plan_minus_direct")].iloc[0]
            d2 = delta_df[(delta_df.level == level) & (delta_df.metric == key) & (delta_df.contrast == "llm_plan_minus_template")].iloc[0]
            table["delta_llm_direct"].append(f"{d1['delta_mean']:+.3f}")
            table["delta_llm_template"].append(f"{d2['delta_mean']:+.3f}")
            table["ci_llm_direct"].append(f"[{d1['ci95_lo']:+.3f}, {d1['ci95_hi']:+.3f}]")
            table["ci_llm_template"].append(f"[{d2['ci95_lo']:+.3f}, {d2['ci95_hi']:+.3f}]")
        table_df = pd.DataFrame(table)
        table_df.to_csv(OUT / "table_x_end_to_end_ablation.csv", index=False)
        print(table_df.to_string(index=False), flush=True)
        print("Scoring complete", flush=True)


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(ph)
