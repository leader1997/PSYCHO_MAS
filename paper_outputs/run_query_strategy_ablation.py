"""End-to-end query-strategy ablation (direct vs template vs LLM Query Planning).

Holds constant: knowledge base, Ollama-context hybrid retriever, embeddings,
per-query K, final evidence budget, Report Agent model/prompt/temperature,
evaluation cases, and evaluation questions (frozen gemma4 plan questions).

Varies ONLY the query-generation strategy used to retrieve evidence.
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

OUT = MAS / "paper_outputs" / "query_strategy_ablation"
OUT.mkdir(parents=True, exist_ok=True)

CASE_IDS = ["hms_low_01", "hms_borderline_01", "hms_high_01"]
PLAN_MODEL = "gemma4"
REPORT_MODEL = "gemma4"
PER_QUERY_K = 5
FINAL_BUDGET = 5
N_BOOT = 2000
SEED = 42

# Heuristic mapping: XAI feature -> preferred knowledge-base category codes
FEATURE_CATEGORIES = {
    "aca_impa": {"D_academic_impairment", "B_college_depression_anxiety", "A_student_mental_health"},
    "lonely": {"C_loneliness_isolation", "A_student_mental_health"},
    "flourish": {"F_sleep_wellbeing", "A_student_mental_health", "G_screening_communication"},
    "positiveMH": {"F_sleep_wellbeing", "A_student_mental_health"},
    "fincur": {"E_financial_food", "A_student_mental_health"},
    "finpast": {"E_financial_food"},
    "food_worry": {"E_financial_food"},
    "sleep_wknight": {"F_sleep_wellbeing"},
    "sleep_wkend": {"F_sleep_wellbeing"},
    "belong1": {"A_student_mental_health", "C_loneliness_isolation"},
    "belong2": {"A_student_mental_health", "C_loneliness_isolation"},
    "sub_any": {"A_student_mental_health", "B_college_depression_anxiety"},
    "binge1pl": {"A_student_mental_health"},
    "discrim_race": {"A_student_mental_health"},
    "discrim_gender": {"A_student_mental_health"},
    "discrim_sexual": {"A_student_mental_health"},
    "age": {"A_student_mental_health"},
    "inst_type": {"A_student_mental_health"},
    "inst_size": {"A_student_mental_health"},
    "inst_public": {"A_student_mental_health"},
}

FEATURE_KEYWORDS = {
    "aca_impa": ["academic impairment", "academic", "grades", "coursework"],
    "lonely": ["loneliness", "lonely", "isolation", "social isolation"],
    "flourish": ["flourishing", "flourish", "well-being", "wellbeing"],
    "positiveMH": ["positive mental health", "flourishing"],
    "fincur": ["financial", "money", "economic"],
    "finpast": ["financial", "past financial"],
    "food_worry": ["food insecurity", "food worry", "hunger"],
    "sleep_wknight": ["sleep", "insomnia"],
    "sleep_wkend": ["sleep"],
    "belong1": ["belonging", "campus belonging"],
    "belong2": ["belonging"],
    "sub_any": ["substance", "alcohol", "drug"],
    "discrim_gender": ["discrimination", "gender"],
    "discrim_race": ["discrimination", "race"],
    "discrim_sexual": ["discrimination", "sexual"],
}


def load_profiles() -> dict[str, dict]:
    path = MAS / "query_agent_eval" / "data" / "case_profiles_50.jsonl"
    out = {}
    for line in path.open(encoding="utf-8"):
        o = json.loads(line)
        if o.get("case_id") in CASE_IDS:
            out[o["case_id"]] = o
    return out


def load_plan(case_id: str) -> dict:
    fp = MAS / "query_agent_eval" / "outputs" / "plans" / PLAN_MODEL / f"{case_id}.json"
    wrap = json.loads(fp.read_text(encoding="utf-8"))
    return wrap.get("plan") or {}


def eval_questions(plan: dict) -> list[dict]:
    rows = []
    for q in plan.get("question_plan") or []:
        rows.append(
            {
                "qid": q.get("question_id"),
                "question": q.get("question"),
                "question_type": q.get("question_type"),
                "retrieval_query": q.get("retrieval_query"),
                "linked_xai_factor": q.get("linked_xai_factor"),
            }
        )
    return rows


def arm_direct_queries(sc: dict) -> list[dict]:
    """Single undecomposed query from Case/XAI card."""
    xai = sc.get("xai_contributors") or []
    names = [x.get("display_name") or x.get("feature") for x in xai[:5]]
    risk = sc.get("risk_label", "screening risk")
    prob = sc.get("prediction_probability")
    q = (
        f"college student {risk.replace('_', ' ')} screening probability {prob} "
        f"contributing factors: " + ", ".join(names)
    )
    return [{"qid": "D1", "retrieval_query": q, "source": "direct_case_xai_card"}]


def arm_template_queries(sc: dict) -> list[dict]:
    """Deterministic templates from XAI + protective factors."""
    rows = []
    rows.append(
        {
            "qid": "T_risk",
            "retrieval_query": (
                "interpreting elevated depression anxiety screening risk "
                "college students non-diagnostic guidelines"
            ),
            "source": "template_risk",
        }
    )
    for i, x in enumerate(sc.get("xai_contributors") or [], start=1):
        name = x.get("display_name") or x.get("feature")
        feat = x.get("feature")
        rows.append(
            {
                "qid": f"T_xai_{i}",
                "retrieval_query": (
                    f"{name} and college student depression anxiety screening "
                    f"evidence ({feat})"
                ),
                "source": f"template_xai:{feat}",
                "linked_feature": feat,
            }
        )
    for i, x in enumerate((sc.get("protective_contributors") or [])[:3], start=1):
        name = x.get("display_name") or x.get("feature")
        feat = x.get("feature")
        rows.append(
            {
                "qid": f"T_prot_{i}",
                "retrieval_query": (
                    f"{name} protective factors student mental health screening"
                ),
                "source": f"template_protective:{feat}",
                "linked_feature": feat,
            }
        )
    rows.append(
        {
            "qid": "T_safety",
            "retrieval_query": (
                "non-diagnostic screening communication referral college counseling"
            ),
            "source": "template_safety",
        }
    )
    return rows


def arm_llm_queries(plan: dict) -> list[dict]:
    rows = []
    for q in plan.get("question_plan") or []:
        rows.append(
            {
                "qid": q.get("question_id"),
                "retrieval_query": q.get("retrieval_query"),
                "source": "llm_query_planning_agent",
                "linked_feature": q.get("linked_xai_factor"),
                "question": q.get("question"),
            }
        )
    return rows


def case_relevance_targets(sc: dict) -> dict:
    features = []
    for x in (sc.get("xai_contributors") or []) + (sc.get("protective_contributors") or [])[:3]:
        features.append(x.get("feature"))
    cats = set()
    kws = set()
    for f in features:
        cats |= FEATURE_CATEGORIES.get(f, set())
        for kw in FEATURE_KEYWORDS.get(f, []):
            kws.add(kw.lower())
        # always add display-ish tokens
        kws.add(str(f).replace("_", " ").lower())
    # always include core screening categories
    cats |= {"A_student_mental_health", "B_college_depression_anxiety", "G_screening_communication"}
    kws |= {"depression", "anxiety", "screening", "college student"}
    return {"features": features, "categories": cats, "keywords": kws}


def is_relevant(hit: dict, targets: dict) -> bool:
    cat = str(hit.get("category_code") or "")
    if cat in targets["categories"]:
        return True
    text = f"{hit.get('title', '')} {hit.get('text', '')}".lower()
    return any(kw in text for kw in targets["keywords"] if len(kw) >= 4)


def retrieve_queries(retriever: ContextualRetriever, queries: list[dict]) -> list[dict]:
    all_hits = []
    for q in queries:
        rq = q.get("retrieval_query") or ""
        if not rq.strip():
            continue
        hits = retriever.retrieve(rq)
        for h in hits:
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
    pooled = []
    for i, h in enumerate(ranked[:budget], start=1):
        row = dict(h)
        row["pool_rank"] = i
        pooled.append(row)
    return pooled


def dcg(rels: list[int]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def retrieval_metrics(pooled: list[dict], targets: dict) -> dict:
    rels = [1 if is_relevant(h, targets) else 0 for h in pooled]
    n_rel = sum(rels)
    hit = 1.0 if n_rel > 0 else 0.0
    # Approximate recall against a soft relevant set size = min(budget, 5)
    denom = max(min(FINAL_BUDGET, 5), 1)
    recall = n_rel / denom
    mrr = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            mrr = 1.0 / i
            break
    ideal = sorted(rels, reverse=True)
    ndcg = (dcg(rels) / dcg(ideal)) if sum(ideal) else 0.0
    return {
        "hit_at_k": hit,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "ndcg_at_k": ndcg,
        "n_relevant_in_pool": n_rel,
        "n_pooled": len(pooled),
        "n_unique_docs": len({h.get("doc_id") for h in pooled}),
    }


def xai_evidence_coverage(pooled: list[dict], sc: dict) -> dict:
    text = " ".join(f"{h.get('title', '')} {h.get('text', '')}" for h in pooled).lower()
    factors = sc.get("xai_contributors") or []
    covered = []
    missing = []
    for x in factors:
        feat = x.get("feature")
        name = (x.get("display_name") or feat or "").lower()
        kws = FEATURE_KEYWORDS.get(feat, []) + [name, str(feat).replace("_", " ")]
        ok = any(kw.lower() in text for kw in kws if kw)
        (covered if ok else missing).append(feat)
    frac = len(covered) / max(len(factors), 1)
    return {
        "xai_coverage": frac,
        "xai_covered": covered,
        "xai_missing": missing,
        "n_xai_factors": len(factors),
    }


def generate_answers_for_arm(
    case_id: str,
    arm: str,
    questions: list[dict],
    pooled: list[dict],
) -> dict:
    contexts = [h["text"] for h in pooled if h.get("text")]
    answers = {}
    for q in questions:
        qid = q["qid"]
        question = q["question"]
        err = None
        try:
            user = build_user_prompt(question, contexts)
            text = chat(REPORT_MODEL, ANSWER_SYSTEM, user, temperature=0.1)
            ans = (text or "").strip()
        except OllamaError as exc:
            ans, err = "", str(exc)
        answers[qid] = {
            "qid": qid,
            "question": question,
            "question_type": q.get("question_type"),
            "answer": ans,
            "error": err,
            "n_contexts": len(contexts),
            "arm": arm,
            "case_id": case_id,
        }
        print(f"    answer {arm} {case_id} {qid} chars={len(ans or '')}", flush=True)
    return {
        "model": REPORT_MODEL,
        "case_id": case_id,
        "arm": arm,
        "answers": answers,
        "n_contexts": len(contexts),
    }


def load_or_gen_answers(case_id: str, arm: str, questions, pooled) -> dict:
    fp = OUT / "answers" / arm / f"{case_id}.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    payload = generate_answers_for_arm(case_id, arm, questions, pooled)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def embed_texts(texts: list[str], model: str = "nomic-embed-text") -> np.ndarray:
    import requests

    vecs = []
    base = "http://localhost:11434"
    for t in texts:
        r = requests.post(
            f"{base}/api/embeddings",
            json={"model": model, "prompt": t[:8000]},
            timeout=120,
        )
        r.raise_for_status()
        vecs.append(np.asarray(r.json()["embedding"], dtype=np.float32))
        time.sleep(0.05)
    return np.vstack(vecs)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_answer_local(question: str, answer: str, contexts: list[str]) -> dict:
    """Local proxies aligned to RAGAS intentions (no gpt-oss judge dependency).

    - answer_relevancy ~ embedding cosine(question, answer)
    - semantic_similarity reserved if reference available
    - faithfulness ~ fraction of answer sentences with lexical overlap to contexts
    - answer_correctness left NaN unless reference provided
    """
    if not answer:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "answer_correctness": np.nan,
            "semantic_similarity": np.nan,
        }
    # relevancy via embeddings
    try:
        qv = embed_texts([question])[0]
        av = embed_texts([answer])[0]
        relevancy = max(0.0, min(1.0, (cos(qv, av) + 1) / 2))  # map [-1,1] -> [0,1] soft
        # nomic cosine usually already ~0.5-0.9; use raw clipped cosine in [0,1]
        relevancy = max(0.0, min(1.0, cos(qv, av)))
    except Exception:
        relevancy = float(len(set(question.lower().split()) & set(answer.lower().split()))
                          / max(len(set(question.lower().split())), 1))

    ctx = " ".join(contexts).lower()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sents:
        faith = 0.0
    else:
        ok = 0
        for s in sents:
            toks = [t for t in re.findall(r"[a-zA-Z]{4,}", s.lower())]
            if not toks:
                continue
            overlap = sum(1 for t in toks if t in ctx)
            if overlap / max(len(toks), 1) >= 0.25:
                ok += 1
        faith = ok / len(sents)
    return {
        "faithfulness": float(faith),
        "answer_relevancy": float(relevancy),
        "answer_correctness": np.nan,
        "semantic_similarity": np.nan,
    }


def maybe_semantic_vs_reference(answer: str, reference: str) -> float:
    if not answer or not reference:
        return float("nan")
    try:
        av, rv = embed_texts([answer, reference])
        return max(0.0, min(1.0, cos(av, rv)))
    except Exception:
        return float("nan")


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        boots.append(values[rng.integers(0, n, n)].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    return float(values.mean()), sd, float(lo), float(hi)


def paired_delta_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    d = a - b
    rng = np.random.default_rng(seed)
    boots = []
    n = len(d)
    for _ in range(n_boot):
        boots.append(d[rng.integers(0, n, n)].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main() -> None:
    profiles = load_profiles()
    print("cases", list(profiles), flush=True)
    retriever = ContextualRetriever(top_k=PER_QUERY_K)

    retrieval_rows = []
    query_rows = []
    answer_score_rows = []
    pooled_by = {}  # (case, arm) -> pooled hits

    for case_id in CASE_IDS:
        profile = profiles[case_id]
        sc = profile["structured_case"]
        plan = load_plan(case_id)
        questions = eval_questions(plan)
        targets = case_relevance_targets(sc)

        arms = {
            "direct": arm_direct_queries(sc),
            "template": arm_template_queries(sc),
            "llm_plan": arm_llm_queries(plan),
        }

        # Prefer reusing frozen LLM-plan per-question contexts when available,
        # then re-pool to FINAL_BUDGET for fairness with other arms.
        for arm, queries in arms.items():
            print(f"\n=== {case_id} / {arm} queries={len(queries)} ===", flush=True)
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

            if arm == "llm_plan":
                hits = []
                for q in queries:
                    frozen = load_frozen_contexts(case_id, q["qid"])
                    if frozen and frozen.get("hits"):
                        for h in frozen["hits"]:
                            row = dict(h)
                            row["query_qid"] = q["qid"]
                            row["retrieval_query"] = q.get("retrieval_query")
                            hits.append(row)
                    else:
                        hits.extend(retrieve_queries(retriever, [q]))
            else:
                hits = retrieve_queries(retriever, queries)

            pooled = pool_hits(hits, FINAL_BUDGET)
            pooled_by[(case_id, arm)] = pooled
            (OUT / "pooled_contexts").mkdir(parents=True, exist_ok=True)
            (OUT / "pooled_contexts" / f"{case_id}__{arm}.json").write_text(
                json.dumps(
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "n_queries": len(queries),
                        "n_raw_hits": len(hits),
                        "final_budget": FINAL_BUDGET,
                        "pooled": pooled,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            rm = retrieval_metrics(pooled, targets)
            xc = xai_evidence_coverage(pooled, sc)
            retrieval_rows.append(
                {
                    "case_id": case_id,
                    "stratum": profile.get("stratum"),
                    "arm": arm,
                    "n_queries": len(queries),
                    **rm,
                    **{k: v for k, v in xc.items() if k != "xai_covered" and k != "xai_missing"},
                    "xai_covered": "|".join(xc["xai_covered"]),
                    "xai_missing": "|".join(xc["xai_missing"]),
                }
            )
            print(
                f"  retrieval hit={rm['hit_at_k']} recall={rm['recall_at_k']:.3f} "
                f"mrr={rm['mrr_at_k']:.3f} ndcg={rm['ndcg_at_k']:.3f} "
                f"xai_cov={xc['xai_coverage']:.3f}",
                flush=True,
            )

    # Answers + local report metrics
    for case_id in CASE_IDS:
        plan = load_plan(case_id)
        questions = eval_questions(plan)
        # optional references from gpt-oss for semantic similarity only
        ref_path = MAS / "report_agent_eval" / "outputs" / "answers" / "gpt-oss_20b" / f"{case_id}.json"
        ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {"answers": {}}
        for arm in ("direct", "template", "llm_plan"):
            pooled = pooled_by[(case_id, arm)]
            contexts = [h["text"] for h in pooled if h.get("text")]
            print(f"\n### Answers {case_id} {arm}", flush=True)
            payload = load_or_gen_answers(case_id, arm, questions, pooled)
            for q in questions:
                qid = q["qid"]
                ans = (payload.get("answers") or {}).get(qid) or {}
                answer = ans.get("answer") or ""
                scores = score_answer_local(q["question"], answer, contexts)
                ref_ans = ((ref.get("answers") or {}).get(qid) or {}).get("answer") or ""
                # semantic similarity vs fixed gpt-oss reference (disclose: ref produced under llm_plan contexts)
                scores["semantic_similarity"] = maybe_semantic_vs_reference(answer, ref_ans)
                answer_score_rows.append(
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "qid": qid,
                        "question_type": q.get("question_type"),
                        **scores,
                        "answer_len": len(answer),
                    }
                )

    ret_df = pd.DataFrame(retrieval_rows)
    q_df = pd.DataFrame(query_rows)
    a_df = pd.DataFrame(answer_score_rows)
    ret_df.to_csv(OUT / "retrieval_by_case_arm.csv", index=False)
    q_df.to_csv(OUT / "queries_by_arm.csv", index=False)
    a_df.to_csv(OUT / "report_scores_by_question.csv", index=False)

    # Summaries + bootstrap
    ret_summary = []
    for arm, g in ret_df.groupby("arm"):
        row = {"arm": arm, "n_cases": len(g)}
        for m in ["hit_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "xai_coverage"]:
            mean, sd, lo, hi = bootstrap_mean_ci(g[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_sd"] = sd
            row[f"{m}_ci95_lo"] = lo
            row[f"{m}_ci95_hi"] = hi
        ret_summary.append(row)
    ret_sum_df = pd.DataFrame(ret_summary)
    ret_sum_df.to_csv(OUT / "retrieval_summary_by_arm.csv", index=False)

    ans_summary = []
    for arm, g in a_df.groupby("arm"):
        row = {"arm": arm, "n_questions": len(g)}
        for m in ["faithfulness", "answer_relevancy", "semantic_similarity"]:
            mean, sd, lo, hi = bootstrap_mean_ci(g[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_sd"] = sd
            row[f"{m}_ci95_lo"] = lo
            row[f"{m}_ci95_hi"] = hi
        ans_summary.append(row)
    ans_sum_df = pd.DataFrame(ans_summary)
    ans_sum_df.to_csv(OUT / "report_summary_by_arm.csv", index=False)

    # Paired deltas vs direct (case-level for retrieval; question-level for report)
    deltas = []
    for arm in ("template", "llm_plan"):
        for m in ["hit_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "xai_coverage"]:
            wide = ret_df.pivot(index="case_id", columns="arm", values=m)
            if "direct" in wide.columns and arm in wide.columns:
                dmean, lo, hi = paired_delta_ci(wide[arm].values, wide["direct"].values)
                deltas.append(
                    {
                        "level": "retrieval",
                        "metric": m,
                        "contrast": f"{arm}_minus_direct",
                        "delta_mean": dmean,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                    }
                )
        for m in ["faithfulness", "answer_relevancy", "semantic_similarity"]:
            # align by case_id+qid
            a = a_df[a_df.arm == arm].set_index(["case_id", "qid"])[m]
            b = a_df[a_df.arm == "direct"].set_index(["case_id", "qid"])[m]
            idx = a.index.intersection(b.index)
            dmean, lo, hi = paired_delta_ci(a.loc[idx].values, b.loc[idx].values)
            deltas.append(
                {
                    "level": "report",
                    "metric": m,
                    "contrast": f"{arm}_minus_direct",
                    "delta_mean": dmean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    # also llm_plan vs template
    for m in ["hit_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "xai_coverage"]:
        wide = ret_df.pivot(index="case_id", columns="arm", values=m)
        dmean, lo, hi = paired_delta_ci(wide["llm_plan"].values, wide["template"].values)
        deltas.append(
            {
                "level": "retrieval",
                "metric": m,
                "contrast": "llm_plan_minus_template",
                "delta_mean": dmean,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        )
    for m in ["faithfulness", "answer_relevancy", "semantic_similarity"]:
        a = a_df[a_df.arm == "llm_plan"].set_index(["case_id", "qid"])[m]
        b = a_df[a_df.arm == "template"].set_index(["case_id", "qid"])[m]
        idx = a.index.intersection(b.index)
        dmean, lo, hi = paired_delta_ci(a.loc[idx].values, b.loc[idx].values)
        deltas.append(
            {
                "level": "report",
                "metric": m,
                "contrast": "llm_plan_minus_template",
                "delta_mean": dmean,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        )

    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(OUT / "paired_bootstrap_deltas.csv", index=False)

    print("\nRetrieval summary:\n", ret_sum_df.to_string(index=False))
    print("\nReport summary:\n", ans_sum_df.to_string(index=False))
    print("\nWrote outputs to", OUT)


if __name__ == "__main__":
    main()
