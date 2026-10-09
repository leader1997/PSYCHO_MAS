"""Collection and manual challenge query sets."""

from __future__ import annotations

from knowledge_db_contextual_eval.config import path

CHALLENGE_QUERIES = [
    {
        "query_id": "challenge_phq9",
        "query": "Which sources support using PHQ-9 or PHQ-4 as screening instruments for college student depression?",
        "expected_category": "B_college_depression_anxiety",
        "expected_doc_ids": {"DOC_B006", "DOC_B007", "DOC_A009"},
        "source": "manual_challenge",
    },
    {
        "query_id": "challenge_gad7",
        "query": "Find evidence on the reliability or validity of GAD-7 for university or college students.",
        "expected_category": "B_college_depression_anxiety",
        "expected_doc_ids": {"DOC_B002", "DOC_B003"},
        "source": "manual_challenge",
    },
    {
        "query_id": "challenge_loneliness",
        "query": "What evidence links loneliness, social support, and depressive symptoms in student or young-adult populations?",
        "expected_category": "C_loneliness_isolation",
        "expected_doc_ids": {"DOC_C001", "DOC_C002", "DOC_C003"},
        "source": "manual_challenge",
    },
    {
        "query_id": "challenge_academic",
        "query": "Retrieve studies about academic stress, academic impairment, procrastination, depression, and anxiety in university students.",
        "expected_category": "D_academic_impairment",
        "expected_doc_ids": {"DOC_D001", "DOC_D002", "DOC_D003", "DOC_D004"},
        "source": "manual_challenge",
    },
    {
        "query_id": "challenge_financial",
        "query": "Find sources about food insecurity, material hardship, financial stress, and student mental health.",
        "expected_category": "E_financial_food",
        "expected_doc_ids": {"DOC_E001", "DOC_E002", "DOC_E003"},
        "source": "manual_challenge",
    },
    {
        "query_id": "challenge_xai",
        "query": "Retrieve sources explaining SHAP, local feature attribution, calibration, and why explanations are not causal claims.",
        "expected_category": "H_xai_calibration",
        "expected_doc_ids": {"DOC_H001", "DOC_H002", "DOC_H003", "DOC_H005"},
        "source": "manual_challenge",
    },
    {
        "query_id": "challenge_rag_eval",
        "query": "Which papers define RAG, citation faithfulness, hallucination evaluation, and factuality checks for LLM generated text?",
        "expected_category": "I_rag_llm_eval",
        "expected_doc_ids": {"DOC_I001", "DOC_I002", "DOC_I003", "DOC_I004", "DOC_I005", "DOC_I006"},
        "source": "manual_challenge",
    },
]


def load_collection_queries() -> list[dict]:
    queries: list[dict] = []
    with path("search_queries").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                queries.append(
                    {
                        "query_id": f"collection_{len(queries) + 1:03d}",
                        "query": parts[1],
                        "expected_category": parts[0],
                        "expected_doc_ids": None,
                        "source": "collection_query",
                    }
                )
    return queries


def load_all_queries(chunks: list[dict], docs: list[dict]) -> list[dict]:
    queries = load_collection_queries() + [dict(q) for q in CHALLENGE_QUERIES]
    available_doc_ids = {chunk["doc_id"] for chunk in chunks}
    for query in queries:
        if query.get("expected_doc_ids"):
            query["expected_doc_ids"] = set(query["expected_doc_ids"]) & available_doc_ids
    return queries


def relevant_doc_ids(query_spec: dict, docs: list[dict]) -> set[str]:
    if query_spec.get("expected_doc_ids"):
        return set(query_spec["expected_doc_ids"])
    return {doc["doc_id"] for doc in docs if doc.get("category_code") == query_spec["expected_category"]}
