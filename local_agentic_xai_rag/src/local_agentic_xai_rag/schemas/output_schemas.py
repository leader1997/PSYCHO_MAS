"""JSON schemas for advisory pipeline."""

EVIDENCE_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "supported_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supporting_chunk_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence_strength": {"type": "string"},
                    "relevance_to_xai": {"type": "string"},
                },
                "required": ["claim", "supporting_chunk_ids", "evidence_strength", "relevance_to_xai"],
            },
        },
        "unsupported_or_uncertain_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["point", "reason"],
            },
        },
        "clinical_boundary": {"type": "string"},
    },
    "required": ["case_id", "supported_claims", "unsupported_or_uncertain_points", "clinical_boundary"],
}

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "screening_summary": {"type": "string"},
        "model_explanation": {"type": "string"},
        "evidence_grounding": {"type": "string"},
        "recommended_next_step": {"type": "string"},
        "limitations": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "claim_supported": {"type": "string"},
                },
                "required": ["chunk_id", "claim_supported"],
            },
        },
        "safety_status": {"type": "string"},
    },
    "required": [
        "case_id", "screening_summary", "model_explanation", "evidence_grounding",
        "recommended_next_step", "limitations", "citations", "safety_status",
    ],
}

FAITHFULNESS_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "support_status": {
            "type": "string",
            "enum": ["supported", "partially_supported", "unsupported", "contradicted"],
        },
        "explanation": {"type": "string"},
    },
    "required": ["support_status", "explanation"],
}

QUERY_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "safe_case_summary": {"type": "string"},
        "probability_level": {
            "type": "string",
            "enum": ["high_above_threshold", "near_threshold", "below_threshold"],
        },
        "internal_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["question", "purpose"],
            },
        },
        "retrieval_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query_type": {"type": "string"},
                    "query_text": {"type": "string"},
                    "focus_keywords": {"type": "array", "items": {"type": "string"}},
                    "related_xai_factor": {"type": "string"},
                },
                "required": ["query_type", "query_text", "focus_keywords"],
            },
        },
        "recommendation_objectives": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "case_id",
        "safe_case_summary",
        "probability_level",
        "internal_questions",
        "retrieval_queries",
        "recommendation_objectives",
    ],
}
