"""JSON schemas for structured generation."""

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
                "additionalProperties": False,
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
                "additionalProperties": False,
            },
        },
        "clinical_boundary": {"type": "string"},
    },
    "required": [
        "case_id",
        "supported_claims",
        "unsupported_or_uncertain_points",
        "clinical_boundary",
    ],
    "additionalProperties": False,
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
                "additionalProperties": False,
            },
        },
        "safety_status": {"type": "string"},
    },
    "required": [
        "case_id",
        "screening_summary",
        "model_explanation",
        "evidence_grounding",
        "recommended_next_step",
        "limitations",
        "citations",
        "safety_status",
    ],
    "additionalProperties": False,
}

SAFETY_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "diagnostic_overclaiming": {"type": "string"},
        "medical_advice": {"type": "string"},
        "causality_misuse": {"type": "string"},
        "unsupported_claims": {"type": "string"},
        "missing_disclaimer": {"type": "string"},
        "missing_referral": {"type": "string"},
        "stigmatizing_language": {"type": "string"},
        "privacy_leakage": {"type": "string"},
        "overall_safety": {"type": "string"},
        "revision_required": {"type": "boolean"},
        "comments": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "case_id",
        "diagnostic_overclaiming",
        "medical_advice",
        "causality_misuse",
        "unsupported_claims",
        "missing_disclaimer",
        "missing_referral",
        "stigmatizing_language",
        "privacy_leakage",
        "overall_safety",
        "revision_required",
        "comments",
    ],
    "additionalProperties": False,
}
