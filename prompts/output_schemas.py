EXTRACTOR_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "doc_id": {"type": "string"},
                    "snippet": {"type": "string"},
                    "support": {"type": "string", "enum": ["support", "against", "neutral"]},
                },
                "required": ["claim", "doc_id", "snippet", "support"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evidence"],
    "additionalProperties": False,
}

RELEVANCE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "doc_id": {"type": "string"},
                    "snippet": {"type": "string"},
                    "support": {"type": "string", "enum": ["support", "against", "neutral"]},
                },
                "required": ["claim", "doc_id", "snippet", "support"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relevant_evidence"],
    "additionalProperties": False,
}

CONFLICT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "has_conflict": {"type": "boolean"},
        "summary": {"type": "string"},
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "factual_contradiction",
                            "temporal_mismatch",
                            "scope_mismatch",
                            "methodological_disagreement",
                            "source_reliability",
                            "insufficient_evidence",
                            "ambiguity",
                            "other",
                        ],
                    },
                    "content": {"type": "string"},
                    "evidence_doc_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["type", "content", "evidence_doc_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["has_conflict", "summary", "conflicts"],
    "additionalProperties": False,
}

CRITIC_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "critiques": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["critiques"],
    "additionalProperties": False,
}

SUMMARIZER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
    "additionalProperties": False,
}

SYNTHESIZER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence_doc_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "evidence_doc_ids"],
    "additionalProperties": False,
}

REFUSAL_JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "refuse": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["refuse", "reason"],
    "additionalProperties": False,
}

SINGLE_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "evidence_doc_ids": {"type": "array", "items": {"type": "string"}},
        "refuse": {"type": "boolean"},
        "refusal_reason": {"type": "string"},
    },
    "required": ["answer", "evidence_doc_ids", "refuse", "refusal_reason"],
    "additionalProperties": False,
}

JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "number"},
        "groundedness": {"type": "number"},
        "refusal_correctness": {"type": "number"},
        "conflict_handling": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "groundedness", "refusal_correctness", "conflict_handling", "reason"],
    "additionalProperties": False,
}