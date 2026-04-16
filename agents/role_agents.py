from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.base import RoleAgent
from prompts.system_prompts import (
    CONFLICT_PROMPT,
    CRITIC_PROMPT,
    EXTRACTOR_PROMPT,
    REFUSAL_JUDGE_PROMPT,
    RELEVANCE_PROMPT,
    SINGLE_AGENT_PROMPT,
    SUMMARIZER_PROMPT,
    SYNTHESIZER_PROMPT,
)
from prompts.output_schemas import (
    CONFLICT_OUTPUT_SCHEMA,
    CRITIC_OUTPUT_SCHEMA,
    EXTRACTOR_OUTPUT_SCHEMA,
    REFUSAL_JUDGE_OUTPUT_SCHEMA,
    RELEVANCE_OUTPUT_SCHEMA,
    SINGLE_AGENT_OUTPUT_SCHEMA,
    SUMMARIZER_OUTPUT_SCHEMA,
    SYNTHESIZER_OUTPUT_SCHEMA,
)
from utils.schemas import FinalAnswer, RefusalDecision


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


class EvidenceExtractorAgent(RoleAgent):
    def __init__(self, llm):
        super().__init__("extractor", llm, EXTRACTOR_PROMPT)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        return f"Query: {state['query']}\nDocs:\n{_dump(state.get('retrieved_docs', []))}"

    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        state["extracted_evidence"] = output.get("evidence", [])
        return state

    def output_schema_name(self) -> str:
        return "extractor_output"

    def output_schema(self) -> Dict[str, Any]:
        return EXTRACTOR_OUTPUT_SCHEMA


class RelevanceClassifierAgent(RoleAgent):
    def __init__(self, llm):
        super().__init__("relevance", llm, RELEVANCE_PROMPT)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        return (
            f"Query: {state['query']}\n"
            f"Evidence:\n{_dump(state.get('extracted_evidence', []))}"
        )

    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        state["relevant_evidence"] = output.get("relevant_evidence", [])
        return state

    def output_schema_name(self) -> str:
        return "relevance_output"

    def output_schema(self) -> Dict[str, Any]:
        return RELEVANCE_OUTPUT_SCHEMA


class ConflictDetectorAgent(RoleAgent):
    def __init__(self, llm):
        super().__init__("conflict", llm, CONFLICT_PROMPT)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        return _dump({"query": state["query"], "relevant_evidence": state.get("relevant_evidence", [])})

    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        state["conflict_report"] = output
        return state

    def output_schema_name(self) -> str:
        return "conflict_output"

    def output_schema(self) -> Dict[str, Any]:
        return CONFLICT_OUTPUT_SCHEMA


class CriticAgent(RoleAgent):
    def __init__(self, llm):
        super().__init__("critic", llm, CRITIC_PROMPT)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        return _dump(
            {
                "query": state["query"],
                "relevant_evidence": state.get("relevant_evidence", []),
                "conflict_report": state.get("conflict_report", {}),
            }
        )

    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        critiques = output.get("critiques", [])
        if not isinstance(critiques, list):
            critiques = [str(critiques)]
        state["critiques"] = [str(c) for c in critiques]
        return state

    def output_schema_name(self) -> str:
        return "critic_output"

    def output_schema(self) -> Dict[str, Any]:
        return CRITIC_OUTPUT_SCHEMA


class SummarizerAgent(RoleAgent):
    def __init__(self, llm):
        super().__init__("summarizer", llm, SUMMARIZER_PROMPT)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        return _dump(
            {
                "query": state["query"],
                "relevant_evidence": state.get("relevant_evidence", []),
                "critiques": state.get("critiques", []),
                "conflict_report": state.get("conflict_report", {}),
            }
        )

    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        state["summary"] = str(output.get("summary", ""))
        return state

    def output_schema_name(self) -> str:
        return "summarizer_output"

    def output_schema(self) -> Dict[str, Any]:
        return SUMMARIZER_OUTPUT_SCHEMA


class FinalSynthesizerAgent(RoleAgent):
    def __init__(self, llm):
        super().__init__("synthesizer", llm, SYNTHESIZER_PROMPT)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        return _dump(
            {
                "query": state["query"],
                "summary": state.get("summary", ""),
                "relevant_evidence": state.get("relevant_evidence", []),
                "conflict_report": state.get("conflict_report", {}),
            }
        )

    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        answer = str(output.get("answer", ""))
        evidence_doc_ids = output.get("evidence_doc_ids", [])
        if not isinstance(evidence_doc_ids, list):
            evidence_doc_ids = []
        state["final_answer"] = FinalAnswer(
            answer=answer,
            evidence_doc_ids=[str(x) for x in evidence_doc_ids],
        ).__dict__
        return state

    def output_schema_name(self) -> str:
        return "synthesizer_output"

    def output_schema(self) -> Dict[str, Any]:
        return SYNTHESIZER_OUTPUT_SCHEMA


class RefusalJudgeAgent(RoleAgent):
    def __init__(self, llm):
        super().__init__("refusal_judge", llm, REFUSAL_JUDGE_PROMPT)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        return _dump(
            {
                "query": state["query"],
                "summary": state.get("summary", ""),
                "conflict_report": state.get("conflict_report", {}),
                "proposed_answer": state.get("final_answer", {}),
            }
        )

    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        refuse = bool(output.get("refuse", False))
        reason = str(output.get("reason", ""))
        state["refusal_decision"] = RefusalDecision(refuse=refuse, reason=reason).__dict__

        final = state.get("final_answer") or {"answer": "", "evidence_doc_ids": []}
        final["refusal"] = refuse
        final["refusal_reason"] = reason
        state["final_answer"] = final
        return state

    def output_schema_name(self) -> str:
        return "refusal_judge_output"

    def output_schema(self) -> Dict[str, Any]:
        return REFUSAL_JUDGE_OUTPUT_SCHEMA


class SingleAgentResponder(RoleAgent):
    def __init__(self, llm):
        super().__init__("single_agent", llm, SINGLE_AGENT_PROMPT)

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        return _dump({"query": state["query"], "retrieved_docs": state.get("retrieved_docs", [])})

    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        answer = str(output.get("answer", ""))
        evidence_doc_ids = output.get("evidence_doc_ids", [])
        refuse = bool(output.get("refuse", False))
        refusal_reason = str(output.get("refusal_reason", ""))
        if not isinstance(evidence_doc_ids, list):
            evidence_doc_ids = []

        state["final_answer"] = FinalAnswer(
            answer=answer,
            evidence_doc_ids=[str(x) for x in evidence_doc_ids],
            refusal=refuse,
            refusal_reason=refusal_reason,
        ).__dict__
        state["refusal_decision"] = RefusalDecision(refuse=refuse, reason=refusal_reason).__dict__
        return state

    def output_schema_name(self) -> str:
        return "single_agent_output"

    def output_schema(self) -> Dict[str, Any]:
        return SINGLE_AGENT_OUTPUT_SCHEMA
