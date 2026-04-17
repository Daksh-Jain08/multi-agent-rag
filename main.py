import json
import os
import time
import argparse
import csv
import re
from pathlib import Path
from statistics import mean
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Optional

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from src.chunk_helpers import reduce_documents

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, sample_id: str) -> Path:
        return self.checkpoint_dir / f"sample_{sample_id}.json"

    def load_checkpoint(self, sample_id: str) -> Optional[Dict[str, Any]]:
        path = self.get_checkpoint_path(sample_id)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None

    def save_checkpoint(self, sample_id: str, checkpoint_data: Dict[str, Any]):
        path = self.get_checkpoint_path(sample_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=True)

    def initialize_checkpoint(self, sample_id: str, query: str, docs: List[Dict[str, str]]) -> Dict[str, Any]:
        checkpoint = {
            "sample_id": sample_id,
            "query": query,
            "documents": docs,
            "completed_steps": [],
            "intermediate_results": {},
            "token_usage": {
                "overall": {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "by_step": {},
                "api_calls": [],
            },
            "checkpoint_created_at": datetime.utcnow().isoformat(),
            "last_updated_at": datetime.utcnow().isoformat(),
        }
        self.save_checkpoint(sample_id, checkpoint)
        return checkpoint

    def update_checkpoint(
        self,
        sample_id: str,
        step_name: str,
        result: Dict[str, Any],
        token_usage: Optional[Dict[str, Any]] = None,
    ):
        checkpoint = self.load_checkpoint(sample_id)
        if checkpoint is None:
            return

        if step_name not in checkpoint["completed_steps"]:
            checkpoint["completed_steps"].append(step_name)

        checkpoint["intermediate_results"][step_name] = result

        usage_root = checkpoint.setdefault(
            "token_usage",
            {
                "overall": {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "by_step": {},
                "api_calls": [],
            },
        )
        usage_root.setdefault("overall", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        usage_root.setdefault("by_step", {})
        usage_root.setdefault("api_calls", [])

        if token_usage:
            step_summary = token_usage.get("step_summary", {})
            api_calls = token_usage.get("api_calls", [])

            existing_step = usage_root["by_step"].get(
                step_name,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

            existing_step["calls"] = int(existing_step.get("calls", 0)) + int(step_summary.get("calls", 0))
            existing_step["prompt_tokens"] = int(existing_step.get("prompt_tokens", 0)) + int(step_summary.get("prompt_tokens", 0))
            existing_step["completion_tokens"] = int(existing_step.get("completion_tokens", 0)) + int(step_summary.get("completion_tokens", 0))
            existing_step["total_tokens"] = int(existing_step.get("total_tokens", 0)) + int(step_summary.get("total_tokens", 0))
            usage_root["by_step"][step_name] = existing_step

            overall = usage_root["overall"]
            overall["calls"] = int(overall.get("calls", 0)) + int(step_summary.get("calls", 0))
            overall["prompt_tokens"] = int(overall.get("prompt_tokens", 0)) + int(step_summary.get("prompt_tokens", 0))
            overall["completion_tokens"] = int(overall.get("completion_tokens", 0)) + int(step_summary.get("completion_tokens", 0))
            overall["total_tokens"] = int(overall.get("total_tokens", 0)) + int(step_summary.get("total_tokens", 0))

            if isinstance(api_calls, list) and api_calls:
                usage_root["api_calls"].extend(api_calls)

        checkpoint["last_updated_at"] = datetime.utcnow().isoformat()
        self.save_checkpoint(sample_id, checkpoint)

    def has_step(self, sample_id: str, step_name: str) -> bool:
        checkpoint = self.load_checkpoint(sample_id)
        if checkpoint is None:
            return False
        return step_name in checkpoint.get("completed_steps", [])

    def get_step_result(self, sample_id: str, step_name: str) -> Optional[Dict[str, Any]]:
        checkpoint = self.load_checkpoint(sample_id)
        if checkpoint is None:
            return None
        return checkpoint.get("intermediate_results", {}).get(step_name)

class ExtractorOutput(BaseModel):
    doc_id: str
    key_facts: List[str]
    claims: List[str]
    evidence_spans: List[str]
    reason: str = Field(description="Under 60 words")


class VerdictOutput(BaseModel):
    doc_id: str
    verdict: str
    reason: str = Field(description="Under 60 words")
    confidence: float


class RelevanceOutput(BaseModel):
    doc_id: str
    relevance_score: float
    is_relevant: bool
    reason: str = Field(description="Under 60 words")


class AggregatorOutput(BaseModel):
    supporting_docs: List[str]
    partial_docs: List[str]
    irrelevant_docs: List[str]
    evidence_summary: List[Dict[str, Any]]


class ConflictDetectionOutput(BaseModel):
    conflict_type: str
    conflict_reason: str = Field(description="Under 60 words")
    clusters: List[Dict[str, Any]]


class RefusalDecisionOutput(BaseModel):
    should_abstain: bool
    reason: str = Field(description="Under 60 words")


class AdjudicatorOutput(BaseModel):
    answer: str
    citations: List[str]
    abstain: bool
    abstain_reason: str
    final_reasoning: str = Field(description="Under 60 words")


class EvaluatorOutput(BaseModel):
    correct_conflict: bool
    correct_type: bool
    correct_abstain: bool
    grounded: bool
    behavior_adherence: bool
    score: float
    feedback: str


class PipelineState(TypedDict, total=False):
    query: str
    retrieved_docs: List[Dict[str, str]]
    relevance_results: List[Dict[str, Any]]
    extraction_results: List[Dict[str, Any]]
    verdict_results: List[Dict[str, Any]]
    aggregation: Dict[str, Any]
    conflict: Dict[str, Any]
    refusal: Dict[str, Any]
    adjudication: Dict[str, Any]


def _extract_section(prompt: str, section_name: str) -> str:
    pattern = rf"{re.escape(section_name)}:\n(.*?)(?:\n\n[A-Z][A-Za-z ]*:\n|\Z)"
    match = re.search(pattern, prompt, flags=re.DOTALL)
    return (match.group(1).strip() if match else "")


def _extract_json_section(prompt: str, section_name: str, fallback: Any) -> Any:
    section = _extract_section(prompt, section_name)
    if not section:
        return fallback
    try:
        return json.loads(section)
    except json.JSONDecodeError:
        return fallback


def _normalize_text_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_overlap_score(query: str, text: str) -> int:
    q_tokens = set(_normalize_text_tokens(query))
    t_tokens = set(_normalize_text_tokens(text))
    if not q_tokens or not t_tokens:
        return 0
    return len(q_tokens.intersection(t_tokens))


class MockStructuredModel:
    def __init__(self, schema: BaseModel):
        self.schema = schema

    def invoke(self, messages: List[Any]) -> BaseModel:
        user_prompt = ""
        if messages:
            user_prompt = str(messages[-1].content)

        payload = self._build_payload(user_prompt)
        return self.schema(**payload)

    def _build_payload(self, user_prompt: str) -> Dict[str, Any]:
        schema_name = self.schema.__name__

        if schema_name == "RelevanceOutput":
            query = _extract_section(user_prompt, "Query")
            doc = _extract_json_section(user_prompt, "Document", {})
            text = str(doc.get("text", ""))
            doc_id = str(doc.get("doc_id", "doc_0"))
            overlap = _token_overlap_score(query, text)
            is_relevant = overlap >= 2
            score = min(1.0, 0.15 * overlap + 0.1)
            return {
                "doc_id": doc_id,
                "relevance_score": score,
                "is_relevant": is_relevant,
                "reason": "Token overlap heuristic from query and passage.",
            }

        if schema_name == "ExtractorOutput":
            doc = _extract_json_section(user_prompt, "Document", {})
            text = str(doc.get("text", "")).strip()
            doc_id = str(doc.get("doc_id", "doc_0"))
            
            # Extract multiple key facts from sentences
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            key_facts = [s[:150] for s in sentences[:3]]  # Top 3 sentences
            
            # Claims are standalone statements
            claims = [f"Claim: {s[:200]}" for s in sentences[:2]]
            
            # Evidence spans are short quotes
            evidence_spans = [s[:100] for s in sentences[:3]]
            
            return {
                "doc_id": doc_id,
                "key_facts": key_facts if key_facts else ["No facts extracted."],
                "claims": claims if claims else ["No claims found."],
                "evidence_spans": evidence_spans if evidence_spans else ["No evidence."],
                "reason": "Extracted multiple atomic facts and claims from document.",
            }

        if schema_name == "VerdictOutput":
            extracted = _extract_json_section(user_prompt, "Extracted", {})
            doc_id = str(extracted.get("doc_id", "doc_0"))
            key_facts = extracted.get("key_facts", [])
            claims = extracted.get("claims", [])
            
            # If we have substantive claims/facts, it supports; otherwise irrelevant
            has_content = (len(key_facts) > 0 and any(f.strip() and "no" not in f.lower() for f in key_facts))
            
            if has_content:
                verdict = "supports"
                confidence = 0.85
            else:
                verdict = "irrelevant"
                confidence = 0.5
            
            return {
                "doc_id": doc_id,
                "verdict": verdict,
                "reason": "Evaluated based on extracted substantive content.",
                "confidence": confidence,
            }

        if schema_name == "AggregatorOutput":
            verdicts = _extract_json_section(user_prompt, "Verdicts", [])
            extractions = _extract_json_section(user_prompt, "Extractions", [])
            extraction_map = {
                str(item.get("doc_id", "")): item for item in extractions if isinstance(item, dict)
            }

            supporting_docs: List[str] = []
            partial_docs: List[str] = []
            irrelevant_docs: List[str] = []
            evidence_summary: List[Dict[str, Any]] = []

            for verdict_item in verdicts:
                if not isinstance(verdict_item, dict):
                    continue
                doc_id = str(verdict_item.get("doc_id", ""))
                verdict = str(verdict_item.get("verdict", "irrelevant"))
                confidence = float(verdict_item.get("confidence", 0.0))
                
                extraction = extraction_map.get(doc_id, {})
                key_facts = extraction.get("key_facts", [])
                key_fact = key_facts[0] if key_facts else ""

                if verdict == "supports":
                    supporting_docs.append(doc_id)
                elif verdict == "partially_supports":
                    partial_docs.append(doc_id)
                else:
                    irrelevant_docs.append(doc_id)

                evidence_summary.append(
                    {
                        "doc_id": doc_id,
                        "key_fact": key_fact,
                        "verdict": verdict,
                        "confidence": confidence,
                    }
                )

            return {
                "supporting_docs": supporting_docs,
                "partial_docs": partial_docs,
                "irrelevant_docs": irrelevant_docs,
                "evidence_summary": evidence_summary,
            }

        if schema_name == "ConflictDetectionOutput":
            extractions = _extract_json_section(user_prompt, "Extractions", [])
            years = set()
            for item in extractions:
                if not isinstance(item, dict):
                    continue
                combined = f"{item.get('key_fact', '')} {item.get('quote', '')}"
                years.update(re.findall(r"\b(?:19|20)\d{2}\b", combined))

            if len(years) >= 2:
                conflict_type = "conflicting"
                conflict_reason = "Multiple distinct years detected across evidence."
            else:
                conflict_type = "complementary"
                conflict_reason = "Evidence appears mutually compatible."

            return {
                "conflict_type": conflict_type,
                "conflict_reason": conflict_reason,
                "clusters": [
                    {"label": "cluster_a", "doc_ids": [], "summary": "Primary evidence cluster."},
                    {"label": "cluster_b", "doc_ids": [], "summary": "Secondary evidence cluster."},
                ],
            }

        if schema_name == "RefusalDecisionOutput":
            aggregation = _extract_json_section(user_prompt, "Aggregation", {})
            conflict = _extract_json_section(user_prompt, "Conflict", {})
            supporting_docs = aggregation.get("supporting_docs", []) if isinstance(aggregation, dict) else []
            partial_docs = aggregation.get("partial_docs", []) if isinstance(aggregation, dict) else []
            conflict_type = str(conflict.get("conflict_type", "")) if isinstance(conflict, dict) else ""

            should_abstain = (
                conflict_type in {"conflicting", "misinformation"}
                or (len(supporting_docs) == 0 and len(partial_docs) < 2)
            )
            return {
                "should_abstain": should_abstain,
                "reason": "Abstain on severe conflict or low supporting evidence.",
            }

        if schema_name == "AdjudicatorOutput":
            aggregation = _extract_json_section(user_prompt, "Aggregation", {})
            refusal = _extract_json_section(user_prompt, "Refusal", {})
            should_abstain = bool(refusal.get("should_abstain", False)) if isinstance(refusal, dict) else False
            evidence_summary = aggregation.get("evidence_summary", []) if isinstance(aggregation, dict) else []

            citations: List[str] = []
            answer = ""
            if isinstance(evidence_summary, list):
                for item in evidence_summary:
                    if not isinstance(item, dict):
                        continue
                    doc_id = str(item.get("doc_id", ""))
                    if doc_id and doc_id not in citations:
                        citations.append(doc_id)
                    if not answer and str(item.get("key_fact", "")).strip():
                        answer = str(item.get("key_fact", ""))

            if should_abstain:
                return {
                    "answer": "Insufficient reliable evidence to answer confidently.",
                    "citations": citations[:3],
                    "abstain": True,
                    "abstain_reason": "Conflict or weak support triggered abstention.",
                    "final_reasoning": "Evidence quality threshold not met for confident answer.",
                }

            return {
                "answer": answer or "Best supported answer synthesized from evidence.",
                "citations": citations[:3],
                "abstain": False,
                "abstain_reason": "",
                "final_reasoning": "Used highest-overlap evidence with deterministic citation selection.",
            }

        if schema_name == "EvaluatorOutput":
            rule_eval = _extract_json_section(user_prompt, "Rule Evaluation", {})
            pipeline_output = _extract_json_section(user_prompt, "Pipeline Output", {})

            correct_type = bool(rule_eval.get("correct_type", False)) if isinstance(rule_eval, dict) else False
            correct_abstain = bool(rule_eval.get("correct_abstain", False)) if isinstance(rule_eval, dict) else False
            correct_conflict = bool(rule_eval.get("correct_conflict", False)) if isinstance(rule_eval, dict) else False

            citations = []
            if isinstance(pipeline_output, dict):
                adjudication = pipeline_output.get("adjudication", {})
                if isinstance(adjudication, dict):
                    citations = adjudication.get("citations", [])

            grounded = isinstance(citations, list) and len(citations) > 0
            behavior_adherence = True
            score = (float(correct_type) + float(correct_abstain) + float(correct_conflict) + float(grounded)) / 4.0
            return {
                "correct_conflict": correct_conflict,
                "correct_type": correct_type,
                "correct_abstain": correct_abstain,
                "grounded": grounded,
                "behavior_adherence": behavior_adherence,
                "score": score,
                "feedback": "Deterministic mock evaluator mirrored rule signals with citation grounding.",
            }

        if schema_name == "AdjudicatorOutput":
            return {
                "answer": "Mock answer.",
                "citations": [],
                "abstain": False,
                "abstain_reason": "",
                "final_reasoning": "Mock adjudication fallback.",
            }

        return {field_name: None for field_name in self.schema.model_fields.keys()}


class MockLLM:
    is_mock = True

    def with_structured_output(self, schema: BaseModel) -> MockStructuredModel:
        return MockStructuredModel(schema)


class BaseAgent:
    def __init__(self, llm: Any):
        self.llm = llm
        self._usage_events: List[Dict[str, Any]] = []

    def _extract_token_usage(self, raw_message: Any) -> Dict[str, int]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if raw_message is None:
            return usage

        usage_meta = {}
        if hasattr(raw_message, "usage_metadata") and isinstance(raw_message.usage_metadata, dict):
            usage_meta = raw_message.usage_metadata
            usage["prompt_tokens"] = int(usage_meta.get("input_tokens", 0))
            usage["completion_tokens"] = int(usage_meta.get("output_tokens", 0))
            usage["total_tokens"] = int(usage_meta.get("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"]))
            return usage

        response_meta = {}
        if hasattr(raw_message, "response_metadata") and isinstance(raw_message.response_metadata, dict):
            response_meta = raw_message.response_metadata

        token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}
        if isinstance(token_usage, dict):
            usage["prompt_tokens"] = int(
                token_usage.get("prompt_tokens")
                or token_usage.get("input_tokens")
                or token_usage.get("prompt_token_count")
                or 0
            )
            usage["completion_tokens"] = int(
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
                or token_usage.get("candidates_token_count")
                or 0
            )
            usage["total_tokens"] = int(
                token_usage.get("total_tokens")
                or token_usage.get("total_token_count")
                or (usage["prompt_tokens"] + usage["completion_tokens"])
            )

        return usage

    def consume_usage_events(self) -> List[Dict[str, Any]]:
        events = self._usage_events[:]
        self._usage_events.clear()
        return events

    def _invoke_structured(self, schema: BaseModel, system_prompt: str, user_prompt: str) -> BaseModel:
        print(f"[LLM] Calling structured model for {schema.__name__}")
        if not getattr(self.llm, "is_mock", False):
            time.sleep(15)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        raw_message = None
        try:
            model = self.llm.with_structured_output(schema, include_raw=True)
            structured_out = model.invoke(messages)
            if isinstance(structured_out, dict) and structured_out.get("parsed") is not None:
                output = structured_out["parsed"]
                raw_message = structured_out.get("raw")
            else:
                output = structured_out
        except TypeError:
            model = self.llm.with_structured_output(schema)
            output = model.invoke(messages)

        token_usage = self._extract_token_usage(raw_message)
        self._usage_events.append(
            {
                "schema": schema.__name__,
                "model": str(getattr(self.llm, "model", "unknown")),
                "timestamp": datetime.utcnow().isoformat(),
                "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(token_usage.get("completion_tokens", 0)),
                "total_tokens": int(token_usage.get("total_tokens", 0)),
            }
        )

        print(f"[LLM] Completed structured model for {schema.__name__}")
        return output


def summarize_usage_events(step_name: str, usage_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "calls": len(usage_events),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    enriched_events: List[Dict[str, Any]] = []
    for idx, event in enumerate(usage_events, start=1):
        prompt_tokens = int(event.get("prompt_tokens", 0))
        completion_tokens = int(event.get("completion_tokens", 0))
        total_tokens = int(event.get("total_tokens", prompt_tokens + completion_tokens))

        summary["prompt_tokens"] += prompt_tokens
        summary["completion_tokens"] += completion_tokens
        summary["total_tokens"] += total_tokens

        enriched = dict(event)
        enriched["step"] = step_name
        enriched["call_index"] = idx
        enriched_events.append(enriched)

    return {"step_summary": summary, "api_calls": enriched_events}


class SingleAgentBaseline(BaseAgent):
    def run(self, query: str, docs: List[Dict[str, str]]) -> AdjudicatorOutput:
        system_prompt = """You are SingleAgentBaseline for RAG adjudication. Read query and documents, detect conflicts, decide abstention if evidence weak, provide grounded answer with citations.
Rules: cite only doc_ids, final_reasoning under 60 words.
""".strip()

        user_prompt = f"""Query: {query}
Documents: {json.dumps(docs[:3], ensure_ascii=True)}
Return AdjudicatorOutput.""".strip()

        return self._invoke_structured(AdjudicatorOutput, system_prompt, user_prompt)


class RelevanceAgent(BaseAgent):
    def run(self, query: str, doc: Dict[str, str]) -> RelevanceOutput:
        system_prompt = """You are RelevanceAgent. Classify document relevance to query.
Rules: relevance_score in [0,1], is_relevant boolean, reason under 60 words.
""".strip()

        user_prompt = f"""Query: {query}
Document: {json.dumps(doc, ensure_ascii=True)}
Return RelevanceOutput.""".strip()

        return self._invoke_structured(RelevanceOutput, system_prompt, user_prompt)


class ExtractorAgent(BaseAgent):
    def run(self, query: str, doc: Dict[str, str]) -> ExtractorOutput:
        system_prompt = """You are ExtractorAgent. Extract key atomic facts from the document.
Rules:
- key_facts: list of main claims (3-5)
- claims: explicit logical statements
- evidence_spans: short direct quotes
- Be concrete and factual.
""".strip()

        user_prompt = f"""Query: {query}
Document: {json.dumps(doc, ensure_ascii=True)}
Return ExtractorOutput.""".strip()

        return self._invoke_structured(ExtractorOutput, system_prompt, user_prompt)


class VerdictAgent(BaseAgent):
    def run(self, extracted: Dict[str, Any]) -> VerdictOutput:
        system_prompt = """You are VerdictAgent. Decide document relevance to query.
Allowed verdicts: supports, partially_supports, irrelevant.
Rules: confidence in [0,1], reason under 60 words.
""".strip()

        user_prompt = f"""Extracted: {json.dumps(extracted, ensure_ascii=True)}
Return VerdictOutput.""".strip()

        return self._invoke_structured(VerdictOutput, system_prompt, user_prompt)


class EvidenceAggregator(BaseAgent):
    def run(self, extractions: List[Dict[str, Any]], verdicts: List[Dict[str, Any]]) -> AggregatorOutput:
        system_prompt = """You are EvidenceAggregator. Group evidence by verdict type.
Rules: supporting_docs for supports, partial_docs for partially_supports, irrelevant_docs for irrelevant.
""".strip()

        user_prompt = f"""Extractions: {json.dumps(extractions, ensure_ascii=True)}
Verdicts: {json.dumps(verdicts, ensure_ascii=True)}
Return AggregatorOutput.""".strip()

        return self._invoke_structured(AggregatorOutput, system_prompt, user_prompt)


class ConflictDetectionAgent(BaseAgent):
    def run(self, extractions: List[Dict[str, Any]], verdicts: List[Dict[str, Any]], aggregation: Dict[str, Any]) -> ConflictDetectionOutput:
        system_prompt = """You are ConflictDetectionAgent. Classify evidence interaction.
Allowed types: no_conflict, complementary, conflicting, outdated, misinformation.
Rules: reason under 60 words, cluster compatible vs incompatible evidence.
""".strip()

        user_prompt = f"""Extractions: {json.dumps(extractions, ensure_ascii=True)}
Verdicts: {json.dumps(verdicts, ensure_ascii=True)}
Aggregation: {json.dumps(aggregation, ensure_ascii=True)}
Return ConflictDetectionOutput.""".strip()

        return self._invoke_structured(ConflictDetectionOutput, system_prompt, user_prompt)


class RefusalDecisionAgent(BaseAgent):
    def run(self, aggregation: Dict[str, Any], conflict: Dict[str, Any]) -> RefusalDecisionOutput:
        system_prompt = """You are RefusalDecisionAgent. Decide whether to abstain.
Abstain when evidence is irrelevant, conflicting, misinformation, or insufficient.
Rules: should_abstain boolean, reason under 60 words.
""".strip()

        user_prompt = f"""Aggregation: {json.dumps(aggregation, ensure_ascii=True)}
Conflict: {json.dumps(conflict, ensure_ascii=True)}
Return RefusalDecisionOutput.""".strip()

        return self._invoke_structured(RefusalDecisionOutput, system_prompt, user_prompt)


class AdjudicatorAgent(BaseAgent):
    def run(self, aggregation: Dict[str, Any], conflict: Dict[str, Any], refusal: Dict[str, Any]) -> AdjudicatorOutput:
        system_prompt = """You are AdjudicatorAgent. Produce final user-facing answer.
Rules: ground in evidence_summary and citations only, abstain if refusal recommends, final_reasoning under 60 words.
""".strip()

        user_prompt = f"""Aggregation: {json.dumps(aggregation, ensure_ascii=True)}
Conflict: {json.dumps(conflict, ensure_ascii=True)}
Refusal: {json.dumps(refusal, ensure_ascii=True)}
Return AdjudicatorOutput.""".strip()

        return self._invoke_structured(AdjudicatorOutput, system_prompt, user_prompt)


class EvaluatorAgent(BaseAgent):
    def run(
        self,
        query: str,
        docs: List[Dict[str, str]],
        pipeline_output: Dict[str, Any],
        ground_truth: Dict[str, Any],
        rule_eval: Dict[str, Any],
    ) -> EvaluatorOutput:
        system_prompt = """You are EvaluatorAgent. Assess pipeline output correctness.
Rules: booleans for correctness fields, score in [0,1], grounded boolean, feedback under 60 words, be strict.
""".strip()

        user_prompt = f"""Query: {query}
Pipeline Output: {json.dumps(pipeline_output, ensure_ascii=True)}
Ground Truth: {json.dumps(ground_truth, ensure_ascii=True)}
Rule Eval: {json.dumps(rule_eval, ensure_ascii=True)}
Return EvaluatorOutput.""".strip()

        return self._invoke_structured(EvaluatorOutput, system_prompt, user_prompt)


class MultiAgentPipeline:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.relevance_agent = RelevanceAgent(llm)
        self.extractor_agent = ExtractorAgent(llm)
        self.verdict_agent = VerdictAgent(llm)
        self.aggregator_agent = EvidenceAggregator(llm)
        self.conflict_agent = ConflictDetectionAgent(llm)
        self.refusal_agent = RefusalDecisionAgent(llm)
        self.adjudicator_agent = AdjudicatorAgent(llm)
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.sample_id: str = ""
        self.graph = self._build_graph()

    def _log(self, message: str):
        print(f"[PIPELINE] {message}")

    def _build_graph(self):
        builder = StateGraph(PipelineState)

        builder.add_node("relevance", self._relevance_node)
        builder.add_node("extract", self._extract_node)
        builder.add_node("verdict", self._verdict_node)
        builder.add_node("aggregate", self._aggregate_node)
        builder.add_node("conflict", self._conflict_node)
        builder.add_node("refusal", self._refusal_node)
        builder.add_node("adjudicate", self._adjudicate_node)

        builder.add_edge(START, "relevance")
        builder.add_edge("relevance", "extract")
        builder.add_edge("extract", "verdict")
        builder.add_edge("verdict", "aggregate")
        builder.add_edge("aggregate", "conflict")
        builder.add_edge("conflict", "refusal")
        builder.add_edge("refusal", "adjudicate")
        builder.add_edge("adjudicate", END)

        return builder.compile()

    def run(self, query: str, docs: List[Dict[str, str]], checkpoint_manager: Optional[CheckpointManager] = None, sample_id: str = "") -> Dict[str, Any]:
        self._log(f"Starting graph run for query: {query[:120]}")
        self.checkpoint_manager = checkpoint_manager
        self.sample_id = sample_id
        
        initial_state: PipelineState = {
            "query": query,
            "retrieved_docs": docs,
        }
        output = self.graph.invoke(initial_state)
        self._log("Completed graph run")
        return output

    def _relevance_node(self, state: PipelineState) -> Dict[str, Any]:
        query = state["query"]
        docs = state["retrieved_docs"]
        self._log(f"Relevance step started for {len(docs)} docs")
        results: List[Dict[str, Any]] = []
        for doc in docs:
            out = self.relevance_agent.run(query, doc)
            results.append(out.model_dump())
        usage_payload = summarize_usage_events("relevance", self.relevance_agent.consume_usage_events())
        self._log("Relevance step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "relevance",
                {"relevance_results": results},
                token_usage=usage_payload,
            )
        return {"relevance_results": results}

    def _extract_node(self, state: PipelineState) -> Dict[str, Any]:
        query = state["query"]
        docs = state["retrieved_docs"]
        self._log(f"Extraction step started for {len(docs)} docs")
        results: List[Dict[str, Any]] = []
        for doc in docs:
            out = self.extractor_agent.run(query, doc)
            results.append(out.model_dump())
        usage_payload = summarize_usage_events("extraction", self.extractor_agent.consume_usage_events())
        self._log("Extraction step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "extraction",
                {"extraction_results": results},
                token_usage=usage_payload,
            )
        return {"extraction_results": results}

    def _verdict_node(self, state: PipelineState) -> Dict[str, Any]:
        extractions = state.get("extraction_results", [])
        self._log(f"Verdict step started for {len(extractions)} docs")
        verdicts: List[Dict[str, Any]] = []

        for extraction in extractions:
            out = self.verdict_agent.run(extraction)
            verdicts.append(out.model_dump())

        usage_payload = summarize_usage_events("verdict", self.verdict_agent.consume_usage_events())
        self._log("Verdict step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "verdict",
                {"verdict_results": verdicts},
                token_usage=usage_payload,
            )
        return {"verdict_results": verdicts}

    def _aggregate_node(self, state: PipelineState) -> Dict[str, Any]:
        self._log("Aggregation step started")
        out = self.aggregator_agent.run(
            state.get("extraction_results", []),
            state.get("verdict_results", []),
        )
        usage_payload = summarize_usage_events("aggregation", self.aggregator_agent.consume_usage_events())
        self._log("Aggregation step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "aggregation",
                {"aggregation": out.model_dump()},
                token_usage=usage_payload,
            )
        return {"aggregation": out.model_dump()}

    def _conflict_node(self, state: PipelineState) -> Dict[str, Any]:
        self._log("Conflict detection step started")
        out = self.conflict_agent.run(
            state.get("extraction_results", []),
            state.get("verdict_results", []),
            state.get("aggregation", {}),
        )
        usage_payload = summarize_usage_events("conflict", self.conflict_agent.consume_usage_events())
        self._log("Conflict detection step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "conflict",
                {"conflict": out.model_dump()},
                token_usage=usage_payload,
            )
        return {"conflict": out.model_dump()}

    def _refusal_node(self, state: PipelineState) -> Dict[str, Any]:
        self._log("Refusal decision step started")
        out = self.refusal_agent.run(
            state.get("aggregation", {}),
            state.get("conflict", {}),
        )
        usage_payload = summarize_usage_events("refusal", self.refusal_agent.consume_usage_events())
        self._log("Refusal decision step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "refusal",
                {"refusal": out.model_dump()},
                token_usage=usage_payload,
            )
        return {"refusal": out.model_dump()}

    def _adjudicate_node(self, state: PipelineState) -> Dict[str, Any]:
        self._log("Adjudication step started")
        out = self.adjudicator_agent.run(
            state.get("aggregation", {}),
            state.get("conflict", {}),
            state.get("refusal", {}),
        )
        usage_payload = summarize_usage_events("adjudication", self.adjudicator_agent.consume_usage_events())
        self._log("Adjudication step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "adjudication",
                {"adjudication": out.model_dump()},
                token_usage=usage_payload,
            )
        return {"adjudication": out.model_dump()}


def build_result_record(
    sample: Dict[str, Any],
    pipeline_output: Dict[str, Any],
    rule_eval: Dict[str, Any],
    evaluator_dict: Dict[str, Any],
    run_id: str,
    architecture: str,
) -> Dict[str, Any]:
    sample_id = str(sample.get("id") or "")
    if not sample_id:
        sample_id = "sample_000"

    documents = [
        {
            "doc_id": d.get("doc_id", ""),
            "snippet": d.get("text", ""),
        }
        for d in sample.get("retrieved_docs", [])
    ]

    evidence_summary = []
    for item in pipeline_output.get("aggregation", {}).get("evidence_summary", []):
        evidence_summary.append(
            {
                "doc_id": item.get("doc_id", ""),
                "verdict": item.get("verdict", ""),
                "key_fact": item.get("key_fact", ""),
            }
        )

    record = {
        "id": sample_id,
        "query": sample.get("query", ""),
        "documents": documents,
        "ground_truth": {
            "conflict_type": rule_eval.get("ground_truth_conflict_type", ""),
            "should_abstain": rule_eval.get("ground_truth_abstain", False),
            "gold_answer": sample.get("gold_answer", ""),
        },
        "pipeline_output": {
            "conflict_type": pipeline_output.get("conflict", {}).get("conflict_type", ""),
            "conflict_reason": pipeline_output.get("conflict", {}).get("conflict_reason", ""),
            "evidence_summary": evidence_summary,
            "abstain": bool(
                pipeline_output.get("adjudication", {}).get("abstain", False)
                or pipeline_output.get("refusal", {}).get("should_abstain", False)
            ),
            "abstain_reason": (
                pipeline_output.get("adjudication", {}).get("abstain_reason", "")
                or pipeline_output.get("refusal", {}).get("reason", "")
            ),
            "answer": pipeline_output.get("adjudication", {}).get("answer", ""),
            "citations": pipeline_output.get("adjudication", {}).get("citations", []),
            "final_reasoning": pipeline_output.get("adjudication", {}).get("final_reasoning", ""),
        },
        "judge_output": {
            "correct_conflict": bool(evaluator_dict.get("correct_conflict", False)),
            "correct_type": bool(evaluator_dict.get("correct_type", False)),
            "correct_abstain": bool(evaluator_dict.get("correct_abstain", False)),
            "grounded": bool(evaluator_dict.get("grounded", False)),
            "behavior_adherence": bool(evaluator_dict.get("behavior_adherence", False)),
            "score": float(evaluator_dict.get("score", 0.0)),
            "feedback": evaluator_dict.get("feedback", ""),
        },
        "rule_based_eval": {
            "conflict_match": bool(rule_eval.get("correct_conflict", False)),
            "type_match": bool(rule_eval.get("correct_type", False)),
            "abstain_match": bool(rule_eval.get("correct_abstain", False)),
        },
        "metadata": {
            "architecture": architecture,
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat(),
            "run_id": run_id,
        },
    }
    return record


def write_outputs(output_dir: Path, run_id: str, records: List[Dict[str, Any]]):
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{run_id}_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    summary_path = output_dir / f"{run_id}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "query",
                "gt_conflict_type",
                "pred_conflict_type",
                "gt_should_abstain",
                "pred_should_abstain",
                "type_match",
                "abstain_match",
                "judge_correct_type",
                "judge_correct_abstain",
                "grounded",
                "behavior_adherence",
                "score",
                "run_id",
                "timestamp",
            ],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "id": rec.get("id", ""),
                    "query": rec.get("query", ""),
                    "gt_conflict_type": rec.get("ground_truth", {}).get("conflict_type", ""),
                    "pred_conflict_type": rec.get("pipeline_output", {}).get("conflict_type", ""),
                    "gt_should_abstain": rec.get("ground_truth", {}).get("should_abstain", False),
                    "pred_should_abstain": rec.get("pipeline_output", {}).get("abstain", False),
                    "type_match": rec.get("rule_based_eval", {}).get("type_match", False),
                    "abstain_match": rec.get("rule_based_eval", {}).get("abstain_match", False),
                    "judge_correct_type": rec.get("judge_output", {}).get("correct_type", False),
                    "judge_correct_abstain": rec.get("judge_output", {}).get("correct_abstain", False),
                    "grounded": rec.get("judge_output", {}).get("grounded", False),
                    "behavior_adherence": rec.get("judge_output", {}).get("behavior_adherence", False),
                    "score": rec.get("judge_output", {}).get("score", 0.0),
                    "run_id": rec.get("metadata", {}).get("run_id", ""),
                    "timestamp": rec.get("metadata", {}).get("timestamp", ""),
                }
            )

    print(f"[OUTPUT] Wrote detailed results: {jsonl_path}")
    print(f"[OUTPUT] Wrote table summary: {summary_path}")


def normalize_documents(raw_docs: Any) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if not isinstance(raw_docs, list):
        return docs

    for i, d in enumerate(raw_docs):
        if isinstance(d, str):
            docs.append({"doc_id": f"doc_{i}", "text": d})
            continue

        if isinstance(d, dict):
            doc_id = (
                d.get("doc_id")
                or d.get("id")
                or d.get("document_id")
                or f"doc_{i}"
            )
            text = (
                d.get("snippet")
                or d.get("text")
                or d.get("content")
                or d.get("document")
                or d.get("passage")
                or ""
            )
            docs.append(
                {
                    "doc_id": str(doc_id),
                    "text": str(text),
                    "source_url": str(d.get("source_url", "")),
                    "timestamp": str(d.get("timestamp", "")),
                }
            )
    return docs

def infer_expected_abstain(sample: Dict[str, Any], source_name: str) -> bool:
    for key in ["expected_abstain", "should_abstain", "abstain", "refusal", "label_abstain"]:
        if key in sample:
            return bool(sample[key])
    return "refusal" in source_name.lower()


def infer_ground_truth_conflict_type(sample: Dict[str, Any], source_name: str) -> str:
    if isinstance(sample.get("conflict_type"), str) and sample["conflict_type"].strip():
        return normalize_conflict_type(sample["conflict_type"])
    if "refusal" in source_name.lower():
        return "no_conflict"
    return "no_conflict"


def normalize_conflict_type(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "no conflict": "no_conflict",
        "no_conflict": "no_conflict",
        "complementary information": "complementary",
        "complementary": "complementary",
        "conflicting opinions and research outcomes": "conflicting",
        "conflicting": "conflicting",
        "outdated information": "outdated",
        "outdated": "outdated",
        "misinformation": "misinformation",
    }
    return mapping.get(normalized, normalized.replace(" ", "_"))


def load_jsonl(path: Path, source_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            query = obj.get("query") or obj.get("question") or ""
            docs = normalize_documents(obj.get("retrieved_docs") or obj.get("documents") or [])
            if not query or not docs:
                continue

            obj["query"] = str(query)
            obj["retrieved_docs"] = docs
            obj["_source"] = source_name
            rows.append(obj)

    return rows


def evaluate_rule_based(sample: Dict[str, Any], pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
    predicted_conflict_type = (
        pipeline_output.get("conflict", {}).get("conflict_type", "")
    )
    predicted_conflict_type = normalize_conflict_type(str(predicted_conflict_type))
    predicted_abstain = bool(
        pipeline_output.get("refusal", {}).get("should_abstain", False)
        or pipeline_output.get("adjudication", {}).get("abstain", False)
    )

    gt_conflict_type = infer_ground_truth_conflict_type(sample, sample.get("_source", ""))
    gt_abstain = infer_expected_abstain(sample, sample.get("_source", ""))

    correct_type = predicted_conflict_type == gt_conflict_type
    correct_abstain = predicted_abstain == gt_abstain

    return {
        "predicted_conflict_type": predicted_conflict_type,
        "ground_truth_conflict_type": gt_conflict_type,
        "predicted_abstain": predicted_abstain,
        "ground_truth_abstain": gt_abstain,
        "correct_type": correct_type,
        "correct_abstain": correct_abstain,
        "correct_conflict": correct_type,
    }


def build_llm(provider: str) -> Any:
    """Build LLM instance based on provider choice."""
    if provider == "mock":
        print("[LLM] Initializing deterministic mock provider")
        return MockLLM()
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        print(f"[LLM] Initializing Google Gemini 2.5 Flash")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
        )
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        print(f"[LLM] Initializing OpenRouter with Gemini 2.5 Flash")
        return ChatOpenAI(
            model="google/gemini-2.5-flash",
            temperature=0,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'google', 'openrouter', or 'mock'.")


def print_sample_debug(index: int, query: str, pipeline_output: Dict[str, Any], rule_eval: Dict[str, Any]):
    print("=" * 80)
    print(f"Sample #{index}")
    print(f"Query: {query}")
    print("-" * 80)
    print("Relevance:")
    print(json.dumps(pipeline_output.get("relevance_results", []), indent=2, ensure_ascii=True))
    print("-" * 80)
    print("Extraction:")
    print(json.dumps(pipeline_output.get("extraction_results", []), indent=2, ensure_ascii=True))
    print("-" * 80)
    print("Verdicts:")
    print(json.dumps(pipeline_output.get("verdict_results", []), indent=2, ensure_ascii=True))
    print("-" * 80)
    print("Aggregation:")
    print(json.dumps(pipeline_output.get("aggregation", {}), indent=2, ensure_ascii=True))
    print("-" * 80)
    print("Conflict:")
    print(json.dumps(pipeline_output.get("conflict", {}), indent=2, ensure_ascii=True))
    print("-" * 80)
    print("Refusal:")
    print(json.dumps(pipeline_output.get("refusal", {}), indent=2, ensure_ascii=True))
    print("-" * 80)
    print("Adjudication:")
    print(json.dumps(pipeline_output.get("adjudication", {}), indent=2, ensure_ascii=True))
    print("-" * 80)
    print("Rule-Based Evaluation:")
    print(json.dumps(rule_eval, indent=2, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent RAG conflict pipeline evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of queries to evaluate. Use 0 or negative to evaluate all.",
    )
    parser.add_argument("--run-id", type=str, default="exp_01", help="Run identifier for output files")
    parser.add_argument(
        "--architecture",
        type=str,
        default="debate_v1",
        help="Architecture label stored in output metadata",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "openrouter", "mock"],
        help="LLM provider to use (google, openrouter, or mock). Model will be gemini-2.5-flash unless mock is selected.",
    )
    args = parser.parse_args()

    llm = build_llm(args.provider)

    pipeline = MultiAgentPipeline(llm)
    evaluator = EvaluatorAgent(llm)
    baseline = SingleAgentBaseline(llm)

    root = Path(__file__).resolve().parent
    conflicts_path = root / "conflicts_normalized.jsonl"
    refusals_path = root / "refusals_normalized.jsonl"
    checkpoint_dir = root / "checkpoints" / args.run_id
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    samples = []
    samples.extend(load_jsonl(conflicts_path, "conflicts_normalized"))
    samples.extend(load_jsonl(refusals_path, "refusals_normalized"))

    if not samples:
        print("No valid samples found in conflicts_normalized.jsonl or refusals_normalized.jsonl")
        return

    if args.limit > 0:
        samples = samples[: args.limit]

    total = 0
    correct_type_count = 0
    correct_abstain_count = 0
    evaluator_scores: List[float] = []
    records: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id") or f"sample_{i}")
        print(f"\n[RUN] Processing sample {i}/{len(samples)} (id: {sample_id})")
        
        checkpoint = checkpoint_manager.load_checkpoint(sample_id)
        if checkpoint:
            print(f"[CHECKPOINT] Found existing checkpoint for {sample_id}")
            print(f"[CHECKPOINT] Completed steps: {checkpoint.get('completed_steps', [])}")
        else:
            print(f"[CHECKPOINT] Creating new checkpoint for {sample_id}")
            checkpoint = checkpoint_manager.initialize_checkpoint(
                sample_id, 
                sample.get("query", ""),
                sample.get("retrieved_docs", [])
            )
        
        query = sample["query"]
        docs = sample["retrieved_docs"]

        # Apply embedding-based chunk selection to reduce tokens
        reduced_docs = reduce_documents(query, docs, k=3)
        pipeline_output = pipeline.run(query, reduced_docs, checkpoint_manager=checkpoint_manager, sample_id=sample_id)
        rule_eval = evaluate_rule_based(sample, pipeline_output)

        gt = {
            "conflict_type": rule_eval["ground_truth_conflict_type"],
            "should_abstain": rule_eval["ground_truth_abstain"],
        }

        print(f"[CHECKPOINT] Running evaluator for {sample_id}")
        evaluator_out = evaluator.run(query, docs, pipeline_output, gt, rule_eval)
        evaluator_dict = evaluator_out.model_dump()
        if checkpoint_manager and sample_id:
            evaluator_usage = summarize_usage_events("evaluator", evaluator.consume_usage_events())
            checkpoint_manager.update_checkpoint(
                sample_id,
                "evaluator",
                {"evaluation": evaluator_dict},
                token_usage=evaluator_usage,
            )

        baseline_out = baseline.run(query, docs).model_dump()
        if checkpoint_manager and sample_id:
            baseline_usage = summarize_usage_events("baseline", baseline.consume_usage_events())
            checkpoint_manager.update_checkpoint(
                sample_id,
                "baseline",
                {"baseline": baseline_out},
                token_usage=baseline_usage,
            )

        record = build_result_record(
            sample=sample,
            pipeline_output=pipeline_output,
            rule_eval=rule_eval,
            evaluator_dict=evaluator_dict,
            run_id=args.run_id,
            architecture=args.architecture,
        )
        records.append(record)
        print(f"[CHECKPOINT] Checkpoint finalized for {sample_id}")

        total += 1
        correct_type_count += int(rule_eval["correct_type"])
        correct_abstain_count += int(rule_eval["correct_abstain"])

        try:
            score = float(evaluator_dict.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        evaluator_scores.append(score)

    print("=" * 80)
    print("FINAL SUMMARY")
    print(f"Total samples: {total}")
    print(f"Conflict type accuracy: {correct_type_count}/{total} = {correct_type_count / total:.3f}")
    print(f"Abstain accuracy: {correct_abstain_count}/{total} = {correct_abstain_count / total:.3f}")
    print(f"Average evaluator score: {mean(evaluator_scores) if evaluator_scores else 0.0:.3f}")

    output_dir = root / "outputs"
    write_outputs(output_dir=output_dir, run_id=args.run_id, records=records)


if __name__ == "__main__":
    main()
