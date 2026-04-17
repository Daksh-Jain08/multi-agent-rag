import json
import re
from typing import Any, List
from pydantic import BaseModel


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

    def _build_payload(self, user_prompt: str):
        schema_name = self.schema.__name__

        if schema_name == "RelevanceOutput":
            query = _extract_section(user_prompt, "Query")
            docs = _extract_json_section(user_prompt, "Documents", [])
            results = []
            for doc in docs if isinstance(docs, list) else []:
                if not isinstance(doc, dict):
                    continue
                text = str(doc.get("text", ""))
                doc_id = str(doc.get("doc_id", "doc_0"))
                overlap = _token_overlap_score(query, text)
                is_relevant = overlap >= 2
                score = min(1.0, 0.15 * overlap + 0.1)
                results.append(
                    {
                        "doc_id": doc_id,
                        "relevance_score": score,
                        "is_relevant": is_relevant,
                        "reason": "Token overlap heuristic from query and passage.",
                    }
                )
            return {"results": results}

        if schema_name == "ExtractorOutput":
            docs = _extract_json_section(user_prompt, "Documents", [])
            results = []
            for doc in docs if isinstance(docs, list) else []:
                if not isinstance(doc, dict):
                    continue
                text = str(doc.get("text", "")).strip()
                doc_id = str(doc.get("doc_id", "doc_0"))

                sentences = [s.strip() for s in text.split(".") if s.strip()]
                key_facts = [s[:150] for s in sentences[:3]]

                results.append(
                    {
                        "doc_id": doc_id,
                        "key_facts": key_facts if key_facts else ["No facts extracted."],
                        "reason": "Extracted atomic facts from the document.",
                    }
                )

            return {"results": results}

        if schema_name == "VerdictOutput":
            extractions = _extract_json_section(user_prompt, "Extractions", [])
            results = []
            for extracted in extractions if isinstance(extractions, list) else []:
                if not isinstance(extracted, dict):
                    continue
                key_facts = extracted.get("key_facts", [])
                if not isinstance(key_facts, list):
                    key_facts = []
                doc_id = str(extracted.get("doc_id", "doc_0"))

                has_content = len(key_facts) > 0 and any(f.strip() and "no" not in f.lower() for f in key_facts)

                if has_content:
                    verdict = "supports"
                    confidence = 0.85
                else:
                    verdict = "irrelevant"
                    confidence = 0.5

                results.append(
                    {
                        "doc_id": doc_id,
                        "verdict": verdict,
                        "reason": "Evaluated based on extracted substantive content.",
                        "confidence": confidence,
                    }
                )

            return {"results": results}

        if schema_name == "AggregatorOutput":
            verdicts = _extract_json_section(user_prompt, "Verdicts", [])
            extractions = _extract_json_section(user_prompt, "Extractions", [])
            extraction_map = {
                str(item.get("doc_id", "")): item for item in extractions if isinstance(item, dict)
            }

            supporting_docs: List[str] = []
            partial_docs: List[str] = []
            irrelevant_docs: List[str] = []
            evidence_summary: List[dict] = []

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
                key_facts = item.get("key_facts", [])
                if isinstance(key_facts, list):
                    for key_fact in key_facts:
                        years.update(re.findall(r"\b(?:19|20)\d{2}\b", str(key_fact)))
                else:
                    years.update(re.findall(r"\b(?:19|20)\d{2}\b", str(key_facts)))

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
            answer = _extract_section(user_prompt, "Answer")
            citations = _extract_json_section(user_prompt, "Citations", [])

            grounded = bool(answer.strip())
            behavior_adherence = True
            score = 1.0 if grounded else 0.0
            return {
                "correct_conflict": False,
                "correct_type": False,
                "correct_abstain": False,
                "grounded": grounded,
                "behavior_adherence": behavior_adherence,
                "score": score,
                "feedback": "Deterministic mock judge checked whether answer citations were present.",
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

    def with_structured_output(self, schema: BaseModel):
        return MockStructuredModel(schema)
