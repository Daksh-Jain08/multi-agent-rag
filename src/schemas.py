from typing import Any, Dict, List
from pydantic import BaseModel, Field


class RelevanceItem(BaseModel):
    doc_id: str
    relevance_score: float
    is_relevant: bool
    reason: str = Field(description="Under 60 words")


class ExtractorItem(BaseModel):
    doc_id: str
    key_facts: List[str]
    reason: str = Field(description="Under 60 words")


class VerdictItem(BaseModel):
    doc_id: str
    verdict: str
    reason: str = Field(description="Under 60 words")
    confidence: float


class RelevanceOutput(BaseModel):
    results: List[RelevanceItem]


class ExtractorOutput(BaseModel):
    results: List[ExtractorItem]


class VerdictOutput(BaseModel):
    results: List[VerdictItem]


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
