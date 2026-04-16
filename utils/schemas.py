from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    text: str
    source: Optional[str] = None


@dataclass
class EvidenceItem:
    claim: str
    doc_id: str
    snippet: str
    support: str = "unknown"


@dataclass
class RefusalDecision:
    refuse: bool
    reason: str


@dataclass
class FinalAnswer:
    answer: str
    evidence_doc_ids: List[str] = field(default_factory=list)
    refusal: bool = False
    refusal_reason: str = ""


@dataclass
class PipelineState:
    query_id: str
    query: str
    retrieved_docs: List[Document] = field(default_factory=list)
    extracted_evidence: List[EvidenceItem] = field(default_factory=list)
    relevant_evidence: List[EvidenceItem] = field(default_factory=list)
    conflict_report: Dict[str, Any] = field(default_factory=dict)
    critiques: List[str] = field(default_factory=list)
    summary: str = ""
    refusal_decision: Optional[RefusalDecision] = None
    final_answer: Optional[FinalAnswer] = None
    expected_answer: Optional[str] = None
    expected_refusal: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
