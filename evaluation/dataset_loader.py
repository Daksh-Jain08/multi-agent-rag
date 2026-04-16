from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.io_utils import load_jsonl
from utils.schemas import Document


@dataclass
class DatasetExample:
    query_id: str
    query: str
    dataset_name: str = ""
    dataset_kind: str = "unknown"
    conflict_type: str = ""
    gold_answer: str = ""
    expected_refusal: Optional[bool] = None
    docs: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["docs"] = [asdict(doc) for doc in self.docs]
        return data


def _to_docs(record: Dict[str, Any]) -> List[Document]:
    raw_docs = (
        record.get("retrieved_docs")
        or
        record.get("context_docs")
        or record.get("documents")
        or record.get("docs")
        or record.get("contexts")
        or []
    )
    docs: List[Document] = []
    for i, item in enumerate(raw_docs):
        if isinstance(item, str):
            docs.append(Document(doc_id=f"doc_{i}", text=item))
            continue

        if not isinstance(item, dict):
            docs.append(Document(doc_id=f"doc_{i}", text=str(item)))
            continue

        doc_id = str(item.get("doc_id") or item.get("id") or f"doc_{i}")
        text = str(item.get("text") or item.get("content") or item.get("snippet") or "")
        source = item.get("source") or item.get("source_url")
        docs.append(Document(doc_id=doc_id, text=text, source=source))
    return docs


def _infer_dataset_kind(dataset_name: str) -> str:
    lower = dataset_name.lower()
    if "refusal" in lower:
        return "refusals"
    if "conflict" in lower:
        return "conflicts"
    return "unknown"


def normalize_record(record: Dict[str, Any], idx: int, dataset_name: str, dataset_kind: str) -> DatasetExample:
    query_id = str(record.get("id") or record.get("query_id") or f"example_{idx}")
    query = str(record.get("query") or record.get("question") or "")
    gold_answer = str(record.get("gold_answer") or record.get("expected_answer") or record.get("answer") or "")
    conflict_type = str(record.get("conflict_type") or "")
    expected_refusal_raw = record.get("expected_refusal")
    if expected_refusal_raw is None:
        if dataset_kind == "refusals":
            expected_refusal_raw = True
        elif dataset_kind == "conflicts":
            expected_refusal_raw = False
    if expected_refusal_raw is None and "label" in record:
        label = str(record.get("label", "")).lower()
        if "refus" in label:
            expected_refusal_raw = True
    expected_refusal = None if expected_refusal_raw is None else bool(expected_refusal_raw)

    docs = _to_docs(record)
    meta = {
        k: v
        for k, v in record.items()
        if k
        not in {
            "id",
            "query_id",
            "query",
            "question",
            "gold_answer",
            "expected_answer",
            "answer",
            "expected_refusal",
            "conflict_type",
            "retrieved_docs",
            "context_docs",
            "documents",
            "docs",
            "contexts",
        }
    }

    return DatasetExample(
        query_id=query_id,
        query=query,
        dataset_name=dataset_name,
        dataset_kind=dataset_kind,
        conflict_type=conflict_type,
        gold_answer=gold_answer,
        expected_refusal=expected_refusal,
        docs=docs,
        metadata=meta,
    )


def load_examples(path: str) -> List[DatasetExample]:
    rows = load_jsonl(path)
    dataset_name = Path(path).name
    dataset_kind = _infer_dataset_kind(dataset_name)
    return [normalize_record(row, idx=i, dataset_name=dataset_name, dataset_kind=dataset_kind) for i, row in enumerate(rows)]
