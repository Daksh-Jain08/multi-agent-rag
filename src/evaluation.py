import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def log_result(record: Dict[str, Any], file_path: str = "outputs/results.jsonl") -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_result_record(
    sample_id: str,
    query: str,
    documents: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    prediction: Dict[str, Any],
    intermediate: Dict[str, Any],
    judge: Dict[str, Any],
    tokens: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "query": query,
        "documents": documents,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "intermediate": intermediate,
        "judge": judge,
        "tokens": tokens,
    }


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
    predicted_conflict_type = pipeline_output.get("conflict", {}).get("conflict_type", "")
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
