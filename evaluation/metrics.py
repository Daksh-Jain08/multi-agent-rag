from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List

METRIC_KEYS = [
    "correctness",
    "groundedness",
    "refusal_correctness",
    "conflict_handling",
]


def normalize_conflict_label(label: str) -> str:
    value = label.strip().lower()
    if not value:
        return "unknown"
    if value == "no conflict":
        return "no_conflict"
    if value in {
        "factual_contradiction",
        "temporal_mismatch",
        "scope_mismatch",
        "methodological_disagreement",
        "source_reliability",
        "insufficient_evidence",
        "ambiguity",
        "other",
    }:
        return value
    if "complement" in value:
        return "complementary_information"
    if "conflict" in value and "opinion" in value:
        return "conflicting_opinions_and_research_outcomes"
    if "temporal" in value:
        return "temporal_mismatch"
    if "scope" in value:
        return "scope_mismatch"
    if "method" in value:
        return "methodological_disagreement"
    if "reliab" in value:
        return "source_reliability"
    if "ambig" in value:
        return "ambiguity"
    if "insufficient" in value:
        return "insufficient_evidence"
    return "other"


def normalize_conflict_family(label: str) -> str:
    value = normalize_conflict_label(label)
    if value in {"no_conflict", "complementary_information"}:
        return value
    if value in {"factual_contradiction", "temporal_mismatch", "scope_mismatch", "methodological_disagreement", "source_reliability"}:
        return "conflicting_opinions_and_research_outcomes"
    if value in {"insufficient_evidence", "ambiguity"}:
        return "insufficient_or_ambiguous"
    return "other"


def extract_model_conflict_type(conflict_report: Dict[str, Any]) -> str:
    conflicts = conflict_report.get("conflicts") or []
    if not conflicts:
        return "no_conflict"
    first_conflict = conflicts[0]
    if isinstance(first_conflict, dict):
        conflict_type = str(first_conflict.get("type") or "other")
        return conflict_type
    return "other"


def is_conflict_present(conflict_report: Dict[str, Any]) -> bool:
    return bool(conflict_report.get("has_conflict", False)) or bool(conflict_report.get("conflicts"))


def derive_benchmark_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    expected_refusal = row.get("expected_refusal")
    final_answer = row.get("final_answer") or {}
    conflict_report = row.get("conflict_report") or {}

    predicted_refusal = bool(final_answer.get("refusal", False))
    gold_refusal = None if expected_refusal is None else bool(expected_refusal)

    gold_conflict_label = normalize_conflict_label(str(row.get("conflict_type") or ""))
    gold_conflict_family = normalize_conflict_family(str(row.get("conflict_type") or ""))
    gold_conflict_present = gold_conflict_label not in {"no_conflict", "unknown"}
    predicted_conflict_present = is_conflict_present(conflict_report)
    predicted_conflict_type = normalize_conflict_label(extract_model_conflict_type(conflict_report))
    predicted_conflict_family = normalize_conflict_family(extract_model_conflict_type(conflict_report))

    metrics = {
        "predicted_refusal": predicted_refusal,
        "predicted_conflict_present": predicted_conflict_present,
        "predicted_conflict_type": predicted_conflict_type,
        "predicted_conflict_family": predicted_conflict_family,
        "gold_conflict_present": gold_conflict_present,
        "gold_conflict_type": gold_conflict_label,
        "gold_conflict_family": gold_conflict_family,
    }

    if gold_refusal is not None:
        metrics["refusal_match"] = 1.0 if predicted_refusal == gold_refusal else 0.0
        metrics["refusal_precision_like"] = 1.0 if predicted_refusal and gold_refusal else 0.0
        metrics["refusal_recall_like"] = 1.0 if gold_refusal and predicted_refusal else 0.0

    metrics["conflict_presence_match"] = 1.0 if predicted_conflict_present == gold_conflict_present else 0.0
    if gold_conflict_label not in {"no_conflict", "unknown"}:
        metrics["conflict_type_match"] = 1.0 if predicted_conflict_type == gold_conflict_label else 0.0
        metrics["conflict_family_match"] = 1.0 if predicted_conflict_family == gold_conflict_family else 0.0
    else:
        metrics["conflict_type_match"] = 1.0 if not predicted_conflict_present else 0.0
        metrics["conflict_family_match"] = 1.0 if not predicted_conflict_present else 0.0

    return metrics


def aggregate_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row.get("dataset_kind", "unknown"))].append(row)

    summary: Dict[str, Any] = {}
    for (arch, dataset_kind), items in grouped.items():
        arch_scores: Dict[str, float] = {}
        for key in METRIC_KEYS:
            values = [float(item.get("judge", {}).get(key, 0.0)) for item in items]
            arch_scores[key] = mean(values) if values else 0.0
        benchmark_keys = ["refusal_match", "conflict_presence_match", "conflict_type_match", "conflict_family_match"]
        benchmark_scores: Dict[str, float] = {}
        for key in benchmark_keys:
            values = [float(item.get("benchmark_metrics", {}).get(key, 0.0)) for item in items]
            benchmark_scores[key] = mean(values) if values else 0.0
        summary.setdefault(arch, {})[dataset_kind] = {
            "num_examples": len(items),
            "judge_metrics": arch_scores,
            "benchmark_metrics": benchmark_scores,
        }
    return summary
