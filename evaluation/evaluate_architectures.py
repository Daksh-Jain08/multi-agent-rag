from __future__ import annotations

from typing import Any, Dict, Iterable, List

from evaluation.dataset_loader import DatasetExample
from evaluation.judge import Judge
from evaluation.metrics import aggregate_scores, derive_benchmark_metrics


def _build_judge_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": row.get("query"),
        "dataset_kind": row.get("dataset_kind"),
        "gold_answer": row.get("gold_answer"),
        "conflict_type": row.get("conflict_type"),
        "expected_refusal": row.get("expected_refusal"),
        "model_output": row.get("final_answer"),
        "retrieved_docs": row.get("retrieved_docs", []),
        "relevant_evidence": row.get("relevant_evidence", []),
        "conflict_report": row.get("conflict_report", {}),
        "summary": row.get("summary", ""),
        "critiques": row.get("critiques", []),
        "traces": row.get("traces", []),
    }


def evaluate_runs(run_rows: Iterable[Dict[str, Any]], judge: Judge) -> Dict[str, Any]:
    scored_rows: List[Dict[str, Any]] = []
    for row in run_rows:
        row["benchmark_metrics"] = derive_benchmark_metrics(row)
        row["judge"] = judge.score(_build_judge_payload(row))
        scored_rows.append(row)

    summary = aggregate_scores(scored_rows)
    return {
        "rows": scored_rows,
        "summary": summary,
    }


def evaluate_examples_with_architectures(
    examples: List[DatasetExample],
    architectures: Dict[str, Any],
    run_one_fn,
    top_k: int,
    judge: Judge | None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for architecture_name, architecture in architectures.items():
        for example in examples:
            final_state = run_one_fn(
                architecture_name=architecture_name,
                query_id=example.query_id,
                query=example.query,
                docs=example.docs,
                top_k=top_k,
                architecture=architecture,
            )
            row = {
                "architecture": architecture_name,
                "dataset_name": example.dataset_name,
                "dataset_kind": example.dataset_kind,
                "query_id": example.query_id,
                "query": example.query,
                "gold_answer": example.gold_answer,
                "conflict_type": example.conflict_type,
                "expected_refusal": example.expected_refusal,
                "retrieved_docs": final_state.get("retrieved_docs", []),
                "relevant_evidence": final_state.get("relevant_evidence", []),
                "conflict_report": final_state.get("conflict_report", {}),
                "summary": final_state.get("summary", ""),
                "critiques": final_state.get("critiques", []),
                "final_answer": final_state.get("final_answer", {}),
                "traces": final_state.get("traces", []),
            }
            row["benchmark_metrics"] = derive_benchmark_metrics(row)
            if judge is not None:
                row["judge"] = judge.score(_build_judge_payload(row))
            else:
                row["judge"] = {
                    "correctness": 0.0,
                    "groundedness": 0.0,
                    "refusal_correctness": 0.0,
                    "conflict_handling": 0.0,
                    "reason": "Judge skipped (--skip_judge)",
                }
            rows.append(row)

    return {
        "rows": rows,
        "summary": aggregate_scores(rows),
    }
