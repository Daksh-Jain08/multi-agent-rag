from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from architectures.factory import build_architectures
from evaluation.dataset_loader import load_examples
from evaluation.evaluate_architectures import evaluate_examples_with_architectures
from evaluation.judge import Judge
from utils.io_utils import write_json
from utils.llm import LLMClient


DEFAULT_DATASETS = ["conflicts_normalized.jsonl", "refusals_normalized.jsonl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all architectures on the provided benchmark datasets")
    parser.add_argument("--datasets", nargs="*", default=None, help="One or more JSONL dataset paths")
    parser.add_argument(
        "--architectures",
        nargs="*",
        default=None,
        help="Optional architecture names to evaluate (e.g., sequential parallel)",
    )
    parser.add_argument(
        "--max_examples_per_dataset",
        type=int,
        default=0,
        help="Optional cap per dataset file. 0 means no cap.",
    )
    parser.add_argument("--output", default="outputs/eval/summary.json")
    parser.add_argument("--report", default="outputs/eval/summary.md")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--skip_judge", action="store_true", help="Skip LLM-as-judge scoring to reduce API calls")
    return parser.parse_args()


def _resolve_datasets(dataset_args: List[str] | None) -> List[str]:
    if dataset_args:
        return dataset_args

    resolved: List[str] = []
    for candidate in DEFAULT_DATASETS:
        if Path(candidate).exists():
            resolved.append(candidate)
    if not resolved:
        raise FileNotFoundError(
            "No default datasets found. Provide --datasets with paths to conflicts_normalized.jsonl and refusals_normalized.jsonl."
        )
    return resolved


def _render_markdown_report(report: Dict[str, Any]) -> str:
    lines = ["# Multi-Agent RAG Benchmark Summary", ""]
    for architecture, dataset_map in report["summary"].items():
        lines.append(f"## {architecture}")
        lines.append("")
        lines.append("| Dataset | Examples | Judge Correctness | Judge Groundedness | Refusal Match | Conflict Presence Match | Conflict Type Match | Conflict Family Match |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for dataset_kind, stats in dataset_map.items():
            judge_metrics = stats["judge_metrics"]
            benchmark_metrics = stats["benchmark_metrics"]
            lines.append(
                f"| {dataset_kind} | {stats['num_examples']} | {judge_metrics['correctness']:.3f} | {judge_metrics['groundedness']:.3f} | "
                f"{benchmark_metrics['refusal_match']:.3f} | {benchmark_metrics['conflict_presence_match']:.3f} | {benchmark_metrics['conflict_type_match']:.3f} | {benchmark_metrics['conflict_family_match']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_evaluation(
    datasets: List[str],
    output: str,
    report_path: str,
    top_k: int = 5,
    architecture_names: List[str] | None = None,
    max_examples_per_dataset: int = 0,
    skip_judge: bool = False,
) -> Dict[str, Any]:
    dataset_paths = _resolve_datasets(datasets)

    examples = []
    for dataset_path in dataset_paths:
        dataset_examples = load_examples(dataset_path)
        if max_examples_per_dataset > 0:
            dataset_examples = dataset_examples[:max_examples_per_dataset]
        examples.extend(dataset_examples)

    llm = LLMClient()
    all_architectures = build_architectures(llm)
    if architecture_names:
        missing = [name for name in architecture_names if name not in all_architectures]
        if missing:
            raise ValueError(f"Unknown architectures requested: {missing}. Available: {list(all_architectures.keys())}")
        architectures = {name: all_architectures[name] for name in architecture_names}
    else:
        architectures = all_architectures

    judge = None if skip_judge else Judge()

    def run_one_fn(
        architecture_name: str,
        query_id: str,
        query: str,
        docs: List[Any],
        top_k: int,
        architecture,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "query_id": query_id,
            "query": query,
            "retrieved_docs": [doc.__dict__ for doc in docs[:top_k]],
            "traces": [],
        }
        print(f"Running architecture '{architecture_name}' on query_id '{query_id}' with {len(docs)} retrieved docs...")
        return architecture.run(state)

    report = evaluate_examples_with_architectures(
        examples=examples,
        architectures=architectures,
        run_one_fn=run_one_fn,
        top_k=top_k,
        judge=judge,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(output_path), report)

    markdown_path = Path(report_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown_report(report), encoding="utf-8")

    print(f"Saved evaluation summary to {output_path}")
    print(f"Saved markdown report to {markdown_path}")
    return report


def main() -> None:
    args = parse_args()
    run_evaluation(
        datasets=args.datasets,
        output=args.output,
        report_path=args.report,
        top_k=args.top_k,
        architecture_names=args.architectures,
        max_examples_per_dataset=args.max_examples_per_dataset,
        skip_judge=args.skip_judge,
    )


if __name__ == "__main__":
    main()
