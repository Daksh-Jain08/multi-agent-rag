from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from architectures.factory import build_architectures
from evaluation.dataset_loader import load_examples
from utils.io_utils import write_jsonl
from utils.llm import LLMClient
from utils.retrieval import simple_lexical_retrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent RAG runner")
    parser.add_argument("--architecture", default="sequential", choices=[
        "single_agent", "sequential", "debate", "parallel", "parallel_summarizer"
    ])
    parser.add_argument("--query", default=None, help="Single query to run")
    parser.add_argument("--dataset_path", default=None, help="JSONL dataset path for batch")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", default="outputs/runs/latest_run.jsonl")
    return parser.parse_args()


def run_one(architecture_name: str, query_id: str, query: str, docs: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    llm = LLMClient()
    arch_map = build_architectures(llm)
    if architecture_name not in arch_map:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    ranked_docs = simple_lexical_retrieve(query, docs, top_k=top_k)
    state: Dict[str, Any] = {
        "query_id": query_id,
        "query": query,
        "retrieved_docs": [doc.__dict__ for doc in ranked_docs],
        "traces": [],
    }

    final_state = arch_map[architecture_name].run(state)
    return final_state


def run_single(args: argparse.Namespace) -> None:
    if not args.query:
        raise ValueError("--query is required when --dataset_path is not provided")

    final_state = run_one(
        architecture_name=args.architecture,
        query_id="single_query",
        query=args.query,
        docs=[],
        top_k=args.top_k,
    )
    print(final_state.get("final_answer", {}))


def run_batch(args: argparse.Namespace) -> None:
    examples = load_examples(args.dataset_path)
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        final_state = run_one(
            architecture_name=args.architecture,
            query_id=ex.query_id,
            query=ex.query,
            docs=ex.docs,
            top_k=args.top_k,
        )
        rows.append(
            {
                "architecture": args.architecture,
                "query_id": ex.query_id,
                "query": ex.query,
                "expected_answer": ex.expected_answer,
                "expected_refusal": ex.expected_refusal,
                "retrieved_docs": final_state.get("retrieved_docs", []),
                "relevant_evidence": final_state.get("relevant_evidence", []),
                "conflict_report": final_state.get("conflict_report", {}),
                "final_answer": final_state.get("final_answer", {}),
                "traces": final_state.get("traces", []),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), rows)
    print(f"Saved run outputs to {output_path}")


def main() -> None:
    args = parse_args()
    if args.dataset_path:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
