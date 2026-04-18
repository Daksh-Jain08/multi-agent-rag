import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Set, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agents import (
    AdjudicatorAgent,
    ConflictDetectionAgent,
    EvaluatorAgent,
    EvidenceAggregator,
    ExtractorAgent,
    RelevanceAgent,
    RefusalDecisionAgent,
    SingleAgentBaseline,
    VerdictAgent,
)
from checkpoint_manager import CheckpointManager
from evaluation import build_result_record, evaluate_rule_based, load_jsonl, log_result, normalize_conflict_type
from mock_model import MockLLM


class PipelineState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, str]]
    token_usage: Dict[str, Any]
    relevance_results: List[Dict[str, Any]]
    relevant_docs: List[Dict[str, str]]
    extraction_results: List[Dict[str, Any]]
    verdict_results: List[Dict[str, Any]]
    aggregation: Dict[str, Any]
    conflict: Dict[str, Any]
    refusal: Dict[str, Any]
    adjudication: Dict[str, Any]


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


def _empty_token_usage() -> Dict[str, Any]:
    return {
        "total": 0,
        "per_agent": {
            "baseline": 0,
            "relevance": 0,
            "extractor": 0,
            "verdict": 0,
            "aggregator": 0,
            "conflict": 0,
            "refusal": 0,
            "adjudicator": 0,
        },
    }


def _merge_token_usage(token_usage: Dict[str, Any], agent_name: str, usage_payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "total": int(token_usage.get("total", 0)),
        "per_agent": dict(token_usage.get("per_agent", {})),
    }
    step_summary = usage_payload.get("step_summary", {}) if isinstance(usage_payload, dict) else {}
    step_total = int(step_summary.get("total_tokens", 0))
    merged["per_agent"][agent_name] = int(merged["per_agent"].get(agent_name, 0)) + step_total
    merged["total"] += step_total
    return merged


def _load_existing_result_ids(results_path: Path) -> Set[str]:
    if not results_path.exists():
        return set()

    ids: Set[str] = set()
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = str(record.get("id", "")).strip()
            if sample_id:
                ids.add(sample_id)
    return ids


def _is_checkpoint_complete(checkpoint: Optional[Dict[str, Any]]) -> bool:
    if not checkpoint:
        return False
    completed_steps = set(checkpoint.get("completed_steps", []))
    required_multi_steps = {
        "relevance",
        "extraction",
        "verdict",
        "aggregation",
        "conflict",
        "refusal",
        "adjudication",
        "evaluator",
    }
    required_baseline_steps = {
        "baseline",
        "evaluator",
    }
    return required_multi_steps.issubset(completed_steps) or required_baseline_steps.issubset(completed_steps)


def _slice_samples_for_batch(samples: List[Dict[str, Any]], batch_size: int, batch_index: int) -> List[Dict[str, Any]]:
    if batch_size <= 0:
        return samples
    start = batch_size * batch_index
    end = start + batch_size
    return samples[start:end]

class MultiAgentPipeline:
    def __init__(self, llm: Any):
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

    def run(
        self,
        query: str,
        docs: List[Dict[str, str]],
        checkpoint_manager: Optional[CheckpointManager] = None,
        sample_id: str = "",
    ) -> Dict[str, Any]:
        self._log(f"Starting graph run for query: {query[:120]}")
        self.checkpoint_manager = checkpoint_manager
        self.sample_id = sample_id

        initial_state: PipelineState = {
            "query": query,
            "retrieved_docs": docs,
            "token_usage": _empty_token_usage(),
            "relevant_docs": [],
        }
        output = self.graph.invoke(initial_state)
        self._log("Completed graph run")
        return output

    def _relevance_node(self, state: PipelineState) -> Dict[str, Any]:
        query = state["query"]
        docs = state["retrieved_docs"]
        self._log(f"Relevance step started for {len(docs)} docs")
        out = self.relevance_agent.run(query, docs)
        results = [item.model_dump() for item in out.results]
        usage_payload = summarize_usage_events("relevance", self.relevance_agent.consume_usage_events())
        token_usage = _merge_token_usage(state.get("token_usage", _empty_token_usage()), "relevance", usage_payload)
        self._log("Relevance step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "relevance",
                {"relevance_results": results},
                token_usage=usage_payload,
            )
        return {"relevance_results": results, "relevant_docs": docs, "token_usage": token_usage}

    def _extract_node(self, state: PipelineState) -> Dict[str, Any]:
        query = state["query"]
        docs = state.get("relevant_docs", [])
        self._log(f"Extraction step started for {len(docs)} relevant docs")
        out = self.extractor_agent.run(query, docs)
        results = [item.model_dump() for item in out.results]
        usage_payload = summarize_usage_events("extraction", self.extractor_agent.consume_usage_events())
        token_usage = _merge_token_usage(state.get("token_usage", _empty_token_usage()), "extractor", usage_payload)
        self._log("Extraction step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "extraction",
                {"extraction_results": results},
                token_usage=usage_payload,
            )
        return {"extraction_results": results, "token_usage": token_usage}

    def _verdict_node(self, state: PipelineState) -> Dict[str, Any]:
        extractions = state.get("extraction_results", [])
        self._log(f"Verdict step started for {len(extractions)} docs")
        out = self.verdict_agent.run(state["query"], extractions)
        verdicts = [item.model_dump() for item in out.results]

        usage_payload = summarize_usage_events("verdict", self.verdict_agent.consume_usage_events())
        token_usage = _merge_token_usage(state.get("token_usage", _empty_token_usage()), "verdict", usage_payload)
        self._log("Verdict step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "verdict",
                {"verdict_results": verdicts},
                token_usage=usage_payload,
            )
        return {"verdict_results": verdicts, "token_usage": token_usage}

    def _aggregate_node(self, state: PipelineState) -> Dict[str, Any]:
        self._log("Aggregation step started")
        out = self.aggregator_agent.run(
            state.get("extraction_results", []),
            state.get("verdict_results", []),
        )
        usage_payload = summarize_usage_events("aggregation", self.aggregator_agent.consume_usage_events())
        token_usage = _merge_token_usage(state.get("token_usage", _empty_token_usage()), "aggregator", usage_payload)
        self._log("Aggregation step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "aggregation",
                {"aggregation": out.model_dump()},
                token_usage=usage_payload,
            )
        return {"aggregation": out.model_dump(), "token_usage": token_usage}

    def _conflict_node(self, state: PipelineState) -> Dict[str, Any]:
        self._log("Conflict detection step started")
        out = self.conflict_agent.run(
            state.get("extraction_results", []),
            state.get("verdict_results", []),
            state.get("aggregation", {}),
        )
        usage_payload = summarize_usage_events("conflict", self.conflict_agent.consume_usage_events())
        token_usage = _merge_token_usage(state.get("token_usage", _empty_token_usage()), "conflict", usage_payload)
        self._log("Conflict detection step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "conflict",
                {"conflict": out.model_dump()},
                token_usage=usage_payload,
            )
        return {"conflict": out.model_dump(), "token_usage": token_usage}

    def _refusal_node(self, state: PipelineState) -> Dict[str, Any]:
        self._log("Refusal decision step started")
        out = self.refusal_agent.run(
            state.get("aggregation", {}),
            state.get("conflict", {}),
        )
        usage_payload = summarize_usage_events("refusal", self.refusal_agent.consume_usage_events())
        token_usage = _merge_token_usage(state.get("token_usage", _empty_token_usage()), "refusal", usage_payload)
        self._log("Refusal decision step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "refusal",
                {"refusal": out.model_dump()},
                token_usage=usage_payload,
            )
        return {"refusal": out.model_dump(), "token_usage": token_usage}

    def _adjudicate_node(self, state: PipelineState) -> Dict[str, Any]:
        self._log("Adjudication step started")
        out = self.adjudicator_agent.run(
            state.get("aggregation", {}),
            state.get("conflict", {}),
            state.get("refusal", {}),
        )
        usage_payload = summarize_usage_events("adjudication", self.adjudicator_agent.consume_usage_events())
        token_usage = _merge_token_usage(state.get("token_usage", _empty_token_usage()), "adjudicator", usage_payload)
        self._log("Adjudication step completed")
        if self.checkpoint_manager and self.sample_id:
            self.checkpoint_manager.update_checkpoint(
                self.sample_id,
                "adjudication",
                {"adjudication": out.model_dump()},
                token_usage=usage_payload,
            )
        return {"adjudication": out.model_dump(), "token_usage": token_usage}


def build_llm(provider: str) -> Any:
    if provider == "mock":
        print("[LLM] Initializing deterministic mock provider")
        return MockLLM()
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        print("[LLM] Initializing Google Gemini 2.5 Flash")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
        )
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        print("[LLM] Initializing OpenRouter with Gemini 2.5 Flash Lite")
        return ChatOpenAI(
            model="google/gemini-2.5-flash-lite",
            temperature=0,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
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
    parser = argparse.ArgumentParser(description="Run single-agent baseline RAG conflict pipeline evaluation")
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
        default="baseline_v1",
        help="Architecture label stored in output metadata",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "openrouter", "mock"],
        help="LLM provider to use (google, openrouter, or mock). Model will be gemini-2.5-flash unless mock is selected.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Number of samples per batch. Use 0 to process the full selected dataset slice.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Zero-based batch index when --batch-size is set.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="",
        help="Optional path to JSONL results file. Defaults to outputs/{run_id}_results.jsonl.",
    )
    parser.add_argument(
        "--append-results",
        action="store_true",
        help="Append to existing results file instead of truncating.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip samples already completed in checkpoints or already present in results file.",
    )
    args = parser.parse_args()

    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0")
    if args.batch_index < 0:
        raise ValueError("--batch-index must be >= 0")

    llm = build_llm(args.provider)

    baseline = SingleAgentBaseline(llm)
    evaluator = EvaluatorAgent(llm)

    root = Path(__file__).resolve().parent.parent
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

    selected_samples = _slice_samples_for_batch(samples, args.batch_size, args.batch_index)
    if not selected_samples:
        print(
            f"No samples selected for batch (batch_size={args.batch_size}, batch_index={args.batch_index})."
        )
        return

    if args.batch_size > 0:
        start = args.batch_size * args.batch_index
        end = start + len(selected_samples)
        print(f"[BATCH] Processing batch {args.batch_index} with samples[{start}:{end}] ({len(selected_samples)} records)")
    else:
        print(f"[BATCH] Processing full selected set ({len(selected_samples)} records)")

    total = 0
    conflict_presence_correct_count = 0
    conflict_type_correct_count = 0
    abstain_correct_count = 0
    e2e_correct_count = 0
    partial_score_total = 0.0
    judge_scores: List[float] = []

    results_path = Path(args.results_path) if args.results_path else (root / "outputs" / f"{args.run_id}_results.jsonl")
    if not results_path.is_absolute():
        results_path = root / results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists() and not args.append_results:
        results_path.unlink()

    existing_result_ids = _load_existing_result_ids(results_path)
    skipped_samples = 0

    for i, sample in enumerate(selected_samples, start=1):
        sample_id = str(sample.get("id") or f"sample_{i}")
        print(f"\n[RUN] Processing sample {i}/{len(selected_samples)} (id: {sample_id})")

        if args.skip_completed and sample_id in existing_result_ids:
            print(f"[SKIP] Sample {sample_id} already present in results file")
            skipped_samples += 1
            continue

        checkpoint = checkpoint_manager.load_checkpoint(sample_id)
        if checkpoint:
            print(f"[CHECKPOINT] Found existing checkpoint for {sample_id}")
            print(f"[CHECKPOINT] Completed steps: {checkpoint.get('completed_steps', [])}")
            if args.skip_completed and _is_checkpoint_complete(checkpoint):
                print(f"[SKIP] Sample {sample_id} already complete in checkpoint")
                skipped_samples += 1
                continue
        else:
            print(f"[CHECKPOINT] Creating new checkpoint for {sample_id}")
            checkpoint_manager.initialize_checkpoint(
                sample_id,
                sample.get("query", ""),
                sample.get("retrieved_docs", []),
            )

        query = sample["query"]
        docs = sample["retrieved_docs"]

        print(f"[CHECKPOINT] Running baseline for {sample_id}")
        baseline_out = baseline.run(query, docs).model_dump()
        baseline_usage = summarize_usage_events("baseline", baseline.consume_usage_events())
        if checkpoint_manager and sample_id:
            checkpoint_manager.update_checkpoint(
                sample_id,
                "baseline",
                {"baseline": baseline_out},
                token_usage=baseline_usage,
            )

        baseline_conflict_type = normalize_conflict_type(str(baseline_out.get("conflict_type", "no_conflict")))
        baseline_should_abstain = bool(baseline_out.get("should_abstain", False))
        baseline_answer = str(baseline_out.get("final_answer", ""))

        token_usage = _merge_token_usage(_empty_token_usage(), "baseline", baseline_usage)
        pipeline_output = {
            "conflict": {
                "conflict_type": baseline_conflict_type,
                "conflict_reason": "Single-agent baseline output.",
            },
            "refusal": {
                "should_abstain": baseline_should_abstain,
                "reason": "Single-agent baseline output.",
            },
            "adjudication": {
                "answer": baseline_answer,
                "citations": [],
                "abstain": baseline_should_abstain,
                "abstain_reason": (
                    "Insufficient reliable evidence for a grounded answer." if baseline_should_abstain else ""
                ),
                "final_reasoning": "",
            },
            "aggregation": {
                "supporting_docs": [],
                "partial_docs": [],
                "irrelevant_docs": [],
                "evidence_summary": [],
            },
            "verdict_results": [],
            "token_usage": token_usage,
        }
        rule_eval = evaluate_rule_based(sample, pipeline_output)

        answer = str(pipeline_output.get("adjudication", {}).get("answer", ""))
        citations = list(pipeline_output.get("adjudication", {}).get("citations", []))

        print(f"[CHECKPOINT] Running evaluator for {sample_id}")
        evaluator_out = evaluator.run(query, docs, answer, citations)
        evaluator_dict = evaluator_out.model_dump()
        if args.provider == "mock":
            evaluator_dict["grounded"] = bool(answer.strip())
            evaluator_dict["score"] = 1.0 if evaluator_dict["grounded"] else 0.0
            if not evaluator_dict.get("feedback"):
                evaluator_dict["feedback"] = "Mock grounding fallback used for smoke testing."
        if checkpoint_manager and sample_id:
            evaluator_usage = summarize_usage_events("evaluator", evaluator.consume_usage_events())
            checkpoint_manager.update_checkpoint(
                sample_id,
                "evaluator",
                {"evaluation": evaluator_dict},
                token_usage=evaluator_usage,
            )

        predicted_conflict_type = normalize_conflict_type(
            str(pipeline_output.get("conflict", {}).get("conflict_type", "no_conflict"))
        )
        ground_truth_conflict_type = normalize_conflict_type(
            str(sample.get("conflict_type", rule_eval["ground_truth_conflict_type"]))
        )
        predicted_abstain = bool(
            pipeline_output.get("adjudication", {}).get("abstain", False)
            or pipeline_output.get("refusal", {}).get("should_abstain", False)
        )
        ground_truth_abstain = bool(rule_eval["ground_truth_abstain"])

        conflict_presence_correct = (predicted_conflict_type != "no_conflict") == (ground_truth_conflict_type != "no_conflict")
        conflict_type_correct = predicted_conflict_type == ground_truth_conflict_type
        abstain_correct = predicted_abstain == ground_truth_abstain
        e2e_correct = conflict_presence_correct and conflict_type_correct and abstain_correct
        partial_score = (
            int(conflict_presence_correct) + int(conflict_type_correct) + int(abstain_correct)
        ) / 3.0

        conflict_presence_correct_count += int(conflict_presence_correct)
        conflict_type_correct_count += int(conflict_type_correct)
        abstain_correct_count += int(abstain_correct)
        e2e_correct_count += int(e2e_correct)
        partial_score_total += partial_score
        try:
            judge_scores.append(float(evaluator_dict.get("score", 0.0)))
        except (TypeError, ValueError):
            judge_scores.append(0.0)

        token_usage = pipeline_output.get("token_usage", _empty_token_usage())
        record = build_result_record(
            sample_id=sample_id,
            query=query,
            documents=docs,
            ground_truth={
                "conflict_type": ground_truth_conflict_type,
                "should_abstain": ground_truth_abstain,
            },
            prediction={
                "conflict_type": predicted_conflict_type,
                "abstain": predicted_abstain,
                "answer": answer,
                "citations": citations,
            },
            intermediate={
                "verdicts": pipeline_output.get("verdict_results", []),
                "aggregation": pipeline_output.get("aggregation", {}),
                "conflict": pipeline_output.get("conflict", {}),
                "refusal": pipeline_output.get("refusal", {}),
            },
            judge={
                "score": float(evaluator_dict.get("score", 0.0)),
                "feedback": str(evaluator_dict.get("feedback", "")),
                "grounded": bool(evaluator_dict.get("grounded", False)),
            },
            tokens=token_usage,
        )
        log_result(record, str(results_path))
        existing_result_ids.add(sample_id)
        print(f"[CHECKPOINT] Checkpoint finalized for {sample_id}")

        total += 1
        # Keep the old rule-based evaluation available for debugging only.
        _ = rule_eval

    print("=" * 80)
    print("FINAL SUMMARY")
    print(f"Total samples: {total}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Results file: {results_path}")
    if total == 0:
        print("No samples were executed in this run.")
        return
    print(f"Conflict presence accuracy: {conflict_presence_correct_count}/{total} = {conflict_presence_correct_count / total:.3f}")
    print(f"Conflict type accuracy: {conflict_type_correct_count}/{total} = {conflict_type_correct_count / total:.3f}")
    print(f"Abstain accuracy: {abstain_correct_count}/{total} = {abstain_correct_count / total:.3f}")
    print(f"End-to-end accuracy: {e2e_correct_count}/{total} = {e2e_correct_count / total:.3f}")
    print(f"Partial credit score: {partial_score_total / total:.3f}")
    print(f"Average evaluator score: {mean(judge_scores) if judge_scores else 0.0:.3f}")


if __name__ == "__main__":
    main()
