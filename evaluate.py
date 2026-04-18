import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ID_PATTERN = re.compile(r"#\d+")


def get_nested(obj: Dict[str, Any], path: List[str]) -> Any:
    current: Any = obj
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def get_first_nested(obj: Dict[str, Any], paths: List[List[str]]) -> Any:
    for path in paths:
        value = get_nested(obj, path)
        if value is not None:
            return value
    return None


def to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    aliases = {
        "no conflict": "no conflict",
        "complementary": "complementary information",
        "complementary information": "complementary information",
        "conflict due to misinformation": "conflict due to misinformation",
        "conflicting opinions and research outcomes": "conflicting opinions and research outcomes",
        "conflicting opinions or research outcomes": "conflicting opinions and research outcomes",
    }
    return aliases.get(text, text)


def normalize_sample_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    match = ID_PATTERN.search(text)
    if match:
        return match.group(0)
    if text.isdigit():
        return f"#{text}"
    return None


def sample_id_from_record(record: Dict[str, Any], file_path: Path) -> Optional[str]:
    for key in ("sample_id", "id"):
        sample_id = normalize_sample_id(record.get(key))
        if sample_id:
            return sample_id

    match = ID_PATTERN.search(file_path.name)
    if match:
        return match.group(0)
    return None


def conflict_requires_abstain(conflict_type: str) -> bool:
    normalized = normalize_label(conflict_type)
    return "conflicting opinions" in normalized or "research outcomes" in normalized


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_jsonl_records(file_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    decode_errors = 0

    if not file_path.exists():
        return records, decode_errors

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            if isinstance(record, dict):
                records.append(record)

    return records, decode_errors


def build_truth_map(conflicts_path: Path, refusals_path: Path) -> Tuple[Dict[str, str], int]:
    truth: Dict[str, str] = {}
    total_decode_errors = 0

    for path in (conflicts_path, refusals_path):
        records, decode_errors = load_jsonl_records(path)
        total_decode_errors += decode_errors

        for record in records:
            sample_id = normalize_sample_id(record.get("id") or record.get("sample_id"))
            conflict_type = record.get("conflict_type")
            if sample_id and isinstance(conflict_type, str):
                truth[sample_id] = conflict_type
            else:
                print(f"Warning: Skipping truth record with missing sample ID or conflict type in {path}: {record}")

    return truth, total_decode_errors


def iter_checkpoint_files(input_dir: Path) -> Iterable[Path]:
    files = sorted(input_dir.glob("sample_#*.json"))
    if files:
        return files
    return sorted(input_dir.glob("*.json"))


def resolve_refusals_path(user_value: Optional[Path]) -> Path:
    if user_value is not None:
        return user_value

    default_candidates = [
        Path("refusals_normalized.jsonl"),
        Path("refusal_normalized.jsonl"),
    ]
    for path in default_candidates:
        if path.exists():
            return path
    return default_candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint sample files against conflict/refusal ground truth JSONL datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("checkpoints/pipeline"),
        help="Folder containing per-sample checkpoint JSON files (default: checkpoints/pipeline).",
    )
    parser.add_argument(
        "--conflicts",
        type=Path,
        default=Path("conflicts_normalized.jsonl"),
        help="Path to conflict ground-truth JSONL file.",
    )
    parser.add_argument(
        "--refusals",
        type=Path,
        default=None,
        help="Path to refusal ground-truth JSONL file (defaults to refusals_normalized.jsonl if present).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics.txt"),
        help="Path to write the metrics report.",
    )
    args = parser.parse_args()

    input_dir = args.input
    conflicts_path = args.conflicts
    refusals_path = resolve_refusals_path(args.refusals)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found or not a directory: {input_dir}")

    truth_by_sample_id, truth_decode_errors = build_truth_map(conflicts_path, refusals_path)
    if not truth_by_sample_id:
        raise SystemExit(
            f"No usable truth rows found in {conflicts_path} and {refusals_path}. "
            "Check file paths and JSONL format."
        )

    checkpoint_files = list(iter_checkpoint_files(input_dir))
    if not checkpoint_files:
        raise SystemExit(f"No checkpoint JSON files found in {input_dir}")

    files_with_parse_errors = 0
    files_missing_sample_id = 0
    files_missing_truth = 0

    evaluated = 0
    conflict_presence_correct = 0
    conflict_type_correct = 0
    abstain_correct = 0
    end_to_end_correct = 0

    missing_eval_correct_conflict = 0
    missing_eval_correct_type = 0
    missing_eval_correct_abstain = 0

    grounded_values: List[bool] = []
    score_values: List[float] = []

    token_prompt_total = 0
    token_completion_total = 0
    token_total_total = 0
    token_prompt_count = 0
    token_completion_count = 0
    token_total_count = 0

    step_token_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"samples": 0, "prompt": 0, "completion": 0, "total": 0, "calls": 0}
    )

    for file_path in checkpoint_files:
        try:
            sample = json.loads(file_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            files_with_parse_errors += 1
            continue

        if not isinstance(sample, dict):
            files_with_parse_errors += 1
            continue

        sample_id = sample_id_from_record(sample, file_path)
        if sample_id is None:
            files_missing_sample_id += 1
            continue

        # Token usage is reported for all parsable checkpoint files.
        token_usage = sample.get("token_usage")
        if isinstance(token_usage, dict):
            overall = token_usage.get("overall")
            if not isinstance(overall, dict):
                overall = token_usage

            prompt_tokens = to_int(overall.get("prompt_tokens"))
            completion_tokens = to_int(overall.get("completion_tokens"))
            total_tokens = to_int(overall.get("total_tokens"))

            if prompt_tokens is not None:
                token_prompt_total += prompt_tokens
                token_prompt_count += 1
            if completion_tokens is not None:
                token_completion_total += completion_tokens
                token_completion_count += 1
            if total_tokens is not None:
                token_total_total += total_tokens
                token_total_count += 1

            by_step = token_usage.get("by_step")
            if isinstance(by_step, dict):
                for step_name, values in by_step.items():
                    if not isinstance(values, dict):
                        continue
                    stats = step_token_stats[step_name]
                    stats["samples"] += 1
                    stats["prompt"] += to_int(values.get("prompt_tokens")) or 0
                    stats["completion"] += to_int(values.get("completion_tokens")) or 0
                    stats["total"] += to_int(values.get("total_tokens")) or 0
                    stats["calls"] += to_int(values.get("calls")) or 0

        gt_conflict_type = truth_by_sample_id.get(sample_id)
        if gt_conflict_type is None:
            files_missing_truth += 1
            continue

        evaluated += 1

        pred_conflict_type = get_first_nested(
            sample,
            [
                ["intermediate_results", "conflict", "conflict", "conflict_type"],
                ["intermediate_results", "conflict", "conflict_type"],
                ["pipeline_output", "conflict_type"],
            ],
        )

        pred_should_abstain = to_bool(
            get_first_nested(
                sample,
                [
                    ["intermediate_results", "refusal", "should_abstain"],
                    ["intermediate_results", "refusal", "refusal", "should_abstain"],
                    ["pipeline_output", "abstain"],
                ],
            )
        )

        score = to_float(
            get_first_nested(
                sample,
                [
                    ["intermediate_results", "evaluator", "evaluation", "score"],
                    ["judge_output", "score"],
                ],
            )
        )
        if score is not None:
            score_values.append(score)

        grounded = to_bool(
            get_first_nested(
                sample,
                [
                    ["intermediate_results", "evaluator", "evaluation", "grounded"],
                    ["judge_output", "grounded"],
                ],
            )
        )
        if grounded is not None:
            grounded_values.append(grounded)

        eval_correct_conflict = to_bool(
            get_first_nested(
                sample,
                [
                    ["intermediate_results", "evaluator", "evaluation", "correct_conflict"],
                    ["judge_output", "correct_conflict"],
                ],
            )
        )
        eval_correct_type = to_bool(
            get_first_nested(
                sample,
                [
                    ["intermediate_results", "evaluator", "evaluation", "correct_type"],
                    ["judge_output", "correct_type"],
                ],
            )
        )
        eval_correct_abstain = to_bool(
            get_first_nested(
                sample,
                [
                    ["intermediate_results", "evaluator", "evaluation", "correct_abstain"],
                    ["judge_output", "correct_abstain"],
                ],
            )
        )

        missing_eval_correct_conflict += int(eval_correct_conflict is None)
        missing_eval_correct_type += int(eval_correct_type is None)
        missing_eval_correct_abstain += int(eval_correct_abstain is None)

        gt_conflict_norm = normalize_label(gt_conflict_type)
        pred_conflict_norm = normalize_label(pred_conflict_type)

        gt_conflict_present = gt_conflict_norm != "no conflict"
        pred_conflict_present = pred_conflict_norm != "" and pred_conflict_norm != "no conflict"

        gt_should_abstain = conflict_requires_abstain(gt_conflict_type)

        computed_conflict_presence_correct = gt_conflict_present == pred_conflict_present
        computed_conflict_type_correct = pred_conflict_norm == gt_conflict_norm
        computed_abstain_correct = pred_should_abstain is not None and pred_should_abstain == gt_should_abstain

        is_conflict_presence_correct = (
            eval_correct_conflict if eval_correct_conflict is not None else computed_conflict_presence_correct
        )
        is_conflict_type_correct = eval_correct_type if eval_correct_type is not None else computed_conflict_type_correct
        is_abstain_correct = eval_correct_abstain if eval_correct_abstain is not None else computed_abstain_correct

        if computed_conflict_presence_correct is False:
            print(f"Conflict presence mismatch for sample {sample_id}: GT present={gt_conflict_present}, Pred present={pred_conflict_present}")
        if computed_conflict_type_correct is False:
            print(f"Conflict type mismatch for sample {sample_id}: GT='{gt_conflict_norm}', Pred='{pred_conflict_norm}'")
        if computed_abstain_correct is False:
            print(f"Abstain mismatch for sample {sample_id}: GT should_abstain={gt_should_abstain}, Pred should_abstain={pred_should_abstain}")
        print("")
        conflict_presence_correct += int(is_conflict_presence_correct)
        conflict_type_correct += int(is_conflict_type_correct)
        abstain_correct += int(is_abstain_correct)
        end_to_end_correct += int(is_conflict_presence_correct and is_conflict_type_correct and is_abstain_correct)

    report_lines: List[str] = []
    report_lines.append(f"Input folder: {input_dir}")
    report_lines.append(f"Conflict truth file: {conflicts_path}")
    report_lines.append(f"Refusal truth file: {refusals_path}")
    report_lines.append(f"Checkpoint files found: {len(checkpoint_files)}")
    report_lines.append(f"Samples evaluated (with truth): {evaluated}")
    report_lines.append(f"Files missing sample ID: {files_missing_sample_id}")
    report_lines.append(f"Files missing ground truth: {files_missing_truth}")
    report_lines.append(f"Files failed to parse: {files_with_parse_errors}")
    report_lines.append(f"Truth JSONL decode errors: {truth_decode_errors}")
    report_lines.append(f"Samples missing evaluator.correct_conflict: {missing_eval_correct_conflict}")
    report_lines.append(f"Samples missing evaluator.correct_type: {missing_eval_correct_type}")
    report_lines.append(f"Samples missing evaluator.correct_abstain: {missing_eval_correct_abstain}")
    report_lines.append("")

    report_lines.append(f"Conflict Presence Accuracy: {safe_rate(conflict_presence_correct, evaluated):.4f}")
    report_lines.append(f"Conflict Type Accuracy: {safe_rate(conflict_type_correct, evaluated):.4f}")
    report_lines.append(f"Should-Abstain Accuracy: {safe_rate(abstain_correct, evaluated):.4f}")
    report_lines.append(f"End-to-End Accuracy: {safe_rate(end_to_end_correct, evaluated):.4f}")
    report_lines.append(
        f"Average Evaluator Score: {(sum(score_values) / len(score_values)) if score_values else 0.0:.4f}"
    )
    report_lines.append(
        f"Grounded Rate: {safe_rate(sum(1 for value in grounded_values if value), len(grounded_values)):.4f}"
    )
    report_lines.append("")

    report_lines.append("Token Metrics (from token_usage.overall):")
    report_lines.append(f"  Total Prompt Tokens: {token_prompt_total}")
    report_lines.append(f"  Total Completion Tokens: {token_completion_total}")
    report_lines.append(f"  Total Tokens: {token_total_total}")
    report_lines.append(
        f"  Avg Prompt Tokens per Sample: {(token_prompt_total / token_prompt_count) if token_prompt_count else 0.0:.2f}"
    )
    report_lines.append(
        f"  Avg Completion Tokens per Sample: {(token_completion_total / token_completion_count) if token_completion_count else 0.0:.2f}"
    )
    report_lines.append(
        f"  Avg Total Tokens per Sample: {(token_total_total / token_total_count) if token_total_count else 0.0:.2f}"
    )

    if step_token_stats:
        report_lines.append("")
        report_lines.append("Token Metrics by Step (averages over samples containing that step):")
        for step_name in sorted(step_token_stats):
            stats = step_token_stats[step_name]
            samples = stats["samples"]
            report_lines.append(
                "  "
                f"{step_name}: "
                f"avg_total={safe_rate(stats['total'], samples):.2f}, "
                f"avg_prompt={safe_rate(stats['prompt'], samples):.2f}, "
                f"avg_completion={safe_rate(stats['completion'], samples):.2f}, "
                f"avg_calls={safe_rate(stats['calls'], samples):.2f}"
            )

    metrics_text = "\n".join(report_lines)
    print(metrics_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(metrics_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
