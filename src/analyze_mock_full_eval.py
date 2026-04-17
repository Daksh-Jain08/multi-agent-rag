import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def normalize_conflict_type(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
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
    return mapping.get(normalized, normalized.replace(" ", "_") or "unknown")


def load_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def is_conflict_present(conflict_type: str) -> bool:
    return conflict_type not in {"no_conflict", "unknown", ""}


def get_predicted_conflict_type(row: Dict[str, Any]) -> str:
    return normalize_conflict_type(row.get("pipeline_output", {}).get("conflict_type", ""))


def get_ground_truth_conflict_type(row: Dict[str, Any]) -> str:
    return normalize_conflict_type(row.get("ground_truth", {}).get("conflict_type", ""))


def get_ground_truth_abstain(row: Dict[str, Any]) -> bool:
    return bool(row.get("ground_truth", {}).get("should_abstain", False))


def get_predicted_abstain(row: Dict[str, Any]) -> bool:
    return bool(row.get("pipeline_output", {}).get("abstain", False))


def get_score(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("judge_output", {}).get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def safe_rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def build_report_lines(rows: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []

    def add_section(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    total = len(rows)
    gt_types = [get_ground_truth_conflict_type(row) for row in rows]
    pred_types = [get_predicted_conflict_type(row) for row in rows]
    gt_abstain = [get_ground_truth_abstain(row) for row in rows]
    pred_abstain = [get_predicted_abstain(row) for row in rows]
    scores = [get_score(row) for row in rows]

    gt_conflict_present = [is_conflict_present(value) for value in gt_types]
    pred_conflict_present = [is_conflict_present(value) for value in pred_types]

    conflict_presence_correct = [gt == pred for gt, pred in zip(gt_conflict_present, pred_conflict_present)]
    conflict_type_correct = [gt == pred for gt, pred in zip(gt_types, pred_types)]
    abstain_correct = [gt == pred for gt, pred in zip(gt_abstain, pred_abstain)]

    support_total = sum(gt_conflict_present)
    no_conflict_total = total - support_total
    predicted_conflict_total = sum(pred_conflict_present)

    true_positive_conflict_presence = sum(1 for gt, pred in zip(gt_conflict_present, pred_conflict_present) if gt and pred)
    false_positive_conflict_presence = sum(1 for gt, pred in zip(gt_conflict_present, pred_conflict_present) if not gt and pred)
    false_negative_conflict_presence = sum(1 for gt, pred in zip(gt_conflict_present, pred_conflict_present) if gt and not pred)

    add_section("Overall Metrics")
    lines.append(f"Total samples: {total}")
    lines.append(f"Ground-truth conflict presence rate: {safe_rate(support_total, total):.3f}")
    lines.append(f"Predicted conflict presence rate: {safe_rate(predicted_conflict_total, total):.3f}")
    lines.append(f"Conflict presence accuracy: {safe_rate(sum(conflict_presence_correct), total):.3f}")
    lines.append(f"Conflict presence detected (recall): {safe_rate(true_positive_conflict_presence, support_total):.3f}")
    lines.append(f"Conflict presence precision: {safe_rate(true_positive_conflict_presence, predicted_conflict_total):.3f}")
    lines.append(f"Conflict presence F1: {safe_rate(2 * true_positive_conflict_presence, 2 * true_positive_conflict_presence + false_positive_conflict_presence + false_negative_conflict_presence):.3f}")
    lines.append(f"Conflict type correctly identified: {safe_rate(sum(conflict_type_correct), total):.3f}")
    lines.append(f"Conflict type accuracy conditioned on conflict present: {safe_rate(sum(1 for gt, ok in zip(gt_conflict_present, conflict_type_correct) if gt and ok), support_total):.3f}")
    lines.append(f"Abstain correct: {safe_rate(sum(abstain_correct), total):.3f}")
    lines.append(f"Average score: {mean(scores):.3f}")
    lines.append(f"Average score on conflict samples: {mean([score for score, present in zip(scores, gt_conflict_present) if present]) if support_total else 0.0:.3f}")
    lines.append(f"Average score on no-conflict samples: {mean([score for score, present in zip(scores, gt_conflict_present) if not present]) if no_conflict_total else 0.0:.3f}")

    add_section("Breakdown By Ground-Truth Conflict Type")
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row, gt_type in zip(rows, gt_types):
        by_type[gt_type].append(row)

    for conflict_type in sorted(by_type.keys()):
        subset = by_type[conflict_type]
        subset_total = len(subset)
        subset_gt_present = [is_conflict_present(get_ground_truth_conflict_type(row)) for row in subset]
        subset_pred_present = [is_conflict_present(get_predicted_conflict_type(row)) for row in subset]
        subset_conflict_type_correct = [get_ground_truth_conflict_type(row) == get_predicted_conflict_type(row) for row in subset]
        subset_abstain_correct = [get_ground_truth_abstain(row) == get_predicted_abstain(row) for row in subset]
        subset_scores = [get_score(row) for row in subset]

        lines.append(f"{conflict_type}: {subset_total} samples")
        lines.append(f"  conflict presence accuracy: {safe_rate(sum(gt == pred for gt, pred in zip(subset_gt_present, subset_pred_present)), subset_total):.3f}")
        lines.append(f"  conflict type accuracy: {safe_rate(sum(subset_conflict_type_correct), subset_total):.3f}")
        lines.append(f"  abstain accuracy: {safe_rate(sum(subset_abstain_correct), subset_total):.3f}")
        lines.append(f"  average score: {mean(subset_scores):.3f}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mock_full_eval_results.jsonl and print useful metrics")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "mock_full_eval_results.jsonl",
        help="Path to the JSONL file produced by the mock evaluation run.",
    )
    parser.add_argument(
        "--by-predicted",
        action="store_true",
        help="Also report average score by predicted conflict type instead of only ground-truth type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "mock_full_eval_metrics.txt",
        help="Path to write the metrics report.",
    )
    args = parser.parse_args()

    rows = load_results(args.input)
    if not rows:
        raise SystemExit(f"No valid records found in {args.input}")

    report_lines = build_report_lines(rows)

    if args.by_predicted:
        pred_types = [get_predicted_conflict_type(row) for row in rows]
        predicted_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row, pred_type in zip(rows, pred_types):
            predicted_by_type[pred_type].append(row)

        report_lines.append("")
        report_lines.append("Breakdown By Predicted Conflict Type")
        report_lines.append("-" * len("Breakdown By Predicted Conflict Type"))
        for conflict_type in sorted(predicted_by_type.keys()):
            subset = predicted_by_type[conflict_type]
            subset_total = len(subset)
            subset_scores = [get_score(row) for row in subset]
            report_lines.append(f"{conflict_type}: {subset_total} samples")
            report_lines.append(f"  average score: {mean(subset_scores):.3f}")
            report_lines.append(f"  abstain rate: {safe_rate(sum(get_predicted_abstain(row) for row in subset), subset_total):.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()