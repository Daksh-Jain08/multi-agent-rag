import argparse
from typing import Dict

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from outputs/results.jsonl")
    parser.add_argument("--input", type=str, default="outputs/results.jsonl", help="Path to the JSONL results file")
    parser.add_argument("--output", type=str, default="outputs/metrics.txt", help="Path to write the metrics output")
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)

    df["pred_conflict"] = df["prediction"].apply(lambda x: x["conflict_type"] != "no_conflict")
    df["gt_conflict"] = df["ground_truth"].apply(lambda x: x["conflict_type"] != "no_conflict")
    conflict_presence_acc = (df["pred_conflict"] == df["gt_conflict"]).mean()

    df["type_correct"] = df.apply(
        lambda x: x["prediction"]["conflict_type"] == x["ground_truth"]["conflict_type"],
        axis=1,
    )
    type_acc = df["type_correct"].mean()

    df["abstain_correct"] = df.apply(
        lambda x: x["prediction"]["abstain"] == x["ground_truth"]["should_abstain"],
        axis=1,
    )
    abstain_acc = df["abstain_correct"].mean()

    df["e2e"] = df["type_correct"] & df["abstain_correct"] & (df["pred_conflict"] == df["gt_conflict"])
    e2e_acc = df["e2e"].mean()

    df["partial_score"] = (
        df["type_correct"].astype(int)
        + df["abstain_correct"].astype(int)
        + (df["pred_conflict"] == df["gt_conflict"]).astype(int)
    ) / 3
    partial_score = df["partial_score"].mean()

    df["grounded"] = df["judge"].apply(lambda x: x["grounded"])
    gc_score = df["grounded"].mean()

    judge_score = df["judge"].apply(lambda x: x["score"]).mean()
    avg_tokens = df["tokens"].apply(lambda x: x["total"]).mean()

    df["num_docs"] = df["documents"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["tokens_per_doc"] = df.apply(
        lambda row: row["tokens"]["total"] / row["num_docs"] if row["num_docs"] > 0 else 0,
        axis=1,
    )
    avg_tokens_per_doc = df["tokens_per_doc"].mean()

    per_agent_tokens: Dict[str, list] = {}
    for row in df["tokens"]:
        for agent, val in row["per_agent"].items():
            per_agent_tokens.setdefault(agent, []).append(val)

    avg_per_agent = {k: sum(v) / len(v) for k, v in per_agent_tokens.items()}

    metrics_output = []
    metrics_output.append(f"Conflict Presence Accuracy: {conflict_presence_acc:.4f}")
    metrics_output.append(f"Conflict Type Accuracy: {type_acc:.4f}")
    metrics_output.append(f"Abstain Accuracy: {abstain_acc:.4f}")
    metrics_output.append(f"End-to-End Accuracy: {e2e_acc:.4f}")
    metrics_output.append(f"Partial Credit Score: {partial_score:.4f}")
    metrics_output.append(f"Grounded Citation Score: {gc_score:.4f}")
    metrics_output.append(f"LLM Judge Score: {judge_score:.4f}")
    metrics_output.append(f"Avg Tokens per Sample: {avg_tokens:.2f}")
    metrics_output.append(f"Avg Tokens per Doc per Sample: {avg_tokens_per_doc:.2f}")
    metrics_output.append(f"Tokens per Agent: {avg_per_agent}")

    metrics_text = "\n".join(metrics_output)
    
    print(metrics_text)
    
    with open(args.output, "w") as f:
        f.write(metrics_text)


if __name__ == "__main__":
    main()
