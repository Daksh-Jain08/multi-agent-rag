import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, sample_id: str) -> Path:
        return self.checkpoint_dir / f"sample_{sample_id}.json"

    def load_checkpoint(self, sample_id: str) -> Optional[Dict[str, Any]]:
        path = self.get_checkpoint_path(sample_id)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return None
        return None

    def save_checkpoint(self, sample_id: str, checkpoint_data: Dict[str, Any]):
        path = self.get_checkpoint_path(sample_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=True)

    def initialize_checkpoint(self, sample_id: str, query: str, docs: List[Dict[str, str]]) -> Dict[str, Any]:
        checkpoint = {
            "sample_id": sample_id,
            "query": query,
            "documents": docs,
            "completed_steps": [],
            "intermediate_results": {},
            "token_usage": {
                "overall": {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "by_step": {},
                "api_calls": [],
            },
            "checkpoint_created_at": datetime.utcnow().isoformat(),
            "last_updated_at": datetime.utcnow().isoformat(),
        }
        self.save_checkpoint(sample_id, checkpoint)
        return checkpoint

    def update_checkpoint(
        self,
        sample_id: str,
        step_name: str,
        result: Dict[str, Any],
        token_usage: Optional[Dict[str, Any]] = None,
    ):
        checkpoint = self.load_checkpoint(sample_id)
        if checkpoint is None:
            return

        if step_name not in checkpoint["completed_steps"]:
            checkpoint["completed_steps"].append(step_name)

        checkpoint["intermediate_results"][step_name] = result

        usage_root = checkpoint.setdefault(
            "token_usage",
            {
                "overall": {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "by_step": {},
                "api_calls": [],
            },
        )
        usage_root.setdefault("overall", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        usage_root.setdefault("by_step", {})
        usage_root.setdefault("api_calls", [])

        if token_usage:
            step_summary = token_usage.get("step_summary", {})
            api_calls = token_usage.get("api_calls", [])

            existing_step = usage_root["by_step"].get(
                step_name,
                {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

            existing_step["calls"] = int(existing_step.get("calls", 0)) + int(step_summary.get("calls", 0))
            existing_step["prompt_tokens"] = int(existing_step.get("prompt_tokens", 0)) + int(step_summary.get("prompt_tokens", 0))
            existing_step["completion_tokens"] = int(existing_step.get("completion_tokens", 0)) + int(step_summary.get("completion_tokens", 0))
            existing_step["total_tokens"] = int(existing_step.get("total_tokens", 0)) + int(step_summary.get("total_tokens", 0))
            usage_root["by_step"][step_name] = existing_step

            overall = usage_root["overall"]
            overall["calls"] = int(overall.get("calls", 0)) + int(step_summary.get("calls", 0))
            overall["prompt_tokens"] = int(overall.get("prompt_tokens", 0)) + int(step_summary.get("prompt_tokens", 0))
            overall["completion_tokens"] = int(overall.get("completion_tokens", 0)) + int(step_summary.get("completion_tokens", 0))
            overall["total_tokens"] = int(overall.get("total_tokens", 0)) + int(step_summary.get("total_tokens", 0))

            if isinstance(api_calls, list) and api_calls:
                usage_root["api_calls"].extend(api_calls)

        checkpoint["last_updated_at"] = datetime.utcnow().isoformat()
        self.save_checkpoint(sample_id, checkpoint)

    def has_step(self, sample_id: str, step_name: str) -> bool:
        checkpoint = self.load_checkpoint(sample_id)
        if checkpoint is None:
            return False
        return step_name in checkpoint.get("completed_steps", [])

    def get_step_result(self, sample_id: str, step_name: str) -> Optional[Dict[str, Any]]:
        checkpoint = self.load_checkpoint(sample_id)
        if checkpoint is None:
            return None
        return checkpoint.get("intermediate_results", {}).get(step_name)
