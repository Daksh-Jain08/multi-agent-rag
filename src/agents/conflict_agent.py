import json
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from schemas import ConflictDetectionOutput
from prompts_loader import load_prompt
from agents.base_agent import BaseAgent
from schemas import ConflictDetectionOutput


class ConflictDetectionAgent(BaseAgent):
    def run(
        self,
        extractions: List[Dict[str, Any]],
        verdicts: List[Dict[str, Any]],
        aggregation: Dict[str, Any],
    ) -> ConflictDetectionOutput:
        system_prompt = load_prompt("conflict_system_prompt.txt")

        user_prompt = f"""Extractions: {json.dumps(extractions, ensure_ascii=True)}
Verdicts: {json.dumps(verdicts, ensure_ascii=True)}
Aggregation: {json.dumps(aggregation, ensure_ascii=True)}

Return ConflictDetectionOutput.""".strip()

        return self._invoke_structured(ConflictDetectionOutput, system_prompt, user_prompt)
