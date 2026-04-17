import json
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from schemas import AggregatorOutput
from prompts_loader import load_prompt
from agents.base_agent import BaseAgent
from schemas import AggregatorOutput


class EvidenceAggregator(BaseAgent):
    def run(self, extractions: List[Dict[str, Any]], verdicts: List[Dict[str, Any]]) -> AggregatorOutput:
        system_prompt = load_prompt("aggregator_system_prompt.txt")

        user_prompt = f"""Extractions: {json.dumps(extractions, ensure_ascii=True)}
Verdicts: {json.dumps(verdicts, ensure_ascii=True)}

Return AggregatorOutput.""".strip()

        return self._invoke_structured(AggregatorOutput, system_prompt, user_prompt)
