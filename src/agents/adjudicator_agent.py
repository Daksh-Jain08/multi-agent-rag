import json
from typing import Any, Dict
from agents.base_agent import BaseAgent
from schemas import AdjudicatorOutput
from prompts_loader import load_prompt
from agents.base_agent import BaseAgent
from schemas import AdjudicatorOutput


class AdjudicatorAgent(BaseAgent):
    def run(self, aggregation: Dict[str, Any], conflict: Dict[str, Any], refusal: Dict[str, Any]) -> AdjudicatorOutput:
        system_prompt = load_prompt("adjudicator_system_prompt.txt")

        user_prompt = f"""Aggregation: {json.dumps(aggregation, ensure_ascii=True)}
Conflict: {json.dumps(conflict, ensure_ascii=True)}
Refusal: {json.dumps(refusal, ensure_ascii=True)}

Return AdjudicatorOutput.""".strip()

        return self._invoke_structured(AdjudicatorOutput, system_prompt, user_prompt)
