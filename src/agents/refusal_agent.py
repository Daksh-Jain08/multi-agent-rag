import json
from typing import Any, Dict
from agents.base_agent import BaseAgent
from schemas import RefusalDecisionOutput
from prompts_loader import load_prompt
from agents.base_agent import BaseAgent
from schemas import RefusalDecisionOutput


class RefusalDecisionAgent(BaseAgent):
    def run(self, aggregation: Dict[str, Any], conflict: Dict[str, Any]) -> RefusalDecisionOutput:
        system_prompt = load_prompt("refusal_system_prompt.txt")

        user_prompt = f"""Aggregation: {json.dumps(aggregation, ensure_ascii=True)}
Conflict: {json.dumps(conflict, ensure_ascii=True)}

Return RefusalDecisionOutput.""".strip()

        return self._invoke_structured(RefusalDecisionOutput, system_prompt, user_prompt)
