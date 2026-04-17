import json
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from schemas import VerdictOutput
from prompts_loader import load_prompt


class VerdictAgent(BaseAgent):
    def run(self, query: str, extractions: List[Dict[str, Any]]) -> VerdictOutput:
        system_prompt = load_prompt("verdict_system_prompt.txt")

        user_prompt = f"""Query: {query}
Extractions: {json.dumps(extractions, ensure_ascii=True)}

Return VerdictOutput results for every extracted document.""".strip()

        return self._invoke_structured(VerdictOutput, system_prompt, user_prompt)
