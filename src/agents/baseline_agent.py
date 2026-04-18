import json
from typing import Dict, List
from agents.base_agent import BaseAgent
from schemas import BaselineOutput
from prompts_loader import load_prompt


class SingleAgentBaseline(BaseAgent):
    def run(self, query: str, docs: List[Dict[str, str]]) -> BaselineOutput:
        system_prompt = load_prompt("baseline_system_prompt.txt")

        user_prompt = f"""Query: {query}
Documents: {json.dumps(docs, ensure_ascii=True)}
Return BaselineOutput.""".strip()

        return self._invoke_structured(BaselineOutput, system_prompt, user_prompt)
