import json
from typing import Dict, List
from agents.base_agent import BaseAgent
from schemas import AdjudicatorOutput
from prompts_loader import load_prompt
from agents.base_agent import BaseAgent
from schemas import AdjudicatorOutput


class SingleAgentBaseline(BaseAgent):
    def run(self, query: str, docs: List[Dict[str, str]]) -> AdjudicatorOutput:
        system_prompt = load_prompt("baseline_system_prompt.txt")

        user_prompt = f"""Query: {query}
Documents: {json.dumps(docs[:3], ensure_ascii=True)}
Return AdjudicatorOutput.""".strip()

        return self._invoke_structured(AdjudicatorOutput, system_prompt, user_prompt)
