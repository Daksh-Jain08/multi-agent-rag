import json
from typing import Dict, List
from agents.base_agent import BaseAgent
from schemas import RelevanceOutput
from prompts_loader import load_prompt


class RelevanceAgent(BaseAgent):
    def run(self, query: str, docs: List[Dict[str, str]]) -> RelevanceOutput:
        system_prompt = load_prompt("relevance_system_prompt.txt")

        user_prompt = f"""Query: {query}
Documents: {json.dumps(docs, ensure_ascii=True)}

Return RelevanceOutput results for every document.""".strip()

        return self._invoke_structured(RelevanceOutput, system_prompt, user_prompt)
