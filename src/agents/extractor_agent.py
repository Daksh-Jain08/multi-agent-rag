import json
from typing import Dict, List
from agents.base_agent import BaseAgent
from schemas import ExtractorOutput
from prompts_loader import load_prompt


class ExtractorAgent(BaseAgent):
    def run(self, query: str, docs: List[Dict[str, str]]) -> ExtractorOutput:
        system_prompt = load_prompt("extractor_system_prompt.txt")

        user_prompt = f"""Query: {query}
Documents: {json.dumps(docs, ensure_ascii=True)}

Return ExtractorOutput results for every document.""".strip()

        return self._invoke_structured(ExtractorOutput, system_prompt, user_prompt)
