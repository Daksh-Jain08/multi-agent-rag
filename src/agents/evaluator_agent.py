import json
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from prompts_loader import load_prompt
from schemas import EvaluatorOutput


class EvaluatorAgent(BaseAgent):
    def run(
        self,
        query: str,
        docs: List[Dict[str, str]],
        answer: str,
        citations: List[str],
    ) -> EvaluatorOutput:
        system_prompt = load_prompt("evaluator_system_prompt.txt")

        user_prompt = f"""Query: {query}
Documents: {json.dumps(docs, ensure_ascii=True)}
Answer: {answer}
Citations: {json.dumps(citations, ensure_ascii=True)}

Return EvaluatorOutput.""".strip()

        return self._invoke_structured(EvaluatorOutput, system_prompt, user_prompt)
