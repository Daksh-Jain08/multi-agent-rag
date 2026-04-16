from __future__ import annotations

import json
from typing import Any, Dict

from prompts.output_schemas import JUDGE_OUTPUT_SCHEMA
from utils.llm import LLMClient

JUDGE_PROMPT = """
You are an impartial evaluator for grounded QA outputs.
Score each dimension in [0,1] and return strict JSON.
Use the benchmark fields to decide the target behavior:
- If expected_refusal is true, reward a justified refusal and penalize answers.
- If expected_refusal is false, reward the best grounded answer supported by retrieved docs.
- If conflict_type is present, judge whether the model handled the conflict sensibly.
Dimensions:
- correctness: semantic correctness against gold_answer if available, otherwise against the evidence.
- groundedness: whether the answer is supported by the retrieved_docs / relevant_evidence.
- refusal_correctness: whether refusal vs answer matches the benchmark expectation.
- conflict_handling: whether the model detected/managed conflict appropriately.
Also include a short reason.
""".strip()


class Judge:
    def __init__(self, model_name: str | None = None) -> None:
        self.llm = LLMClient(model=model_name)

    def score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = json.dumps(payload, ensure_ascii=False)
        try:
            return self.llm.invoke_structured(
                system_prompt=JUDGE_PROMPT,
                user_prompt=user_prompt,
                schema_name="judge_output",
                schema=JUDGE_OUTPUT_SCHEMA,
                temperature=0.0,
            )
        except Exception:
            return {
                "correctness": 0.0,
                "groundedness": 0.0,
                "refusal_correctness": 0.0,
                "conflict_handling": 0.0,
                "reason": "Judge output parsing failed",
            }
