from __future__ import annotations

from typing import Any, Dict

from architectures.base_arch import BaseArchitecture


class DebateArchitecture(BaseArchitecture):
    name = "debate"

    def __init__(self, extractor, relevance, proposer, critic, refiner, refusal_judge) -> None:
        self.extractor = extractor
        self.relevance = relevance
        self.proposer = proposer
        self.critic = critic
        self.refiner = refiner
        self.refusal_judge = refusal_judge

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = self.extractor.run(state)
        state = self.relevance.run(state)

        state = self.proposer.run(state)
        proposed = state.get("final_answer", {})

        state = self.critic.run(state)
        state["summary"] = (
            f"Draft answer: {proposed.get('answer', '')}\n"
            f"Critiques: {state.get('critiques', [])}"
        )

        state = self.refiner.run(state)
        state = self.refusal_judge.run(state)
        return state
