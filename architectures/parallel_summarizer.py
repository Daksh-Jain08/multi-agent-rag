from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Dict

from architectures.base_arch import BaseArchitecture


class ParallelSummarizerArchitecture(BaseArchitecture):
    name = "parallel_summarizer"

    def __init__(self, extractor, relevance, conflict, critic, summarizer, synthesizer, refusal_judge) -> None:
        self.extractor = extractor
        self.relevance = relevance
        self.conflict = conflict
        self.critic = critic
        self.summarizer = summarizer
        self.synthesizer = synthesizer
        self.refusal_judge = refusal_judge

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state = self.extractor.run(state)

        with ThreadPoolExecutor(max_workers=2) as ex:
            relevance_future = ex.submit(self.relevance.run, deepcopy(state))
            conflict_future = ex.submit(self.conflict.run, deepcopy(state))
            relevance_state = relevance_future.result()
            conflict_state = conflict_future.result()

        state["relevant_evidence"] = relevance_state.get("relevant_evidence", [])
        state["conflict_report"] = conflict_state.get("conflict_report", {})
        state.setdefault("traces", []).extend(relevance_state.get("traces", [])[-1:])
        state.setdefault("traces", []).extend(conflict_state.get("traces", [])[-1:])

        state = self.critic.run(state)
        state = self.summarizer.run(state)
        state = self.synthesizer.run(state)
        state = self.refusal_judge.run(state)
        return state
