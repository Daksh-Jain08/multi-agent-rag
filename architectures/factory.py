from __future__ import annotations

from typing import Dict

from agents.role_agents import (
    CriticAgent,
    ConflictDetectorAgent,
    EvidenceExtractorAgent,
    FinalSynthesizerAgent,
    RefusalJudgeAgent,
    RelevanceClassifierAgent,
    SingleAgentResponder,
    SummarizerAgent,
)
from architectures.debate import DebateArchitecture
from architectures.parallel import ParallelArchitecture
from architectures.parallel_summarizer import ParallelSummarizerArchitecture
from architectures.sequential import SequentialArchitecture
from architectures.single_agent import SingleAgentArchitecture
from utils.llm import LLMClient


def build_architectures(llm: LLMClient) -> Dict[str, object]:
    extractor = EvidenceExtractorAgent(llm)
    relevance = RelevanceClassifierAgent(llm)
    conflict = ConflictDetectorAgent(llm)
    critic = CriticAgent(llm)
    summarizer = SummarizerAgent(llm)
    synthesizer = FinalSynthesizerAgent(llm)
    refusal_judge = RefusalJudgeAgent(llm)
    single = SingleAgentResponder(llm)

    return {
        "single_agent": SingleAgentArchitecture(single),
        "sequential": SequentialArchitecture([
            extractor,
            relevance,
            conflict,
            critic,
            summarizer,
            synthesizer,
            refusal_judge,
        ]),
        "debate": DebateArchitecture(
            extractor=extractor,
            relevance=relevance,
            proposer=synthesizer,
            critic=critic,
            refiner=synthesizer,
            refusal_judge=refusal_judge,
        ),
        "parallel": ParallelArchitecture(
            extractor=extractor,
            relevance=relevance,
            conflict=conflict,
            critic=critic,
            synthesizer=synthesizer,
            refusal_judge=refusal_judge,
        ),
        "parallel_summarizer": ParallelSummarizerArchitecture(
            extractor=extractor,
            relevance=relevance,
            conflict=conflict,
            critic=critic,
            summarizer=summarizer,
            synthesizer=synthesizer,
            refusal_judge=refusal_judge,
        ),
    }
