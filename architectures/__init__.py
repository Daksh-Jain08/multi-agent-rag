from architectures.debate import DebateArchitecture
from architectures.parallel import ParallelArchitecture
from architectures.parallel_summarizer import ParallelSummarizerArchitecture
from architectures.sequential import SequentialArchitecture
from architectures.single_agent import SingleAgentArchitecture

__all__ = [
    "SingleAgentArchitecture",
    "SequentialArchitecture",
    "DebateArchitecture",
    "ParallelArchitecture",
    "ParallelSummarizerArchitecture",
]
