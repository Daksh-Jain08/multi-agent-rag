from __future__ import annotations

from typing import Any, Dict, List

from architectures.base_arch import BaseArchitecture


class SequentialArchitecture(BaseArchitecture):
    name = "sequential"

    def __init__(self, agents: List[Any]) -> None:
        self.agents = agents

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        for agent in self.agents:
            state = agent.run(state)
        return state
