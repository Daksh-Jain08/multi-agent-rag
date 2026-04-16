from __future__ import annotations

from typing import Any, Dict

from architectures.base_arch import BaseArchitecture


class SingleAgentArchitecture(BaseArchitecture):
    name = "single_agent"

    def __init__(self, responder) -> None:
        self.responder = responder

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.responder.run(state)
