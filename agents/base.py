from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from utils.llm import LLMClient
from utils.logging_utils import trace_step


class RoleAgent(ABC):
    def __init__(self, name: str, llm: LLMClient, system_prompt: str) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt

    @abstractmethod
    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def apply_output(self, state: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def output_schema_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        raise NotImplementedError

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = self.build_user_prompt(state)
        parsed = self.llm.invoke_structured(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            schema_name=self.output_schema_name(),
            schema=self.output_schema(),
        )
        updated = self.apply_output(state, parsed)
        updated.setdefault("traces", []).append(trace_step(self.name, parsed))
        return updated
