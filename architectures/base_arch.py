from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseArchitecture(ABC):
    name: str

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
