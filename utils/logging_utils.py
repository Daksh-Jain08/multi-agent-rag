from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def trace_step(agent_name: str, update: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": now_iso(),
        "agent": agent_name,
        "update": update,
    }
