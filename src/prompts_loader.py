from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_prompt(filename: str) -> str:
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    return (prompts_dir / filename).read_text(encoding="utf-8").strip()
