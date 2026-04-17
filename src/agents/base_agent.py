import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class BaseAgent:
    def __init__(self, llm: Any):
        self.llm = llm
        self._usage_events: List[Dict[str, Any]] = []

    def _extract_token_usage(self, raw_message: Any) -> Dict[str, int]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if raw_message is None:
            return usage

        usage_meta = {}
        if hasattr(raw_message, "usage_metadata") and isinstance(raw_message.usage_metadata, dict):
            usage_meta = raw_message.usage_metadata
            usage["prompt_tokens"] = int(usage_meta.get("input_tokens", 0))
            usage["completion_tokens"] = int(usage_meta.get("output_tokens", 0))
            usage["total_tokens"] = int(
                usage_meta.get("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"])
            )
            return usage

        response_meta = {}
        if hasattr(raw_message, "response_metadata") and isinstance(raw_message.response_metadata, dict):
            response_meta = raw_message.response_metadata

        token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}
        if isinstance(token_usage, dict):
            usage["prompt_tokens"] = int(
                token_usage.get("prompt_tokens")
                or token_usage.get("input_tokens")
                or token_usage.get("prompt_token_count")
                or 0
            )
            usage["completion_tokens"] = int(
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
                or token_usage.get("candidates_token_count")
                or 0
            )
            usage["total_tokens"] = int(
                token_usage.get("total_tokens")
                or token_usage.get("total_token_count")
                or (usage["prompt_tokens"] + usage["completion_tokens"])
            )

        return usage

    def consume_usage_events(self) -> List[Dict[str, Any]]:
        events = self._usage_events[:]
        self._usage_events.clear()
        return events

    def _invoke_structured(self, schema: BaseModel, system_prompt: str, user_prompt: str) -> BaseModel:
        print(f"[LLM] Calling structured model for {schema.__name__}")
        # if not getattr(self.llm, "is_mock", False):
        #     time.sleep(15)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        raw_message = None
        try:
            model = self.llm.with_structured_output(schema, include_raw=True)
            structured_out = model.invoke(messages)
            if isinstance(structured_out, dict) and structured_out.get("parsed") is not None:
                output = structured_out["parsed"]
                raw_message = structured_out.get("raw")
            else:
                output = structured_out
        except TypeError:
            model = self.llm.with_structured_output(schema)
            output = model.invoke(messages)

        token_usage = self._extract_token_usage(raw_message)
        if token_usage["total_tokens"] <= 0:
            output_payload = output.model_dump() if hasattr(output, "model_dump") else str(output)
            prompt_text = f"{system_prompt}\n{user_prompt}"
            token_usage = {
                "prompt_tokens": self.get_token_count(prompt_text),
                "completion_tokens": self.get_token_count(json.dumps(output_payload, ensure_ascii=True)),
                "total_tokens": 0,
            }
            token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
        self._usage_events.append(
            {
                "schema": schema.__name__,
                "model": str(getattr(self.llm, "model", "unknown")),
                "timestamp": datetime.utcnow().isoformat(),
                "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(token_usage.get("completion_tokens", 0)),
                "total_tokens": int(token_usage.get("total_tokens", 0)),
            }
        )

        print(f"[LLM] Completed structured model for {schema.__name__}")
        return output

    @staticmethod
    def extract_json_section(prompt: str, section_name: str, fallback: Any = None) -> Any:
        import json

        pattern = rf"{re.escape(section_name)}:\n(.*?)(?:\n\n[A-Z][A-Za-z ]*:\n|\Z)"
        match = re.search(pattern, prompt, flags=re.DOTALL)
        if not match:
            return fallback
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            return fallback

    @staticmethod
    def extract_section(prompt: str, section_name: str) -> str:
        pattern = rf"{re.escape(section_name)}:\n(.*?)(?:\n\n[A-Z][A-Za-z ]*:\n|\Z)"
        match = re.search(pattern, prompt, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def get_token_count(text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(enc.encode(text))
        except Exception:
            return len(text.split())
