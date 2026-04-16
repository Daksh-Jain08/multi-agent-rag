from __future__ import annotations

import json
import logging
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class LLMClient:
    _rate_lock = threading.Lock()
    _next_allowed_ts = 0.0
    _semaphore: Optional[threading.Semaphore] = None
    _logger: Optional[logging.Logger] = None

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("MODEL_NAME", "gemini-2.5-flash")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Set it in .env or environment.")
        self.client = genai.Client(api_key=api_key)
        self.requests_per_minute = max(int(os.getenv("LLM_REQUESTS_PER_MINUTE", "5")), 1)
        self.min_interval_seconds = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", str(60.0 / self.requests_per_minute)))

        if LLMClient._semaphore is None:
            max_concurrent = max(int(os.getenv("LLM_MAX_CONCURRENT_CALLS", "1")), 1)
            LLMClient._semaphore = threading.Semaphore(max_concurrent)

        if LLMClient._logger is None:
            log_path = Path(os.getenv("LLM_LOG_PATH", "outputs/logs/llm_calls.log"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger = logging.getLogger("llm_client")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.FileHandler(log_path, encoding="utf-8")
                formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.propagate = False
            LLMClient._logger = logger

    def invoke(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, retries: int = 2) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                start = time.monotonic()
                self._log_call_start(kind="plain", attempt=attempt)
                with self._acquire_slot_and_wait():
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                        ),
                    )
                elapsed = time.monotonic() - start
                self._log_call_success(kind="plain", attempt=attempt, elapsed=elapsed)
                return getattr(response, "text", None) or ""
            except Exception as exc:  # pragma: no cover - network/API behavior
                last_error = exc
                self._log_call_error(kind="plain", attempt=attempt, error=exc)
                if attempt < retries:
                    time.sleep(1 + attempt)

        raise RuntimeError(f"LLM call failed after retries: {last_error}")

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        temperature: float = 0.0,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """Invoke the model with a strict JSON schema and return parsed JSON."""
        last_error: Optional[Exception] = None
        sanitized_schema = _sanitize_schema_for_gemini(schema)
        for attempt in range(retries + 1):
            try:
                start = time.monotonic()
                self._log_call_start(kind=f"structured:{schema_name}", attempt=attempt)
                with self._acquire_slot_and_wait():
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                            response_mime_type="application/json",
                            response_schema=sanitized_schema,
                        ),
                    )
                text = getattr(response, "text", None) or "{}"
                parsed = parse_json_response_safe(text)
                if parsed is not None:
                    elapsed = time.monotonic() - start
                    self._log_call_success(kind=f"structured:{schema_name}", attempt=attempt, elapsed=elapsed)
                    return parsed
                raise RuntimeError(f"Structured output parse failed for schema {schema_name}")
            except Exception as exc:  # pragma: no cover - network/API behavior
                last_error = exc
                self._log_call_error(kind=f"structured:{schema_name}", attempt=attempt, error=exc)
                if attempt < retries:
                    time.sleep(1 + attempt)

        raise RuntimeError(f"Structured LLM call failed after retries: {last_error}")

    def _acquire_slot_and_wait(self):
        class _RateLimiterContext:
            def __init__(self, outer: "LLMClient") -> None:
                self.outer = outer

            def __enter__(self):
                assert LLMClient._semaphore is not None
                LLMClient._semaphore.acquire()
                with LLMClient._rate_lock:
                    now = time.monotonic()
                    wait_time = max(0.0, LLMClient._next_allowed_ts - now)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    LLMClient._next_allowed_ts = time.monotonic() + self.outer.min_interval_seconds

            def __exit__(self, exc_type, exc, tb):
                assert LLMClient._semaphore is not None
                LLMClient._semaphore.release()

        return _RateLimiterContext(self)

    def _log_call_start(self, kind: str, attempt: int) -> None:
        if LLMClient._logger is None:
            return
        LLMClient._logger.info(
            "event=llm_call_start model=%s kind=%s attempt=%s rpm=%s min_interval=%.3f",
            self.model,
            kind,
            attempt,
            self.requests_per_minute,
            self.min_interval_seconds,
        )

    def _log_call_success(self, kind: str, attempt: int, elapsed: float) -> None:
        if LLMClient._logger is None:
            return
        LLMClient._logger.info(
            "event=llm_call_success model=%s kind=%s attempt=%s elapsed=%.3f",
            self.model,
            kind,
            attempt,
            elapsed,
        )

    def _log_call_error(self, kind: str, attempt: int, error: Exception) -> None:
        if LLMClient._logger is None:
            return
        error_str = str(error).replace("\n", " ")
        LLMClient._logger.error(
            "event=llm_call_error model=%s kind=%s attempt=%s error=%s",
            self.model,
            kind,
            attempt,
            error_str,
        )


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON content with a small fallback for fenced blocks."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def parse_json_response_safe(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = parse_json_response(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _sanitize_schema_for_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove JSON Schema keys not accepted by Gemini's response_schema payload."""
    sanitized = deepcopy(schema)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            node.pop("additional_properties", None)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(sanitized)
    return sanitized
