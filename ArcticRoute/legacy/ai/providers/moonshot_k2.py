"""Moonshot K2 provider implementation."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests

from .base import LLMClient

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_ENDPOINT = "/chat/completions"


class MoonshotK2Error(RuntimeError):
    """Raised when the Moonshot K2 client fails to obtain a completion."""


class MoonshotK2Client(LLMClient):
    """HTTP client for the Moonshot K2 API with retry and timeout handling."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        session: Optional[requests.Session] = None,
        sleep_fn: Any = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError("MOONSHOT_API_KEY is not configured")

        configured_base = base_url or os.getenv("MOONSHOT_BASE_URL", DEFAULT_BASE_URL)
        self.base_url = configured_base.rstrip("/")
        self.endpoint = os.getenv("MOONSHOT_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout = timeout if timeout is not None else float(os.getenv("MOONSHOT_TIMEOUT", "30"))
        self.max_retries = max(1, max_retries)
        self.backoff_factor = max(0.0, backoff_factor)
        self._session = session or requests.Session()
        self._sleep = sleep_fn or time.sleep
        self._last_meta: Dict[str, Any] = {}

    def complete(self, prompt: str, **kwargs: Any) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "stream" in kwargs:
            payload["stream"] = bool(kwargs["stream"])

        url = f"{self.base_url}{self.endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < self.max_retries:
            attempt += 1
            try:
                response = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._backoff(attempt)
                continue

            if 500 <= response.status_code < 600:
                last_error = MoonshotK2Error(f"server error {response.status_code}")
                if attempt >= self.max_retries:
                    break
                self._backoff(attempt)
                continue

            try:
                response.raise_for_status()
                data = response.json()
            except Exception as exc:  # pragma: no cover - defensive
                raise MoonshotK2Error(f"invalid response from Moonshot API: {exc}") from exc

            text = self._extract_text(data)
            if text is None:
                raise MoonshotK2Error("Moonshot API response missing completion text")
            self._last_meta = {
                "attempts": attempt,
                "timestamp": time.time(),
            }
            return text

        self._last_meta = {
            "attempts": attempt,
            "timestamp": time.time(),
            "error": str(last_error) if last_error else None,
        }
        raise MoonshotK2Error(f"Moonshot completion failed after {self.max_retries} attempts") from last_error

    def _backoff(self, attempt: int) -> None:
        delay = self.backoff_factor * (2 ** (attempt - 1))
        if delay > 0:
            self._sleep(delay)

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> Optional[str]:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                content = first.get("text")
                if isinstance(content, str):
                    return content
        return None


__all__ = ["MoonshotK2Client", "MoonshotK2Error"]
