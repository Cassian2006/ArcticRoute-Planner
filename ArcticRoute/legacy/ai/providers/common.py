"""Common utilities for LLM providers (sanitisation, trimming, logging)."""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "ai_calls.log"

PATH_PATTERN = re.compile(r"^(?:[a-zA-Z]:\\|\\\\|/).+")
BANNED_TOKEN_PATTERN = re.compile(r"(sk-[0-9a-zA-Z]+|AKIA[0-9A-Z]{16}|ASI[0-9A-Z]{16})", re.IGNORECASE)


def trim_dict_for_llm(data: Dict[str, Any], max_chars: int = 6000) -> Dict[str, Any]:
    """Return a trimmed copy of data suitable for LLM prompts."""

    def _trim(value: Any, depth: int = 0) -> Any:
        if isinstance(value, dict):
            trimmed: Dict[str, Any] = {}
            for key in value:
                trimmed[key] = _trim(value[key], depth + 1)
            return trimmed
        if isinstance(value, list):
            if not value:
                return value
            limit = 5 if depth == 0 else 3
            trimmed_items = [_trim(item, depth + 1) for item in value[:limit]]
            if len(value) > limit:
                trimmed_items.append(f"...(+{len(value) - limit} items)")
            return trimmed_items
        if isinstance(value, str):
            if len(value) > 1024:
                return value[:1021] + "..."
            return value
        return value

    trimmed = _trim(deepcopy(data))
    if len(json.dumps(trimmed, ensure_ascii=False)) <= max_chars:
        return trimmed

    # If still too long, drop the largest entries until within limit.
    if isinstance(trimmed, dict):
        ordered_keys = sorted(
            trimmed,
            key=lambda k: len(json.dumps(trimmed[k], ensure_ascii=False)),
            reverse=True,
        )
        for key in ordered_keys:
            trimmed[key] = f"...omitted ({key})"
            if len(json.dumps(trimmed, ensure_ascii=False)) <= max_chars:
                break
    return trimmed


def redact_paths(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask absolute paths and host identifiers."""

    def _redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_redact(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if PATH_PATTERN.match(stripped):
                return "[redacted-path]"
            if "\\" in stripped and ":" in stripped:
                return "[redacted-path]"
            if re.search(r"[A-Za-z0-9_-]+@[A-Za-z0-9._-]+", stripped):
                return re.sub(r"[A-Za-z0-9._-]+@[A-Za-z0-9._-]+", "[redacted-email]", stripped)
        return value

    return _redact(deepcopy(data))


def add_safety_guardrails(text: str, *, max_length: int = 6000) -> str:
    """Enforce basic safety limits on LLM prompts."""
    result = text.strip()
    if len(result) > max_length:
        result = result[: max_length - 3] + "..."
    result = BANNED_TOKEN_PATTERN.sub("[redacted-token]", result)
    return result


def log_ai_call(
    action: str,
    *,
    duration_s: float,
    retries: int,
    degraded: bool,
    error: str | None = None,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": action,
        "duration_s": round(duration_s, 3),
        "retries": retries,
        "degraded": degraded,
    }
    if error:
        entry["error"] = error
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + os.linesep)


__all__ = [
    "trim_dict_for_llm",
    "redact_paths",
    "add_safety_guardrails",
    "log_ai_call",
]

