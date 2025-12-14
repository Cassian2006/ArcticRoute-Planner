"""Configuration loader for AI providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "ai.yaml"

ENV_MAP: Dict[str, str] = {
    "model_name": "AI_MODEL_NAME",
    "max_tokens": "AI_MAX_TOKENS",
    "temperature": "AI_TEMPERATURE",
    "timeout_s": "AI_TIMEOUT_S",
}


@dataclass
class AIConfig:
    model_name: str
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout_s: float = 30.0


def load_ai_config(path: Optional[Path] = None) -> AIConfig:
    if load_dotenv:
        load_dotenv()

    file_config: Dict[str, Any] = {}
    config_path = path or DEFAULT_CONFIG
    if config_path and config_path.exists():
        try:
            file_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to parse {config_path}: {exc}") from exc

    values: Dict[str, Any] = {}
    for key, env_name in ENV_MAP.items():
        env_value = os.getenv(env_name)
        if env_value is not None:
            values[key] = env_value
        elif key in file_config:
            values[key] = file_config[key]

    model_name = str(values.get("model_name") or file_config.get("model_name") or "").strip()
    if not model_name:
        raise RuntimeError("AI model_name must be configured via environment or config/ai.yaml")

    max_tokens = _to_int(values.get("max_tokens", file_config.get("max_tokens", AIConfig.max_tokens)))
    temperature = _to_float(values.get("temperature", file_config.get("temperature", AIConfig.temperature)))
    timeout_s = _to_float(values.get("timeout_s", file_config.get("timeout_s", AIConfig.timeout_s)))

    return AIConfig(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )


def _to_int(value: Any, *, default: int = AIConfig.max_tokens) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, *, default: float = AIConfig.temperature) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = ["AIConfig", "load_ai_config"]

