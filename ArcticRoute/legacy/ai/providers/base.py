"""Base interfaces for AI providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract large-language-model client."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return a text completion for the given prompt."""


__all__ = ["LLMClient"]

