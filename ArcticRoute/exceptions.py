from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(eq=False)
class ArcticRouteError(Exception):
    """Business-level exception carrying a stable error code for the UI."""

    code: str
    message: str
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        super().__init__(self.__str__())

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = f"[{self.code}] {self.message}"
        if self.detail:
            return f"{base}: {self.detail}"
        return base

    def to_dict(self) -> dict[str, Optional[str]]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


__all__ = ["ArcticRouteError"]
