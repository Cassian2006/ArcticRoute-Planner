"""ArcticRoute package initialisation helpers."""

from __future__ import annotations

import sys as _sys

from .settings import settings  # noqa: F401

_sys.modules.setdefault("arcticroute", _sys.modules[__name__])

__all__ = ["settings"]
