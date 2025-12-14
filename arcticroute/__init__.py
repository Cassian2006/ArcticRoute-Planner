"""ArcticRoute package initialisation helpers."""

from __future__ import annotations

import sys as _sys

from .settings import settings  # noqa: F401

# 双向别名，确保历史模块名 "ArcticRoute" 可用
_sys.modules.setdefault("arcticroute", _sys.modules[__name__])
_sys.modules.setdefault("ArcticRoute", _sys.modules[__name__])

__all__ = ["settings"]






