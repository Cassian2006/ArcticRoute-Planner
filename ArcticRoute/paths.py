"""Utilities for working with standardised project paths."""

from __future__ import annotations

from pathlib import Path

from .settings import settings


def _prepare_path(path_value: str) -> Path:
    """Normalise a path string and ensure the directory exists."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


DATA_DIR = _prepare_path(settings.DATA_DIR)
CACHE_DIR = _prepare_path(settings.CACHE_DIR)
LOG_DIR = _prepare_path(settings.LOG_DIR)

__all__ = ["DATA_DIR", "CACHE_DIR", "LOG_DIR"]
