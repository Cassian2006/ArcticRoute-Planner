from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CACHE_ENV_KEYS = ("CACHE_DIR", "CV_CACHE_DIR", "SAT_CACHE_DIR", "STAC_CACHE_DIR")
_EXTRA_ENV_VAR = "ARCTICROUTE_EXTRA_CACHE_DIRS"


def _normalise_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / value
    return path.resolve()


def discover_cache_roots(extra: Iterable[str] | None = None) -> List[Path]:
    """Return a de-duplicated list of cache directories to manage."""
    seen: Set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            resolved = path
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)

    add(PROJECT_ROOT / ".cache")
    add(PROJECT_ROOT / "cache")

    data_processed = PROJECT_ROOT / "data_processed"
    if data_processed.exists():
        for child in data_processed.iterdir():
            if child.is_dir() and "cache" in child.name.lower():
                add(child)

    for key in _CACHE_ENV_KEYS:
        value = os.getenv(key)
        if value:
            add(_normalise_path(value))

    env_extra = os.getenv(_EXTRA_ENV_VAR)
    if env_extra:
        for token in env_extra.split(os.pathsep):
            token = token.strip()
            if token:
                add(_normalise_path(token))

    if extra:
        for token in extra:
            if token:
                add(_normalise_path(token))

    return sorted(seen)


def to_relative(path: Path) -> str:
    """Render a path relative to the project root when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


__all__ = ["discover_cache_roots", "to_relative", "PROJECT_ROOT"]
