"""Helper to resolve CLI module path regardless of working directory.

@role: core
"""

"""Helpers to resolve the CLI module regardless of the launching directory."""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
PARENT = ROOT.parent

_CLI_MOD: Optional[str] = None
_PATH_INITIALISED = False


def ensure_path() -> None:
    """Ensure the project parent directory is available on sys.path."""
    global _PATH_INITIALISED
    if _PATH_INITIALISED:
        return
    parent_str = str(PARENT)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
    _PATH_INITIALISED = True


def get_cli_mod() -> str:
    """Return the import path for the CLI module, adding fallbacks as needed."""
    global _CLI_MOD
    if _CLI_MOD is not None:
        return _CLI_MOD
    try:
        import_module("ArcticRoute.api.cli")
    except ModuleNotFoundError:
        root_str = str(ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        import_module("api.cli")
        _CLI_MOD = "api.cli"
    else:
        _CLI_MOD = "ArcticRoute.api.cli"
    return _CLI_MOD

