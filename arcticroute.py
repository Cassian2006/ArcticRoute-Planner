"""Compatibility loader mapping the legacy ``ArcticRoute`` package to ``arcticroute``."""

from __future__ import annotations

import importlib
import sys

_pkg = importlib.import_module("ArcticRoute")
sys.modules[__name__] = _pkg
