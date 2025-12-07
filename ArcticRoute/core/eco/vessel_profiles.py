# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "ArcticRoute" / "config" / "vessel_profiles.yaml"


def load_all_profiles() -> Dict[str, Dict[str, Any]]:
    if not PROFILES_PATH.exists():
        return {}
    try:
        obj = yaml.safe_load(PROFILES_PATH.read_text(encoding="utf-8")) or {}
        profs = obj.get("profiles") or {}
        if isinstance(profs, dict):
            return profs
        return {}
    except Exception:
        return {}


def load_vessel_profile(name: str) -> Dict[str, Any] | None:
    profs = load_all_profiles()
    if not profs:
        return None
    p = profs.get(name)
    if not p and profs:
        # fallback first
        k0 = next(iter(profs.keys()))
        p = profs.get(k0)
    return p

