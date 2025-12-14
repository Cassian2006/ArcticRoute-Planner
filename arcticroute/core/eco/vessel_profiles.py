# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "ArcticRoute" / "config" / "vessel_profiles.yaml"


@dataclass
class VesselProfile:
    """船舶参数数据类。"""
    name: str
    design_speed_kn: float
    base_fuel_per_km: float
    dwt_min: float = 0.0
    dwt_max: float = 0.0
    label: str = ""
    description: str = ""
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> VesselProfile:
        """从字典创建 VesselProfile 实例。"""
        return cls(
            name=name,
            design_speed_kn=data.get("design_speed_kn", 13.0),
            base_fuel_per_km=data.get("base_fuel_per_km", 0.05),
            dwt_min=data.get("dwt_range", [0, 0])[0],
            dwt_max=data.get("dwt_range", [0, 0])[1],
            label=data.get("label", name),
            description=data.get("description", ""),
        )


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


def get_default_profiles() -> Dict[str, VesselProfile]:
    """获取所有默认船舶参数。"""
    profs = load_all_profiles()
    result = {}
    for name, data in profs.items():
        if isinstance(data, dict):
            result[name] = VesselProfile.from_dict(name, data)
    return result

