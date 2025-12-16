# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "ArcticRoute" / "config" / "vessel_profiles.yaml"


class VesselProfile:
    """简化的船舶配置类（占位符）。"""
    
    def __init__(self, name: str, key: str, **kwargs):
        self.name = name
        self.key = key
        self.data = kwargs
    
    def get_effective_max_ice_thickness(self) -> float:
        """获取有效的最大冰厚（米）。"""
        return self.data.get("max_ice_thickness", 1.0)
    
    def __repr__(self):
        return f"VesselProfile(name={self.name}, key={self.key})"


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
    """获取默认的船舶配置字典。
    
    返回一个字典，键为船舶类型（如 "handy", "panamax"），
    值为 VesselProfile 对象。
    """
    # 默认配置：三种标准船舶类型
    default_profiles = {
        "handy": VesselProfile(
            name="Handysize",
            key="handy",
            max_ice_thickness=0.5,
        ),
        "panamax": VesselProfile(
            name="Panamax",
            key="panamax",
            max_ice_thickness=0.3,
        ),
        "ice_class": VesselProfile(
            name="Ice Class",
            key="ice_class",
            max_ice_thickness=2.0,
        ),
    }
    return default_profiles


def get_profile_by_key(key: str) -> VesselProfile | None:
    """按 key 获取船舶配置。"""
    profiles = get_default_profiles()
    return profiles.get(key)

